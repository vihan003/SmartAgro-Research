"""Experiment C: next-day fire-DETECTION risk per grid cell (NASA FIRMS VIIRS + NASA POWER daily weather).

WHAT A POSITIVE MEANS: at least one non-low-confidence VIIRS detection in the cell on day t+1.
It is NOT a confirmed agricultural fire. Without a cropland mask the label is "satellite fire detection in an
agricultural region and season". Use --cropland-mask (CSV: cell_id) to restrict to cropland cells.

PREDICTION TIME: end of day t. Features use only data from days <= t (observed weather, antecedent detections).
Observed weather is an optimistic stand-in for a real forecast; say so in the paper.

SPLIT: time-based. train = years <= --train-end, val = --val-year, test = years > --val-year. No shuffling.

USAGE (needs internet + a free FIRMS MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/):
  python exp_c_fire_risk.py --map-key KEY --years 2017-2023 --out results_c
LOGIC TEST WITHOUT DATA (synthetic frames; proves the pipeline runs, says nothing about the real world):
  python exp_c_fire_risk.py --mock --years 2017-2023 --out results_c_mock
"""
import argparse, json, os, sys, time, platform
import numpy as np, pandas as pd

BBOX = (74.0, 28.4, 77.6, 32.6)           # west, south, east, north (Punjab/Haryana region; edit as needed)
DLAT, DLON = 0.5, 0.625                    # MERRA-2 native grid used by NASA POWER meteorology
SEASON = (10, 11)                          # October-November (post-harvest residue burning season)

def cell_id(lat, lon):
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_cell = (np.floor(lat_arr / DLAT) * DLAT + DLAT / 2).round(3)
    lon_cell = (np.floor(lon_arr / DLON) * DLON + DLON / 2).round(3)
    out = np.char.add(lat_cell.astype(str), "_")
    out = np.char.add(out, lon_cell.astype(str))
    return out.item() if np.asarray(out).shape == () else out

def cells():
    out = []
    for la in np.arange(np.floor(BBOX[1] / DLAT) * DLAT + DLAT / 2, BBOX[3], DLAT):
        for lo in np.arange(np.floor(BBOX[0] / DLON) * DLON + DLON / 2, BBOX[2], DLON):
            out.append((round(la, 3), round(lo, 3)))
    return out

# ----------------------------------------------------------------------------- data access
def fetch_firms(key, years, cache):
    import requests, io
    frames = []
    for y in years:
        d = pd.Timestamp(f"{y}-{SEASON[0]:02d}-01")
        end = pd.Timestamp(f"{y}-{SEASON[1]:02d}-30")
        while d <= end:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_SP/"
                   f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}/5/{d:%Y-%m-%d}")
            fn = os.path.join(cache, f"firms_{d:%Y%m%d}.csv")
            if not os.path.exists(fn):
                r = requests.get(url, timeout=120); r.raise_for_status(); open(fn, "w").write(r.text); time.sleep(0.3)
            frames.append(pd.read_csv(fn)); d += pd.Timedelta(days=5)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df[df["confidence"].astype(str).str.lower() != "l"]

def fetch_power(years, cache):
    import requests
    rows = []
    for la, lo in cells():
        fn = os.path.join(cache, f"power_{la}_{lo}.json")
        if not os.path.exists(fn):
            url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M,PRECTOTCORR"
                   f"&community=AG&longitude={lo}&latitude={la}&start={min(years)}0101&end={max(years)}1231&format=JSON")
            r = requests.get(url, timeout=180); r.raise_for_status(); open(fn, "w").write(r.text); time.sleep(0.5)
        p = json.load(open(fn))["properties"]["parameter"]
        d = pd.DataFrame(p); d.index = pd.to_datetime(d.index, format="%Y%m%d"); d = d.replace(-999, np.nan)
        d["cell"] = cell_id(np.array([la]), np.array([lo]))[0]; d["lat"], d["lon"] = la, lo
        rows.append(d.reset_index().rename(columns={"index": "date"}))
    return pd.concat(rows, ignore_index=True)

def mock(years, seed=0):
    """Synthetic stand-ins with the same columns. FOR TESTING THE CODE ONLY."""
    r = np.random.default_rng(seed); wx, fires = [], []
    for la, lo in cells():
        for y in years:
            for d in pd.date_range(f"{y}-01-01", f"{y}-12-31"):
                s = np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
                wx.append((d, 28 + 8 * s + r.normal(0, 2), np.clip(50 - 20 * s + r.normal(0, 10), 5, 99),
                           abs(r.normal(2.5, 1)), max(0, r.normal(1, 4)) * (r.random() < .2), la, lo))
    wx = pd.DataFrame(wx, columns=["date", "T2M", "RH2M", "WS2M", "PRECTOTCORR", "lat", "lon"])
    wx["cell"] = cell_id(wx["lat"].values, wx["lon"].values)
    p = 1 / (1 + np.exp(-(-6 + .15 * (wx["T2M"] - 28) - .05 * (wx["RH2M"] - 45) + .3 * wx["WS2M"])))
    hit = wx[(r.random(len(wx)) < p) & wx["date"].dt.month.isin(SEASON)]
    fires = pd.DataFrame({"latitude": hit["lat"], "longitude": hit["lon"], "acq_date": hit["date"].dt.strftime("%Y-%m-%d"), "confidence": "n"})
    return fires, wx

# ----------------------------------------------------------------------------- dataset construction
def build(fires, wx):
    fires = fires.copy(); fires["date"] = pd.to_datetime(fires["acq_date"]); fires["cell"] = cell_id(fires["latitude"].values, fires["longitude"].values)
    cnt = fires.groupby(["cell", "date"]).size().rename("n_det").reset_index()
    df = wx.merge(cnt, on=["cell", "date"], how="left").fillna({"n_det": 0}).sort_values(["cell", "date"])
    df = df[df["date"].dt.month.isin(SEASON)].copy() if False else df       # keep full calendar for lags, filter later
    g = df.groupby("cell")
    df["det_today"] = (df["n_det"] > 0).astype(int)
    df["det_prev1"] = g["det_today"].shift(1); df["det_prev3"] = g["det_today"].transform(lambda s: s.rolling(3).sum().shift(1))
    df["det_prev7"] = g["det_today"].transform(lambda s: s.rolling(7).sum().shift(1))
    df["rain3"] = g["PRECTOTCORR"].transform(lambda s: s.rolling(3).sum())
    df["dry_days"] = g["PRECTOTCORR"].transform(lambda s: (s.fillna(0) < 1).astype(int).groupby((s.fillna(0) >= 1).cumsum()).cumsum())
    df["doy"] = df["date"].dt.dayofyear
    df["y"] = g["det_today"].shift(-1)                                       # label = detection on day t+1
    df = df[df["date"].dt.month.isin(SEASON)].dropna(subset=["y", "T2M", "RH2M", "WS2M", "det_prev7"])
    df["y"] = df["y"].astype(int); df["year"] = df["date"].dt.year
    return df

FEATS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR", "rain3", "dry_days", "det_today", "det_prev3", "det_prev7", "doy", "lat", "lon"]

def make_time_split(df, train_end=None, val_year=None):
    years = sorted(df["year"].dropna().unique())
    if len(years) >= 3 and train_end is not None and val_year is not None:
        tr = df[df.year <= train_end].copy()
        va = df[df.year == val_year].copy()
        te = df[df.year > val_year].copy()
        if len(tr) and len(va) and len(te):
            return tr, va, te, "year-based"

    d = df.sort_values("date").reset_index(drop=True)
    dates = sorted(d["date"].drop_duplicates().tolist())
    n = len(dates)
    if n < 3:
        raise ValueError("Need at least 3 unique dates in the dataset to build a temporal split.")
    tr_end = max(1, int(np.floor(n * 0.7)))
    va_end = max(tr_end + 1, int(np.ceil(n * 0.85)))
    if va_end >= n:
        va_end = n - 1
    if tr_end >= va_end:
        tr_end = max(1, va_end - 1)
    tr_dates = set(dates[:tr_end])
    va_dates = set(dates[tr_end:va_end])
    te_dates = set(dates[va_end:])
    tr = d[d["date"].isin(tr_dates)].copy()
    va = d[d["date"].isin(va_dates)].copy()
    te = d[d["date"].isin(te_dates)].copy()
    if not (len(tr) and len(va) and len(te)):
        tr = d[d["date"].isin(set(dates[:-2]))].copy(); va = d[d["date"].isin(set(dates[-2:-1]))].copy(); te = d[d["date"].isin(set(dates[-1:]))].copy()
    return tr, va, te, "fallback_temporal_fraction"

def evaluate(df, train_end, val_year, outdir, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import average_precision_score, brier_score_loss
    tr, va, te, split_mode = make_time_split(df, train_end, val_year)
    assert len(tr) and len(va) and len(te), "empty split: check --years/--train-end/--val-year"
    assert tr.date.max() < va.date.min() <= va.date.max() < te.date.min(), "temporal leakage between splits"
    info = {"rows": {k: int(len(v)) for k, v in dict(train=tr, val=va, test=te).items()},
            "positives": {k: int(v.y.sum()) for k, v in dict(train=tr, val=va, test=te).items()},
            "base_rate_test": float(te.y.mean()), "features": FEATS, "seed": seed, "split_mode": split_mode}
    lrm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[FEATS].fillna(0), tr.y)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=.08, max_iter=150, random_state=seed).fit(tr[FEATS], tr.y)
    z = lambda d: ((d["T2M"] - tr["T2M"].mean()) / tr["T2M"].std() - (d["RH2M"] - tr["RH2M"].mean()) / tr["RH2M"].std()
                   + (d["WS2M"] - tr["WS2M"].mean()) / tr["WS2M"].std())
    clim = tr.groupby("doy").y.mean()
    scorers = {"persistence (fire yesterday)": lambda d: d["det_today"].values.astype(float),
               "climatology (day-of-season rate)": lambda d: d["doy"].map(clim).fillna(tr.y.mean()).values,
               "weather index (hot, dry, windy)": lambda d: z(d).values,
               "logistic regression": lambda d: lrm.predict_proba(d[FEATS].fillna(0))[:, 1],
               "gradient-boosted trees": lambda d: gbm.predict_proba(d[FEATS])[:, 1]}
    rng = np.random.default_rng(seed); res = {}
    te = te.reset_index(drop=True); te["wk"] = (te["year"].astype(str) + "-" + (te["date"].dt.dayofyear // 7).astype(str))
    weeks = te["wk"].unique()
    for name, f in scorers.items():
        sv, st = f(va), f(te); ap = average_precision_score(te.y, st)
        o = np.argsort(-sv); cum = np.cumsum(va.y.values[o]) / max(va.y.sum(), 1); thr = sv[o][min(np.searchsorted(cum, .8), len(sv) - 1)]
        al = st >= thr; tp = int((al & (te.y == 1)).sum()); fp = int((al & (te.y == 0)).sum()); fn = int(((~al) & (te.y == 1)).sum())
        boots = []
        for _ in range(200):                                              # block bootstrap over test weeks (respects serial dependence)
            idx = np.concatenate([np.where(te.wk.values == w)[0] for w in rng.choice(weeks, len(weeks))])
            boots.append(average_precision_score(te.y.values[idx], st[idx]) if te.y.values[idx].sum() else np.nan)
        res[name] = {"AUPRC": float(ap), "AUPRC_CI95_block_bootstrap": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "missed_event_rate": fn / max(tp + fn, 1),
                     "false_alarms_per_100_negative_cell_days": 100 * fp / max(int((te.y == 0).sum()), 1),
                     "alert_share_pct": float(100 * al.mean()),
                     "brier": float(brier_score_loss(te.y, np.clip(st, 0, 1))) if name in ("logistic regression", "gradient-boosted trees", "climatology (day-of-season rate)") else None}
    json.dump({"setup": info, "models": res}, open(os.path.join(outdir, "results_c.json"), "w"), indent=1)
    return info, res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--map-key"); ap.add_argument("--years", default="2017-2023"); ap.add_argument("--out", default="results_c")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--train-end", type=int, default=2020); ap.add_argument("--val-year", type=int, default=2021)
    ap.add_argument("--cropland-mask"); a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-")); years = list(range(y0, y1 + 1)); os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)
    if a.mock: fires, wx = mock(years); print("MOCK DATA: logic test only")
    else:
        if not a.map_key: sys.exit("need --map-key (or use --mock)")
        fires, wx = fetch_firms(a.map_key, years, cache), fetch_power(years, cache)
    df = build(fires, wx)
    if a.cropland_mask: df = df[df.cell.isin(set(pd.read_csv(a.cropland_mask)["cell_id"]))]
    info, res = evaluate(df, a.train_end, a.val_year, a.out)
    log = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "mock": a.mock, "years": years,
           "bbox": BBOX, "grid": [DLAT, DLON], "season_months": SEASON, "accessed_utc": pd.Timestamp.utcnow().isoformat()}
    json.dump(log, open(os.path.join(a.out, "run_log.json"), "w"), indent=1)
    print(json.dumps(info, indent=1)); print(pd.DataFrame(res).T.round(3).to_string())
