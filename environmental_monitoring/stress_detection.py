from __future__ import annotations

from typing import Iterable

DEFAULT_THRESHOLDS = {
    "soil_moisture_low_pct": 35,
    "soil_moisture_high_pct": 85,
    "temperature_high_c": 35,
    "temperature_low_c": 10,
    "humidity_high_pct": 85,
    "soil_ph_low": 5.5,
    "soil_ph_high": 7.5,
}


def stress_for_row(row: dict, thresholds: dict | None = None) -> dict:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts = []
    score = 0

    soil = row.get("soil_moisture_pct")
    if soil is not None:
        if soil < t["soil_moisture_low_pct"]:
            alerts.append({"type": "Low soil moisture", "severity": "high", "reason": f"Soil moisture is {soil:.1f}% below the configurable {t['soil_moisture_low_pct']}% threshold.", "action": "Check irrigation availability and moisture at the crop root zone."})
            score += 3
        elif soil > t["soil_moisture_high_pct"]:
            alerts.append({"type": "Excess soil moisture", "severity": "moderate", "reason": f"Soil moisture is {soil:.1f}%, above the configurable {t['soil_moisture_high_pct']}% threshold.", "action": "Check drainage and avoid additional watering until conditions improve."})
            score += 2

    temp = row.get("temperature_c")
    if temp is not None:
        if temp >= t["temperature_high_c"]:
            alerts.append({"type": "Heat stress risk", "severity": "high", "reason": f"Air temperature is {temp:.1f}°C, at or above {t['temperature_high_c']}°C.", "action": "Inspect crop canopy and consider field-level heat protection measures."})
            score += 3
        elif temp <= t["temperature_low_c"]:
            alerts.append({"type": "Cold stress risk", "severity": "moderate", "reason": f"Air temperature is {temp:.1f}°C, at or below {t['temperature_low_c']}°C.", "action": "Inspect sensitive crops and consider local cold-risk protection."})
            score += 2

    humidity = row.get("humidity_pct")
    if humidity is not None and humidity >= t["humidity_high_pct"]:
        alerts.append({"type": "Excessive humidity", "severity": "moderate", "reason": f"Relative humidity is {humidity:.1f}%, above {t['humidity_high_pct']}%.", "action": "Monitor canopy conditions and fungal disease symptoms conservatively."})
        score += 2

    ph = row.get("soil_ph")
    if ph is not None:
        if ph < t["soil_ph_low"]:
            alerts.append({"type": "Low soil pH", "severity": "moderate", "reason": f"Soil pH is {ph:.2f}, below {t['soil_ph_low']}.", "action": "Confirm with a field soil test before applying corrective amendments."})
            score += 2
        elif ph > t["soil_ph_high"]:
            alerts.append({"type": "High soil pH", "severity": "moderate", "reason": f"Soil pH is {ph:.2f}, above {t['soil_ph_high']}.", "action": "Confirm with a field soil test before any corrective treatment."})
            score += 2

    if score >= 5:
        category, severity = "High stress", "high"
    elif score >= 3:
        category, severity = "Moderate stress", "moderate"
    elif score >= 1:
        category, severity = "Low stress", "low"
    else:
        category, severity = "Normal", "normal"

    if len(alerts) >= 2:
        alerts.append({"type": "Combined environmental stress", "severity": "high", "reason": "Multiple abnormal environmental conditions occurred together.", "action": "Prioritize field inspection; do not rely on this dashboard alone for intervention decisions."})

    return {"category": category, "severity": severity, "score": score, "alerts": alerts,
            "timestamp": row.get("timestamp")}
