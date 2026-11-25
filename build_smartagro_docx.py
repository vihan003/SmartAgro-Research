from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from xml.sax.saxutils import escape
import re
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "outputs" / "SmartAgro_Research_Paper.html"
DOCX_PATH = ROOT / "outputs" / "SmartAgro_Research_Paper.docx"
EMU_PER_INCH = 914400


class Element:
    def __init__(self, tag: str, attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.children = []


class TreeBuilder(HTMLParser):
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Element("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        el = Element(tag, attrs)
        self.stack[-1].children.append(el)
        if tag not in self.void:
            self.stack.append(el)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Element(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if data:
            self.stack[-1].children.append(data)


def cls(el: Element) -> set[str]:
    return set(el.attrs.get("class", "").split())


def find_first(node: Element, tag: str):
    for child in node.children:
        if isinstance(child, Element):
            if child.tag == tag:
                return child
            found = find_first(child, tag)
            if found is not None:
                return found
    return None


def plain_text(children) -> str:
    out = []
    for child in children:
        if isinstance(child, str):
            out.append(child)
        elif child.tag == "br":
            out.append(" ")
        else:
            out.append(plain_text(child.children))
    return re.sub(r"\s+", " ", "".join(out)).strip()


class DocxBuilder:
    def __init__(self, root: Element):
        self.root = root
        self.media = {}
        self.relationships = []
        self.next_rid = 1
        self.hyperlinks = {}
        self.paragraphs = 0
        self.tables = 0
        self.images = 0

    def rid(self, rel_type: str, target: str, external=False) -> str:
        rid = f"rId{self.next_rid}"
        self.next_rid += 1
        mode = ' TargetMode="External"' if external else ""
        self.relationships.append(
            f'<Relationship Id="{rid}" Type="{rel_type}" Target="{escape(target)}"{mode}/>'
        )
        return rid

    def hyperlink_rid(self, url: str) -> str:
        if url not in self.hyperlinks:
            self.hyperlinks[url] = self.rid(
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                url,
                external=True,
            )
        return self.hyperlinks[url]

    def run_xml(self, value: str, formatting=None) -> str:
        if not value:
            return ""
        formatting = formatting or {}
        props = []
        if formatting.get("bold"):
            props.append("<w:b/>")
        if formatting.get("italic"):
            props.append("<w:i/>")
        if formatting.get("underline"):
            props.append('<w:u w:val="single"/>')
        if formatting.get("color"):
            props.append(f'<w:color w:val="{formatting["color"]}"/>')
        if formatting.get("font"):
            f = escape(formatting["font"])
            props.append(f'<w:rFonts w:ascii="{f}" w:hAnsi="{f}" w:eastAsia="{f}"/>')
        if formatting.get("size"):
            props.append(f'<w:sz w:val="{int(round(formatting["size"] * 2))}"/>')
            props.append(f'<w:szCs w:val="{int(round(formatting["size"] * 2))}"/>')
        if formatting.get("sup"):
            props.append('<w:vertAlign w:val="superscript"/>')
        if formatting.get("sub"):
            props.append('<w:vertAlign w:val="subscript"/>')
        rpr = f'<w:rPr>{"".join(props)}</w:rPr>' if props else ""
        return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(value)}</w:t></w:r>'

    def inline(self, children, formatting=None, in_link=None) -> str:
        formatting = dict(formatting or {})
        out = []
        for child in children:
            if isinstance(child, str):
                run = self.run_xml(child, formatting)
                if in_link:
                    out.append(f'<w:hyperlink r:id="{in_link}" w:history="1">{run}</w:hyperlink>')
                else:
                    out.append(run)
                continue
            if child.tag == "br":
                out.append("<w:r><w:br/></w:r>")
                continue
            f = dict(formatting)
            if child.tag in ("b", "strong"):
                f["bold"] = True
            elif child.tag in ("i", "em"):
                f["italic"] = True
            elif child.tag == "u":
                f["underline"] = True
            elif child.tag == "sup":
                f["sup"] = True
            elif child.tag == "sub":
                f["sub"] = True
            link = in_link
            if child.tag == "a" and child.attrs.get("href", "").startswith(("https://", "http://", "mailto:")):
                link = self.hyperlink_rid(child.attrs["href"])
            out.append(self.inline(child.children, f, link))
        return "".join(out)

    def p_xml(self, children, style="Normal", align=None, before=0, after=120,
              line=276, keep=False, left=0, first=0, shading=None, border_left=None,
              font=None, size=None, color=None, bold=False, italic=False):
        self.paragraphs += 1
        ppr = [f'<w:pStyle w:val="{style}"/>',
               f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>']
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        if keep:
            ppr.append("<w:keepNext/>")
        if left or first:
            ppr.append(f'<w:ind w:left="{left}" w:firstLine="{first}"/>')
        if shading:
            ppr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
        if border_left:
            ppr.append(f'<w:pBdr><w:left w:val="single" w:sz="20" w:space="7" w:color="{border_left}"/></w:pBdr>')
        run_format = {}
        if font: run_format["font"] = font
        if size: run_format["size"] = size
        if color: run_format["color"] = color
        if bold: run_format["bold"] = True
        if italic: run_format["italic"] = True
        content = self.inline(children, run_format)
        if not content:
            content = "<w:r><w:t></w:t></w:r>"
        return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{content}</w:p>'

    def page_break(self) -> str:
        self.paragraphs += 1
        return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

    def image_xml(self, src: str, alt: str, max_width=6.18, max_height=6.1) -> str:
        source = (HTML_PATH.parent / src).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        key = source.name
        if key not in self.media:
            self.media[key] = source.read_bytes()
        rid = self.rid("http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"media/{key}")
        with Image.open(source) as im:
            width_px, height_px = im.size
        width_in = min(max_width, max_height * width_px / height_px)
        height_in = width_in * height_px / width_px
        cx = int(width_in * EMU_PER_INCH)
        cy = int(height_in * EMU_PER_INCH)
        self.images += 1
        alt = escape(alt or source.stem)
        drawing = (
            '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self.images}" name="{escape(source.name)}" descr="{alt}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="image"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            '</a:graphicData></a:graphic></wp:inline></w:drawing>'
        )
        self.paragraphs += 1
        return f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="40"/></w:pPr><w:r>{drawing}</w:r></w:p>'

    def table_xml(self, table: Element) -> str:
        self.tables += 1
        rows = [el for el in table.children if isinstance(el, Element) and el.tag in ("thead", "tbody", "tfoot", "tr")]
        flat = []
        for group in rows:
            if group.tag == "tr":
                flat.append(group)
            else:
                flat.extend([el for el in group.children if isinstance(el, Element) and el.tag == "tr"])
        if not flat:
            return ""
        col_count = max(1, max(sum(1 for c in row.children if isinstance(c, Element) and c.tag in ("td", "th")) for row in flat))
        grid = "<w:tblGrid>" + "".join('<w:gridCol w:w="1700"/>' for _ in range(col_count)) + "</w:tblGrid>"
        props = (
            '<w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblLayout w:type="autofit"/>'
            '<w:tblCellMar><w:top w:w="55" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
            '<w:bottom w:w="55" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>'
            '<w:tblBorders><w:top w:val="single" w:sz="5" w:color="CBD5DC"/>'
            '<w:left w:val="single" w:sz="5" w:color="CBD5DC"/><w:bottom w:val="single" w:sz="5" w:color="CBD5DC"/>'
            '<w:right w:val="single" w:sz="5" w:color="CBD5DC"/><w:insideH w:val="single" w:sz="4" w:color="CBD5DC"/>'
            '<w:insideV w:val="single" w:sz="4" w:color="CBD5DC"/></w:tblBorders></w:tblPr>'
        )
        row_xml = []
        for ri, row in enumerate(flat):
            cells = [el for el in row.children if isinstance(el, Element) and el.tag in ("td", "th")]
            cell_xml = []
            for cell in cells:
                is_header = cell.tag == "th" or ri == 0
                fill = "18324B" if is_header else ("F4F7F8" if ri % 2 == 0 else "FFFFFF")
                text_color = "FFFFFF" if is_header else "202C36"
                fsize = 7.6 if "class-table" in cls(table) else 7.9
                para = self.p_xml(cell.children, style="TableText", align="left", before=0, after=0,
                                  line=220, font="Arial", size=fsize, color=text_color, bold=is_header)
                cell_xml.append(
                    f'<w:tc><w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
                    f'<w:tcW w:w="{max(900, 9600 // col_count)}" w:type="dxa"/></w:tcPr>{para}</w:tc>'
                )
            trpr = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader w:val="true"/>' if ri == 0 else '') + '</w:trPr>'
            row_xml.append(f'<w:tr>{trpr}{"".join(cell_xml)}</w:tr>')
        return f'<w:tbl>{props}{grid}{"".join(row_xml)}</w:tbl>'

    def paragraph_for(self, node: Element, title_context=False) -> str:
        classes = cls(node)
        if node.tag == "h1":
            return self.p_xml(node.children, style="Title", align="left", before=180, after=260,
                              line=560, keep=True, font="Times New Roman", size=21,
                              color="18324B", bold=True)
        if node.tag == "h2":
            is_abstract = "abstracthead" in classes
            return self.p_xml(node.children, style="Heading1", align="left", before=300 if not is_abstract else 220,
                              after=120, line=300, keep=True, font="Arial", size=13.5,
                              color="18324B", bold=True)
        if node.tag == "figcaption":
            return self.p_xml(node.children, style="Caption", align="both", before=0, after=180,
                              line=230, font="Arial", size=8.2, color="344653")
        if node.tag != "p":
            return ""
        if "eyebrow" in classes:
            return self.p_xml(node.children, style="Normal", align="left", before=280, after=180,
                              line=240, font="Arial", size=8.8, color="27877B", bold=True)
        if "authors" in classes:
            return self.p_xml(node.children, style="Normal", align="left", before=180, after=60,
                              line=250, font="Arial", size=10.8, color="18324B", bold=True)
        if "affiliation" in classes:
            return self.p_xml(node.children, style="Normal", align="left", before=0, after=250,
                              line=230, font="Arial", size=8.8, color="56636D")
        if "status" in classes:
            return self.p_xml(node.children, style="Normal", align="both", before=100, after=170,
                              line=260, left=180, font="Arial", size=9, color="344653",
                              shading="EEF3F5", border_left="D89B2B")
        if "abstract" in classes:
            return self.p_xml(node.children, style="Normal", align="both", before=0, after=115,
                              line=280, size=10.2)
        if "keywords" in classes or "counting" in classes:
            return self.p_xml(node.children, style="Normal", align="left", before=100, after=100,
                              line=230, font="Arial", size=8.8, color="43515D")
        if "tablecap" in classes:
            return self.p_xml(node.children, style="TableCaption", align="left", before=90, after=45,
                              line=230, font="Arial", size=8.5, color="18324B")
        if "reference" in classes:
            return self.p_xml(node.children, style="Reference", align="left", before=0, after=135,
                              line=255, left=420, first=-420, size=9.0)
        return self.p_xml(node.children, style="Normal", align="both", before=0, after=135, line=280)

    def process(self, node: Element, title_context=False) -> str:
        if node.tag == "section" and "titlepage" in cls(node):
            content = "".join(self.process(ch, title_context=True) for ch in node.children if isinstance(ch, Element))
            return content + self.page_break()
        if node.tag == "div" and ("references-break" in cls(node) or "appendix-break" in cls(node)):
            return self.page_break()
        if node.tag == "figure":
            img = find_first(node, "img")
            result = ""
            if img is not None:
                result += self.image_xml(img.attrs.get("src", ""), img.attrs.get("alt", ""))
            caption = find_first(node, "figcaption")
            if caption is not None:
                result += self.paragraph_for(caption)
            return result
        if node.tag == "div" and "table-wrap" in cls(node):
            out = []
            for ch in node.children:
                if not isinstance(ch, Element):
                    continue
                if ch.tag == "p": out.append(self.paragraph_for(ch))
                elif ch.tag == "table": out.append(self.table_xml(ch))
            return "".join(out)
        if node.tag == "table":
            return self.table_xml(node)
        if node.tag in ("p", "h1", "h2", "figcaption"):
            return self.paragraph_for(node, title_context)
        if node.tag in ("body", "root"):
            return "".join(self.process(ch, title_context) for ch in node.children if isinstance(ch, Element))
        if node.tag in ("html", "head"):
            return "".join(self.process(ch, title_context) for ch in node.children if isinstance(ch, Element)) if node.tag == "html" else ""
        out = []
        for ch in node.children:
            if isinstance(ch, Element): out.append(self.process(ch, title_context))
        return "".join(out)

    def document_xml(self) -> str:
        body = self.process(self.root)
        footer_rid = self.rid("http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", "footer1.xml")
        sect = (
            f'<w:sectPr><w:footerReference w:type="default" r:id="{footer_rid}"/>'
            '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="1020" w:bottom="964" w:left="1020" w:header="567" w:footer="510" w:gutter="0"/>'
            '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<w:body>{body}{sect}</w:body></w:document>'
        )

    def styles_xml(self):
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="135" w:line="280" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="18324B"/><w:sz w:val="42"/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="18324B"/><w:sz w:val="27"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="344653"/><w:sz w:val="17"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="18324B"/><w:sz w:val="17"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0" w:line="220" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="16"/><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Reference"><w:name w:val="Reference"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="420" w:hanging="420"/></w:pPr><w:rPr><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''

    def write(self):
        document = self.document_xml()
        rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(self.relationships) + "</Relationships>")
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
        root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
        footer = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:p><w:pPr><w:jc w:val="right"/><w:spacing w:before="0" w:after="0"/></w:pPr>
<w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="16"/><w:color w:val="56636D"/></w:rPr><w:t>SmartAgro evidence-audited manuscript  |  Page </w:t></w:r>
<w:fldSimple w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="16"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>
</w:p></w:ftr>'''
        core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>SmartAgro for Agricultural Monitoring: Evidence-Audited Component Results and a Validation Framework</dc:title>
<dc:subject>Evidence audit, crop disease, VIIRS detection, environmental stress, and validation design</dc:subject>
<dc:creator>[Author name(s) to be confirmed]</dc:creator>
<dc:description>Editable manuscript generated from the evidence-audited HTML source.</dc:description>
</cp:coreProperties>'''
        app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>'''
        with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", root_rels)
            z.writestr("word/document.xml", document)
            z.writestr("word/_rels/document.xml.rels", rels)
            z.writestr("word/styles.xml", self.styles_xml())
            z.writestr("word/footer1.xml", footer)
            z.writestr("docProps/core.xml", core)
            z.writestr("docProps/app.xml", app)
            for name, content in self.media.items():
                z.writestr(f"word/media/{name}", content)
        return DOCX_PATH


def main():
    parser = TreeBuilder()
    parser.feed(HTML_PATH.read_text(encoding="utf-8"))
    builder = DocxBuilder(parser.root)
    path = builder.write()
    print(f"Wrote native OOXML DOCX: {path}")
    print(f"Paragraphs: {builder.paragraphs}; tables: {builder.tables}; embedded figures: {builder.images}")
    print(f"File size: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
