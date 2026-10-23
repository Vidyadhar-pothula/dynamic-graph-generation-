"""Builds Computer_Vision_Review3.pptx on top of the Amrita 'Computer vision.pptx' template.
Everything is native (text boxes, shapes, tables, charts) so it stays editable in PowerPoint."""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.oxml.ns import qn
from lxml import etree
from PIL import ImageFont

TEMPLATE = "/Users/vidyadhar/Downloads/Computer vision.pptx"
OUT = "Computer_Vision_Review3.pptx"
IMG = "images/"

CRIM = RGBColor(0xB4, 0x03, 0x47)   # Amrita crimson (sampled from the logo)
INK = RGBColor(0x1F, 0x1F, 0x1F)
GREY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xF3, 0xF3, 0xF3)
LINE = RGBColor(0xBF, 0xBF, 0xBF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_TEAL = RGBColor(0x2A, 0x9D, 0x8F)
C_AMBER = RGBColor(0xE0, 0x9F, 0x1F)
C_BLUE = RGBColor(0x3F, 0x51, 0xB5)
C_CORAL = RGBColor(0xE7, 0x6F, 0x51)

prs = Presentation(TEMPLATE)

# ---- drop the template's own slides, keep master/layouts/logo ----
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)

LAYOUT = [l for l in prs.slide_layouts if l.name == "Title Only"][0]
fit_log = []   # (slide_no, name, text-box, font size) for the overflow check

FONT_BODY = "Arial"
FONT_HEAD = "Arial Black"


def new_slide(header=True):
    s = prs.slides.add_slide(LAYOUT)
    t = s.shapes.title
    t.text_frame.text = "23CSE373 - Computer Vision"
    return s


def _run(p, text, size, bold=False, color=INK, font=FONT_BODY, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    return r


def textbox(s, x, y, w, h, text, size=22, bold=False, color=INK, font=FONT_BODY,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, name=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = align
    _run(p, text, size, bold, color, font, italic)
    if name:
        tb.name = name
    fit_log.append((len(prs.slides), name or text[:25], tb, size, font, bold))
    return tb


def heading(s, text, x=0.8, y=2.15, w=18.4, size=30):
    return textbox(s, x, y, w, 0.75, text, size=size, color=CRIM, font=FONT_HEAD, name="Heading")


def bullets(s, x, y, w, h, items, size=22, gap=10, color=INK):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.38)))
        pPr.set("indent", str(-Inches(0.38)))
        for tag in ("a:buNone", "a:buChar"):
            for e in pPr.findall(qn(tag)):
                pPr.remove(e)
        bc = etree.SubElement(pPr, qn("a:buClr"))
        sc = etree.SubElement(bc, qn("a:srgbClr")); sc.set("val", "B40347")
        bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•")
        if isinstance(it, tuple):
            _run(p, it[0], size, True, color)
            _run(p, it[1], size, False, color)
        else:
            _run(p, it, size, False, color)
    tb.name = "Bullets"
    fit_log.append((len(prs.slides), "bullets", tb, size, FONT_BODY, False))
    return tb


def rect(s, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, lw=1.5, dash=False, radius=0.08):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
        if dash:
            ln = sh.line._get_or_add_ln()
            pd = etree.SubElement(ln, qn("a:prstDash")); pd.set("val", "dash")
    sh.shadow.inherit = False
    return sh


def shape_text(sh, text, size=20, bold=False, color=INK, align=PP_ALIGN.CENTER, font=FONT_BODY, lines=None):
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    items = lines or [(text, size, bold, color)]
    for i, (t, sz, b, c) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _run(p, t, sz, b, c, font)
    fit_log.append((len(prs.slides), "shape:" + items[0][0][:20], sh, max(i[1] for i in items), font, True))


def line_(s, x1, y1, x2, y2, color=GREY, w=2.25, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    if dash:
        ln = c.line._get_or_add_ln()
        pd = etree.SubElement(ln, qn("a:prstDash")); pd.set("val", "dash")
    return c


def arrow(s, x1, y1, x2, y2, color=GREY, w=2.5):
    c = line_(s, x1, y1, x2, y2, color, w)
    ln = c.line._get_or_add_ln()
    te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle")
    return c


def picture(s, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def node(s, cx, cy, d, label, color=CRIM, size=16):
    o = rect(s, cx - d / 2, cy - d / 2, d, d, fill=WHITE, line=color, shape=MSO_SHAPE.OVAL, lw=3)
    shape_text(o, label, size, True, INK)
    return o


def table(s, x, y, w, col_w, rows, row_h=0.62, size=20, header=True, first_col_bold=False):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    # kill built-in banding style, style cells ourselves
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = tbl.cell(i, j)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.15)
            hdr = header and i == 0
            c.fill.solid()
            c.fill.fore_color.rgb = CRIM if hdr else (LIGHT if i % 2 == 1 else WHITE)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            _run(p, str(rows[i][j]), size, hdr or (first_col_bold and j == 0), WHITE if hdr else INK)
    return gt


def style_chart(ch, title, ymin, ymax, fmt, colors, legend_bottom=True):
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name = Pt(20), True, FONT_BODY
    r.font.color.rgb = INK
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(16)
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale = ymin, ymax
    va.tick_labels.font.size = Pt(16); ca.tick_labels.font.size = Pt(16)
    va.tick_labels.number_format = fmt; va.tick_labels.number_format_is_linked = False
    va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    ca.has_title = True
    ca.axis_title.text_frame.text = "epoch"
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(16)
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    for ser, col in zip(ch.plots[0].series, colors):
        ser.format.line.color.rgb = col
        ser.format.line.width = Pt(3.5)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 9
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col
        ser.marker.format.line.color.rgb = col


EP = [str(i) for i in range(1, 11)]
D = dict(
    map50=[0.22684, 0.25678, 0.24108, 0.21363, 0.22603, 0.25286, 0.26339, 0.24626, 0.24764, 0.2454],
    map5095=[0.13529, 0.15102, 0.14345, 0.12227, 0.13157, 0.14764, 0.1526, 0.14368, 0.14472, 0.14602],
    tcls=[2.96703, 1.78755, 1.5923, 1.49179, 1.35952, 1.272, 1.20149, 1.14002, 1.08518, 1.03725],
    vcls=[2.09962, 2.05509, 2.10536, 2.17195, 2.13572, 2.06972, 2.0767, 2.10949, 2.12532, 2.1144],
)

# =============== 1  TITLE ===============
s = new_slide()
textbox(s, 2.8, 2.9, 14, 0.7, "Project Title", 30, color=GREY, font=FONT_HEAD, name="Label")
textbox(s, 2.8, 3.7, 14.4, 2.6, "Dynamic Scene Graph Generation from Video", 54, color=CRIM, font=FONT_HEAD, name="Project title")
textbox(s, 2.8, 6.35, 14, 0.7, "Using the VidOR dataset  |  Review 3", 28, color=INK, name="Subtitle")
xs = [4.6, 8.6, 12.6, 16.2]
labs = ["adult", "baby", "table", "toy"]
rels = ["holds", "on", "beside"]
for i in range(3):
    line_(s, xs[i] + 0.55, 8.15, xs[i + 1] - 0.55, 8.15, GREY, 2.5)
    textbox(s, (xs[i] + xs[i + 1]) / 2 - 0.8, 7.6, 1.6, 0.45, rels[i], 18, color=GREY, italic=True, align=PP_ALIGN.CENTER, name="Edge label")
for x, l in zip(xs, labs):
    node(s, x, 8.15, 1.1, l, CRIM, 17)
b = rect(s, 5.6, 9.45, 8.8, 0.8, fill=None, line=CRIM, lw=2.5, radius=0.5)
shape_text(b, "VidOR parts 1-8 trained  |  78-class pipeline", 22, True, CRIM)

# =============== 2  WHAT IS IT ===============
s = new_slide()
heading(s, "What is Dynamic Scene Graph Generation?")
bullets(s, 0.8, 3.1, 8.8, 6.6, [
    "A scene graph describes a scene as objects (nodes) and relations (edges).",
    "For video, the graph must follow each object through time.",
    "Output for every frame: <subject, relation, object> triplets.",
    "Built on VidOR with a YOLOv8s detector, tracking and relation prediction.",
], size=26, gap=16)
picture(s, IMG + "val_batch1_pred.jpg", 10.2, 3.0, w=9.0)
textbox(s, 10.2, 8.2, 9.0, 0.5, "Real validation frames with detector output (run 3)", 17, color=GREY, italic=True, name="Caption")
for i, (a, r, o) in enumerate([("adult", "holds", "baby"), ("baby", "looks at", "adult"), ("adult", "next to", "baby")]):
    c = rect(s, 10.2 + i * 3.05, 8.95, 2.9, 0.9)
    shape_text(c, "", lines=[(f"{a} - {r} - {o}", 17, True, INK)])
textbox(s, 10.2, 9.95, 9.0, 0.45, "Example triplets (illustrative)", 16, color=GREY, italic=True, name="Caption")

# =============== 3  OBJECTIVE / PROBLEM / APPROACH ===============
s = new_slide()
y = 2.2
for head_t, body_t, hh in [
    ("Objective", "Generate a dynamic scene graph for video: detect objects, track them across frames and predict how they relate over time.", 2.1),
    ("Problem Addressed", "Per-frame scene graphs have no identity across frames, so relations flicker and temporal context is lost.", 2.1),
    ("Approach", "Detect with YOLOv8s, link boxes into tracks, build one graph per frame, connect graphs over time and predict relation labels.", 2.3),
]:
    textbox(s, 0.8, y, 11, 0.6, head_t, 28, color=CRIM, font=FONT_HEAD, name="Block heading")
    textbox(s, 0.8, y + 0.65, 11.2, hh - 0.6, body_t, 24, name="Block text")
    y += hh + 0.55
rect(s, 12.8, 2.3, 6.4, 8.2)
textbox(s, 13.0, 2.45, 6.0, 0.6, "Relations as intervals", 22, True, name="Card title")
for i in range(5):
    x = 14.9 + i * 0.85
    textbox(s, x - 0.3, 3.2, 0.6, 0.4, f"t{i}", 15, color=GREY, align=PP_ALIGN.CENTER, name="tick")
    line_(s, x, 3.65, x, 9.0, LINE, 1.25, dash=True)
rows = [("adult - holds - baby", 1, 4, CRIM), ("baby - near - adult", 0, 3, C_TEAL), ("baby - looks at - toy", 2, 4, C_AMBER)]
for i, (lbl, a, b_, col) in enumerate(rows):
    yy = 4.0 + i * 1.75
    textbox(s, 13.0, yy - 0.5, 5.9, 0.45, lbl, 17, name="Row label")
    bar = rect(s, 14.9 + a * 0.85, yy, (b_ - a) * 0.85, 0.5, fill=col, radius=0.5)
textbox(s, 13.0, 9.35, 6.0, 0.9, "Bars show how long each relation holds across frames.", 16, color=GREY, italic=True, name="Caption")

# =============== 4  DATASET IDENTIFICATION ===============
s = new_slide()
heading(s, "Dataset Identification")
textbox(s, 0.8, 3.0, 9, 0.5, "VidOR is split into 8 parts, all now trained", 22, True, name="Sub")
for i in range(8):
    col, row = i % 4, i // 4
    x, y = 0.9 + col * 2.35, 3.7 + row * 2.45
    newp = i >= 5
    t = rect(s, x, y, 2.1, 2.1, fill=GREY if newp else CRIM, radius=0.1)
    shape_text(t, "", align=PP_ALIGN.CENTER, lines=[("PART", 15, False, WHITE), (str(i + 1), 40, True, WHITE), ("added now" if newp else "earlier", 16, False, WHITE)])
textbox(s, 0.9, 8.75, 9.2, 0.5, "Crimson = trained earlier   |   Grey = added for this review", 17, color=GREY, name="Legend")
table(s, 10.6, 3.2, 8.6, [4.4, 4.2], [
    ["Property", "Value"],
    ["Dataset", "VidOR"],
    ["Frame sampling", "1 fps"],
    ["Training frames", "60,588"],
    ["Validation frames", "3,594"],
    ["New videos added", "750 (3 groups)"],
    ["Object classes", "78"],
], row_h=0.75, size=22)
textbox(s, 10.6, 8.65, 8.6, 1.2, "Training frames = 33,277 earlier + 27,311 new. The validation set is unchanged, so results are comparable.", 17, color=GREY, italic=True, name="Caption")

# =============== 5  PIPELINE ===============
s = new_slide()
heading(s, "Pipeline: Eight Stages from Video to Scene Graph")
stages = [("1  preprocessing", "clean and prepare frames"), ("2  dataset", "parse labels, sample at 1 fps"),
          ("3  detection", "YOLOv8s finds objects"), ("4  tracking", "link boxes into tracks"),
          ("5  features", "appearance and geometry"), ("6  graph", "one graph per frame"),
          ("7  temporal", "connect graphs over time"), ("8  relations", "predict predicate per pair")]
for i, (a, b_) in enumerate(stages):
    col, row = i % 4, i // 4
    x, y = 0.9 + col * 4.6, 3.3 + row * 3.1
    hot = i == 2
    c = rect(s, x, y, 4.0, 2.2, fill=CRIM if hot else LIGHT, line=CRIM if hot else None, lw=2)
    shape_text(c, "", lines=[(a, 24, True, WHITE if hot else INK), (b_, 19, False, WHITE if hot else GREY)])
    if col < 3:
        arrow(s, x + 4.0, y + 1.1, x + 4.6, y + 1.1, GREY)
arrow(s, 0.9 + 3 * 4.6 + 2.0, 3.3 + 2.2, 0.9 + 3 * 4.6 + 2.0, 3.3 + 2.65, GREY)
line_(s, 0.9 + 3 * 4.6 + 2.0, 3.3 + 2.65, 0.9 + 2.0, 3.3 + 2.65, GREY, 2.5)
arrow(s, 0.9 + 2.0, 3.3 + 2.65, 0.9 + 2.0, 3.3 + 3.1, GREY)
textbox(s, 0.9, 9.8, 18.2, 0.9, "Trained and evaluated: detection, the relation model and the temporal LSTM/GRU models. Later stages consume the detector's boxes, so its quality sets the ceiling.", 21, color=INK, name="Note")

# =============== 6  DATASET EXPORT ===============
s = new_slide()
heading(s, "What the Dataset Export Contains")
table(s, 1.6, 3.2, 16.8, [6.4, 10.4], [
    ["Folder / file", "Contents"],
    ["annotations", "VidOR labels for train and validation"],
    ["videos", "Source clips"],
    ["yolo_format", "Frames and boxes in YOLO layout"],
    ["tracks", "Per-video object tracks"],
    ["features", "Extracted object features"],
    ["relation_dataset", "Object-pair relation samples"],
    ["relation_dataset_corrected", "Revised relation samples"],
    ["dataset_manifest.json", "Parts, fps and counts"],
], row_h=0.78, size=22, first_col_bold=True)

# =============== 7  DETECTOR ===============
s = new_slide()
heading(s, "Detector: YOLOv8s, Three Colab Runs and a Fine-tune")
table(s, 0.9, 3.2, 9.2, [3.7, 5.5], [
    ["Setting", "Value"],
    ["Base model", "yolov8s.pt, pretrained"],
    ["Input size", "640 px"],
    ["Epochs", "10 per run, 3 fine-tune"],
    ["Optimizer", "auto (AdamW)"],
    ["Augmentation", "mosaic, flip 0.5, HSV"],
    ["Hardware", "Colab GPU, then M3 Pro Mac"],
], row_h=0.9, size=22, first_col_bold=True)
for i, (r, bch, note, kind) in enumerate([("Run 1", "batch 16", "weights saved", 0), ("Run 2", "batch 32", "weights saved", 0), ("Run 3", "batch 32", "best mAP50 0.263", 1), ("Fine-tune", "batch 16", "from run 3, all parts", 2)]):
    y = 3.2 + i * 1.7
    c = rect(s, 11.0, y, 8.2, 1.45, fill=LIGHT if kind == 2 else None, line=CRIM if kind else GREY, lw=3 if kind == 2 else 1.5, dash=(kind == 0), radius=0.1)
    textbox(s, 11.3, y + 0.15, 3.6, 0.6, r, 26, color=CRIM if kind else INK, font=FONT_HEAD, name="Run")
    textbox(s, 11.3, y + 0.8, 3.0, 0.45, bch, 19, color=GREY, name="Run batch")
    textbox(s, 14.6, y + 0.4, 4.4, 0.7, note, 20, color=INK, align=PP_ALIGN.RIGHT, name="Run note")

# =============== 8  CHARTS ===============
s = new_slide()
heading(s, "Training Results: mAP Plateaus as Validation Loss Stays Flat")
cd = CategoryChartData(); cd.categories = EP
cd.add_series("mAP50", D["map50"]); cd.add_series("mAP50-95", D["map5095"])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.8), Inches(3.0), Inches(9.0), Inches(6.5), cd)
style_chart(gf.chart, "mAP by epoch", 0.1, 0.3, "0.00", [C_AMBER, C_TEAL])
cd = CategoryChartData(); cd.categories = EP
cd.add_series("train cls loss", D["tcls"]); cd.add_series("val cls loss", D["vcls"])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(10.2), Inches(3.0), Inches(9.0), Inches(6.5), cd)
style_chart(gf.chart, "Classification loss by epoch", 0.5, 3.5, "0.0", [C_CORAL, C_BLUE])
textbox(s, 0.8, 9.65, 18.4, 1.0, "Training loss keeps falling while validation loss stays near 2.1, a sign of over-fitting. Full run: 10 epochs, about 79 minutes on one Colab GPU.", 20, color=GREY, name="Caption")

# =============== 9  STATS + PR ===============
s = new_slide()
heading(s, "Result Metrics: Peak 0.263 mAP50 at Epoch 7")
stats = [("0.263", "mAP50, best (epoch 7)", CRIM), ("0.153", "mAP50-95, best", C_TEAL), ("0.363", "precision, final epoch", C_CORAL), ("0.251", "recall, final epoch", C_BLUE)]
for i, (v, l, col) in enumerate(stats):
    x, y = 0.9 + (i % 2) * 4.6, 3.2 + (i // 2) * 3.6
    c = rect(s, x, y, 4.3, 3.3)
    shape_text(c, "", lines=[(v, 54, True, col), (l, 19, False, GREY)])
picture(s, IMG + "BoxPR_curve.png", 10.4, 3.1, w=8.8)
textbox(s, 10.4, 8.95, 8.8, 0.9, "Precision-recall curve: all classes (blue) and each class (grey).", 18, color=GREY, italic=True, name="Caption")

# =============== FINE-TUNE RESULTS ===============
s = new_slide()
heading(s, "Fine-tune on All Parts: Detector Results")
cd = CategoryChartData(); cd.categories = ["1", "2", "3"]
cd.add_series("fine-tune mAP50", [0.2492, 0.2556, 0.2602]); cd.add_series("run-3 baseline 0.2635", [0.2635, 0.2635, 0.2635])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.8), Inches(3.0), Inches(8.6), Inches(6.0), cd)
style_chart(gf.chart, "mAP50, same validation set", 0.24, 0.27, "0.000", [C_AMBER, C_BLUE])
table(s, 9.9, 3.3, 9.3, [3.3, 1.2, 1.2, 1.8, 1.8], [
    ["", "P", "R", "mAP50", "mAP50-95"],
    ["Baseline", "0.381", "0.285", "0.2635", "0.1526"],
    ["Epoch 1", "0.410", "0.235", "0.2492", "0.1471"],
    ["Epoch 2", "0.407", "0.272", "0.2556", "0.1508"],
    ["Epoch 3", "0.331", "0.303", "0.2602", "0.1548"],
], row_h=0.8, size=19, first_col_bold=True)
textbox(s, 9.9, 7.5, 9.3, 2.4, "27,311 new frames did not raise mAP50 (0.2635 to 0.2602). Recall rose, precision fell, and mAP50-95 is slightly higher. Validation classification loss improved from about 2.1 to 1.96. Each epoch took about 76 minutes on an M3 Pro.", 19, color=INK, name="Caption")
textbox(s, 0.8, 9.2, 8.6, 0.8, "Baseline = Colab run 3, scored on the same 3,594 validation images.", 16, color=GREY, italic=True, name="Caption")

# =============== FINE-TUNE GRAPHS ===============
s = new_slide()
heading(s, "Fine-tune Training Graphs")
picture(s, IMG + "finetune_results.png", 0.8, 3.2, w=11.4)
textbox(s, 0.8, 8.95, 11.4, 0.9, "Losses, precision, recall and mAP over the 3 fine-tune epochs.", 18, color=GREY, italic=True, name="Caption")
picture(s, IMG + "finetune_confusion_crop.png", 12.6, 3.0, h=5.7)
textbox(s, 12.6, 8.85, 6.6, 0.8, "Confusion matrix after fine-tuning.", 18, color=GREY, italic=True, name="Caption")

# =============== 10  ERROR ANALYSIS ===============
s = new_slide()
heading(s, "Error Analysis: Strong on Adults, Thin on Other Classes")
picture(s, IMG + "confusion_matrix_crop.png", 1.2, 3.0, h=6.3)
textbox(s, 1.0, 9.4, 8.0, 0.9, "Confusion matrix: a dark adult cell, faint scatter elsewhere.", 18, color=GREY, italic=True, name="Caption")
picture(s, IMG + "val_batch0_pred.jpg", 9.4, 3.2, w=9.8)
textbox(s, 9.4, 8.9, 9.8, 0.9, "Validation predictions: adult and baby are found, with confidence 0.3 to 0.5.", 18, color=GREY, italic=True, name="Caption")

# =============== 11  TRAINING GRAPHS ===============
s = new_slide()
heading(s, "Training Graphs from the Run")
picture(s, IMG + "results.png", 0.8, 3.2, w=11.4)
textbox(s, 0.8, 8.95, 11.4, 0.9, "Losses, precision, recall and mAP over the 10 epochs.", 18, color=GREY, italic=True, name="Caption")
picture(s, IMG + "BoxF1_curve.png", 12.6, 3.2, w=6.6)
textbox(s, 12.6, 8.2, 6.6, 0.9, "F1 versus confidence threshold.", 18, color=GREY, italic=True, name="Caption")

# =============== 12  TEMPORAL GRAPH ===============
s = new_slide()
heading(s, "Tracks Turn Per-Frame Boxes into a Temporal Graph")
cols = [3.4, 9.6, 15.8]
rowsy = [4.3, 6.7, 9.1]
ids = [("adult #1", CRIM), ("baby #2", C_TEAL), ("toy #3", C_AMBER)]
for k, cx in enumerate(cols):
    textbox(s, cx - 1.5, 3.0, 3.0, 0.5, ["frame t", "frame t+1", "frame t+2"][k], 20, True, color=GREY, align=PP_ALIGN.CENTER, name="Frame label")
for r in range(3):
    for k in range(2):
        line_(s, cols[k] + 0.75, rowsy[r], cols[k + 1] - 0.75, rowsy[r], LINE, 2.25, dash=True)
line_(s, cols[0], rowsy[0] + 0.75, cols[0], rowsy[1] - 0.75, CRIM, 3); textbox(s, cols[0] + 0.15, 5.2, 1.5, 0.45, "holds", 17, color=GREY, italic=True, name="Rel")
line_(s, cols[1], rowsy[0] + 0.75, cols[1], rowsy[1] - 0.75, CRIM, 3); textbox(s, cols[1] + 0.15, 5.2, 1.5, 0.45, "holds", 17, color=GREY, italic=True, name="Rel")
line_(s, cols[1], rowsy[1] + 0.75, cols[1], rowsy[2] - 0.75, C_TEAL, 3); textbox(s, cols[1] + 0.15, 7.6, 1.8, 0.45, "looks at", 17, color=GREY, italic=True, name="Rel")
line_(s, cols[2], rowsy[1] + 0.75, cols[2], rowsy[2] - 0.75, C_TEAL, 3); textbox(s, cols[2] + 0.15, 7.6, 1.8, 0.45, "looks at", 17, color=GREY, italic=True, name="Rel")
for r in range(3):
    for cx in cols:
        node(s, cx, rowsy[r], 1.5, ids[r][0], ids[r][1], 16)
textbox(s, 0.8, 10.05, 18.4, 0.6, "Solid lines: relations inside a frame.   Dashed lines: the same object linked through time.", 19, color=GREY, name="Caption")

# =============== COMBINING OLD AND NEW ===============
s = new_slide()
heading(s, "Combining Old and New Training: 28 to 78 Classes")
flow = [("Old relation model", "28 classes"), ("Copy weights", "by class name"), ("Expanded model", "78 classes, identical output"), ("Fine-tune", "old + new data")]
for i, (a, b_) in enumerate(flow):
    x = 0.9 + i * 4.7
    c = rect(s, x, 3.2, 4.0, 1.9, fill=LIGHT if i < 3 else None, line=CRIM if i == 3 else None, lw=3)
    shape_text(c, "", lines=[(a, 22, True, INK), (b_, 17, False, GREY)])
    if i < 3:
        arrow(s, x + 4.0, 4.15, x + 4.7, 4.15, GREY)
table(s, 3.0, 5.7, 14.0, [6.0, 2.6, 2.7, 2.7], [
    ["", "Old data", "New data", "Combined"],
    ["Relation training pairs", "7,536", "145,031", "152,567"],
    ["Temporal training sequences", "1,890", "38,388", "40,278"],
], row_h=0.8, size=21, first_col_bold=True)
textbox(s, 0.9, 8.5, 18.2, 2.2, "The expanded model gave exactly the same answers as the old one on the old validation set (accuracy 0.429, loss 1.758), so no knowledge was lost. 38 new videos (every 10th) were held out of training to test on unseen data.", 21, name="Note")

# =============== RELATION + TEMPORAL RESULTS ===============
s = new_slide()
heading(s, "Relation and Temporal Models: Before and After Fine-tuning")
table(s, 1.0, 3.0, 18.0, [4.2, 3.4, 3.4, 3.5, 3.5], [['Model', 'Test set', 'Accuracy', 'Macro-F1', 'Loss'], ['Relation', 'old val', '0.429 -> 0.529', '0.068 -> 0.077', '1.758 -> 1.490'], ['Relation', 'new videos', '0.303 -> 0.407', '0.098 -> 0.124', '2.264 -> 1.781'], ['Temporal LSTM', 'old val', '0.645 -> 0.586', '0.072 -> 0.099', '1.317 -> 1.375'], ['Temporal LSTM', 'new videos', '0.278 -> 0.372', '0.026 -> 0.074', '2.100 -> 1.891'], ['Temporal GRU', 'old val', '0.576 -> 0.628', '0.075 -> 0.113', '1.510 -> 1.372'], ['Temporal GRU', 'new videos', '0.281 -> 0.390', '0.032 -> 0.063', '2.284 -> 1.809']], row_h=0.62, size=20, first_col_bold=True)
bullets(s, 1.0, 7.5, 18.0, 3.2, [
    "Every model improves on the unseen new videos; the old models had never seen most of the 78 classes.",
    "The LSTM is slightly worse on the old validation set (accuracy 0.645 to 0.586); the GRU improves on both.",
    "Scores are still low: macro-F1 is at most 0.13, because background pairs dominate the data.",
    "Validation loss is best after 1 to 3 epochs, then rises (over-fitting), so early stopping chose the checkpoints.",
], size=19, gap=6)

# =============== WHY TWO PREDICATES DOMINATED ===============
s = new_slide()
heading(s, "Why Two Predicates Dominated the Graphs")
cd = CategoryChartData()
share = [("hold", 0.7), ("towards", 0.7), ("play(instrument)", 1.0), ("above", 2.3), ("lean_on", 2.5), ("beneath", 4.7), ("watch", 6.4), ("behind", 7.0), ("in_front_of", 34.0), ("next_to", 38.5)]
cd.categories = [a for a, _ in share]; cd.add_series("share of positive training labels (%)", [b for _, b in share])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(3.0), Inches(9.4), Inches(7.2), cd)
ch = gf.chart; ch.has_legend = False; ch.has_title = True
ch.chart_title.text_frame.text = "Share of positive training labels (%)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(20); ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.category_axis.tick_labels.font.size = Pt(16); ch.value_axis.tick_labels.font.size = Pt(16)
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
pl = ch.plots[0]; pl.has_data_labels = True; pl.data_labels.font.size = Pt(15); pl.data_labels.number_format = "0.0"; pl.data_labels.number_format_is_linked = False
pl.series[0].format.fill.solid(); pl.series[0].format.fill.fore_color.rgb = CRIM; pl.gap_width = 60
bullets(s, 10.6, 3.1, 8.6, 7.4, [
    ("Data: ", "next_to and in_front_of are 72% of all positive labels; the rarest predicates have fewer than 30 samples."),
    ("Loss: ", "v1 used an unweighted loss, so guessing the common classes pays off."),
    ("Checkpoints: ", "v1 was chosen by validation loss, but macro-F1 on new videos was still rising (0.124 to 0.156 by epoch 5)."),
    ("Result: ", "v1 predicted only 11 of 32 predicates (the GRU only 4)."),
], size=21, gap=12)

# =============== THE FIX: V1 VS V2 ===============
s = new_slide()
heading(s, "The Fix: Class-balanced Loss, Checkpoints by Macro-F1")
table(s, 1.0, 3.0, 18.0, [3.6, 2.8, 3.4, 3.2, 2.7, 2.3], [['Model', 'Test set', 'Macro-F1', 'Accuracy', 'Predicates used', 'Top-2 share'], ['Relation', 'new videos', '0.124 -> 0.189', '0.407 -> 0.406', '11 -> 23', '74% -> 58%'], ['Relation', 'old val', '0.084 -> 0.064', '0.529 -> 0.314', '8 -> 14', '73% -> 53%'], ['Temporal LSTM', 'new videos', '0.074 -> 0.145', '0.372 -> 0.339', '8 -> 17', '95% -> 54%'], ['Temporal LSTM', 'old val', '0.099 -> 0.059', '0.586 -> 0.322', '4 -> 10', '92% -> 47%'], ['Temporal GRU', 'new videos', '0.063 -> 0.172', '0.390 -> 0.333', '4 -> 22', '99% -> 45%'], ['Temporal GRU', 'old val', '0.113 -> 0.130', '0.628 -> 0.396', '3 -> 13', '100% -> 49%']], row_h=0.62, size=19, first_col_bold=True)
bullets(s, 1.0, 7.9, 18.0, 3.0, [
    "v2 = v1 retrained with loss weights proportional to 1/sqrt(frequency), capped at 12x, with the checkpoint chosen by macro-F1. Top-2 share = share of non-background predictions that are next_to or in_front_of.",
    "On new videos every model gains macro-F1 and uses far more predicates; the GRU's top-2 share falls from 99% to 45%.",
    "Cost: accuracy drops. On the small old validation set the result is mixed: relation worse, LSTM worse, GRU better.",
], size=18, gap=6)

# =============== V2 GRAPHS ===============
for k, (f1, f2, cap) in enumerate([("v2_graph_1_trim.png", "v2_graph_2_trim.png", "Held-out new videos. Examples were picked for predicate variety, not at random."),
                                   ("v2_graph_3_trim.png", "v2_graph_4_trim.png", "Some relations are plausible (watch, hold, towards), others are wrong: scores remain low.")], 1):
    s = new_slide()
    heading(s, f"Dynamic Scene Graphs After the Fix ({k}/2)")
    from PIL import Image as _I
    for yy, ff in ((2.25, f1), (6.35, f2)):
        _im = _I.open(IMG + ff); _h = 3.85; _w = _h * _im.width / _im.height
        picture(s, IMG + ff, (20 - _w) / 2, yy, h=_h)
    textbox(s, 2.25, 10.3, 15.5, 0.5, cap, 16, color=GREY, italic=True, name="Caption", align=PP_ALIGN.CENTER)

# =============== 13  QUALITATIVE RESULTS ===============
s = new_slide()
heading(s, "Qualitative Results: Frames and Predicted Graphs")
for i, f in enumerate(["qual_q1", "qual_q5", "qual_q6", "qual_q10"]):
    x, y = 2.0 + (i % 2) * 8.1, 3.0 + (i // 2) * 3.95
    picture(s, IMG + f + ".png", x, y, w=7.7)
textbox(s, 2.0, 10.4, 16, 0.5, "Left of each panel: tracked, labelled objects. Right: predicted scene graph with relation labels.", 17, color=GREY, italic=True, name="Caption")

# =============== 14  REVIEW 3 ===============
s = new_slide()
heading(s, "Review 3: VidOR Training Progress")
for i in range(8):
    newp = i >= 5
    t = rect(s, 1.5 + i * 2.2, 3.0, 1.9, 0.95, fill=GREY if newp else CRIM, radius=0.15)
    shape_text(t, str(i + 1), 26, True, WHITE)
textbox(s, 1.5, 4.05, 9.5, 0.5, "Parts 1-5 trained earlier", 19, True, color=CRIM, name="Legend")
textbox(s, 10.0, 4.05, 8.7, 0.5, "Parts 6-8 added now", 19, True, color=GREY, align=PP_ALIGN.RIGHT, name="Legend")
table(s, 1.5, 4.9, 17.0, [7.0, 5.0, 5.0], [
    ["Stage", "Before (parts 1-5)", "After (parts 6-8 added)"],
    ["Detector mAP50 / mAP50-95", "0.2635 / 0.1526", "0.2602 / 0.1548"],
    ["Detector recall", "0.285", "0.303"],
    ["Relation, new videos (acc / macro-F1)", "0.303 / 0.098", "0.408 / 0.124"],
    ["Temporal GRU, new videos (acc / macro-F1)", "0.281 / 0.032", "0.390 / 0.063"],
], row_h=0.78, size=21, first_col_bold=True)
textbox(s, 1.5, 9.0, 17.0, 1.6, "Honest summary: more data did not raise detector mAP50. The relation and temporal models improved on unseen new videos, but scores remain low.", 21, True, color=INK, name="Summary")

# =============== 15  CHALLENGES + NEXT REVIEW ===============
s = new_slide()
textbox(s, 0.9, 2.3, 8.6, 0.7, "Challenges Faced", 30, color=CRIM, font=FONT_HEAD, name="Heading")
bullets(s, 0.9, 3.2, 8.6, 7.0, [
    ("Compute: ", "limited GPU access, so the final fine-tune ran on a Mac (about 4 hours for 3 epochs)."),
    ("Class mismatch: ", "the pipeline had 28 classes, the detector 78. Fixed by expanding by class name."),
    ("Over-fitting: ", "relation and temporal validation loss rises after 1 to 3 epochs."),
    ("Low scores: ", "detector recall 0.30 and macro-F1 at most 0.13."),
], size=22, gap=14)
textbox(s, 10.6, 2.3, 8.6, 0.7, "Next Review", 30, color=CRIM, font=FONT_HEAD, name="Heading")
bullets(s, 10.6, 3.2, 8.6, 7.0, [
    "Run the full pipeline end to end and evaluate it.",
    "Train the detector longer with a tuned learning rate.",
    "Use a class-weighted relation loss to lift macro-F1.",
    "Track and featurise every new video, not every second one.",
], size=22, gap=14)

prs.save(OUT)

# ---------- text-fit check (Arial metrics) ----------
def font_for(name, bold, size):
    path = {"Arial": "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "Arial Black": "/System/Library/Fonts/Supplemental/Arial Black.ttf"}[name]
    return ImageFont.truetype(path, int(size * 10))   # scaled x10 for precision

problems = []
for sn, nm, sh, size, font, bold in fit_log:
    w_in = Emu(sh.width).inches - 0.2
    h_in = Emu(sh.height).inches
    total = 0.0
    for p in sh.text_frame.paragraphs:
        text = "".join(r.text for r in p.runs)
        if not text:
            continue
        sz = p.runs[0].font.size.pt if p.runs and p.runs[0].font.size else size
        f = font_for(font, bold or (p.runs and p.runs[0].font.bold), sz)
        words, lines, cur = text.split(" "), 1, ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if f.getlength(t) / 10 / 72 > w_in and cur:
                lines += 1; cur = wd
            else:
                cur = t
        total += lines * sz * 1.2 / 72 + (p.space_after.pt / 72 if p.space_after else 0)
    if total > h_in + 0.15:
        problems.append((sn, nm, round(total, 2), round(h_in, 2)))
print("slides:", len(prs.slides))
print("possible overflow:", problems if problems else "none")
