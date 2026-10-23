const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const THEME = {
  name: "Scene Graph Night", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1:"E8ECF1", lt1:"0F1318", dk2:"8A94A3", lt2:"1A2028",
    accent1:"FFB020", accent2:"4FD1C5", accent3:"FF6B5B", accent4:"8B9CFF", accent5:"A3E635", accent6:"5B6573",
    hlink:"4FD1C5", folHlink:"8B9CFF" }
};
const HX = { bg:"0F1318", panel:"1A2028", text:"E8ECF1", mute:"8A94A3", amber:"FFB020", teal:"4FD1C5", coral:"FF6B5B", peri:"8B9CFF", grid:"2A323D" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Dynamic Scene Graph Generation - Review 3";
const C = pres.SchemeColor;
const S = pres.ShapeType;

pres.defineSlideMaster({
  title: "TITLE_ONLY", background: { color: HX.bg },
  objects: [
    { text: { text: "DYNAMIC SCENE GRAPH  ·  VIDOR  ·  REVIEW 3", options: { x:0.6, y:7.0, w:6, h:0.3, fontSize:10, color: C.text2, charSpacing:3, margin:0, fontFace:"Calibri" } } },
  ],
  slideNumber: { x:12.2, y:7.0, w:0.6, h:0.3, fontSize:10, color: C.text2, align:"right" },
  placeholders: [
    { placeholder: { options: { name:"title", type:"title", x:0.6, y:0.7, w:12.1, h:0.9, fontSize:32, bold:true, fontFace:"Cambria", color: C.text1, valign:"top", margin:0 }, text:"" } },
  ],
});
pres.defineSlideMaster({ title: "COVER", background: { color: HX.bg } });

let n = 0;
function content(kicker, title, notes) {
  n++;
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText(kicker, { x:0.6, y:0.4, w:8, h:0.3, fontSize:12, bold:true, color: C.accent1, charSpacing:4, margin:0, isTextBox:true });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function line(s, x1,y1,x2,y2, color, w=1.5, dash) {
  const o = { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1), h: Math.abs(y2-y1), line:{ color, width:w, dashType: dash||"solid" } };
  if ((y2<y1) !== (x2<x1)) o.flipV = true;
  s.addShape(S.line, o);
}
function node(s, cx, cy, d, label, color, fs=12) {
  s.addShape(S.ellipse, { x:cx-d/2, y:cy-d/2, w:d, h:d, fill:{ color: C.background2 }, line:{ color, width:2 } });
  s.addText(label, { x:cx-d/2, y:cy-d/2, w:d, h:d, align:"center", valign:"middle", fontSize:fs, bold:true, color: C.text1, margin:0, isTextBox:true });
}
function edgeLabel(s, x, y, text, color) {
  s.addText(text, { x:x-0.6, y:y-0.15, w:1.2, h:0.3, align:"center", valign:"middle", fontSize:11, italic:true, color, fill:{ color: C.background1 }, margin:0, isTextBox:true });
}
function card(s, x,y,w,h, fillScheme) {
  s.addShape(S.roundRect, { x,y,w,h, rectRadius:0.12, fill:{ color: fillScheme||C.background2 }, line:{ color: C.background2, width:0 } });
}
const chartBase = (title) => ({
  showTitle:true, title, titleColor: HX.text, titleFontSize:14, titleFontFace:"+mn-lt",
  catAxisLabelColor: HX.mute, valAxisLabelColor: HX.mute, catAxisLabelFontFace:"+mn-lt", valAxisLabelFontFace:"+mn-lt",
  catAxisLabelFontSize:11, valAxisLabelFontSize:11,
  valGridLine:{ color: HX.grid, size:0.75 }, catGridLine:{ style:"none" },
  showLegend:true, legendPos:"b", legendColor: HX.mute, legendFontSize:11, legendFontFace:"+mn-lt",
  lineSize:3, lineDataSymbolSize:7,
});
const EP = ["1","2","3","4","5","6","7","8","9","10"];
const D = {
  map50:[0.22684,0.25678,0.24108,0.21363,0.22603,0.25286,0.26339,0.24626,0.24764,0.2454],
  map5095:[0.13529,0.15102,0.14345,0.12227,0.13157,0.14764,0.1526,0.14368,0.14472,0.14602],
  prec:[0.49376,0.46445,0.38156,0.38332,0.38057,0.43729,0.37864,0.34915,0.40492,0.36274],
  rec:[0.23091,0.25514,0.2585,0.22514,0.24203,0.25129,0.28468,0.26022,0.25506,0.25096],
  tcls:[2.96703,1.78755,1.5923,1.49179,1.35952,1.272,1.20149,1.14002,1.08518,1.03725],
  vcls:[2.09962,2.05509,2.10536,2.17195,2.13572,2.06972,2.0767,2.10949,2.12532,2.1144],
  tbox:[1.28717,1.21723,1.19172,1.17323,1.13472,1.09987,1.07094,1.04694,1.01931,0.9983],
  vbox:[1.3919,1.41032,1.44216,1.47072,1.45015,1.44617,1.44621,1.45207,1.44155,1.44095],
};

// ---------- 1 COVER ----------
{
  n++;
  const s = pres.addSlide({ masterName: "COVER" });
  s.addText("REVIEW 3  ·  VIDOR DATASET", { x:0.7, y:0.8, w:7, h:0.3, fontSize:13, bold:true, color: C.accent1, charSpacing:5, margin:0, isTextBox:true });
  s.addText("Dynamic Scene Graph Generation from Video", { x:0.7, y:1.5, w:7.0, h:2.6, fontSize:50, bold:true, fontFace:"Cambria", color: C.text1, valign:"top", margin:0, isTextBox:true });
  s.addText("Detect objects, track them through time, and predict how they relate - frame by frame.", { x:0.7, y:4.3, w:6.6, h:0.9, fontSize:18, color: C.text2, margin:0, isTextBox:true });
  s.addShape(S.roundRect, { x:0.7, y:5.9, w:5.6, h:0.5, rectRadius:0.25, fill:{ color: C.background2 }, line:{ color: C.accent1, width:1.25 } });
  s.addText("5 of 8 VidOR parts complete  ·  parts 6-8 training", { x:0.7, y:5.9, w:5.6, h:0.5, align:"center", valign:"middle", fontSize:14, bold:true, color: C.accent1, margin:0, isTextBox:true });
  // node art
  const P = { person:[10.3,1.9], bottle:[12.2,3.3], table:[9.9,4.6], dog:[11.9,5.9] };
  line(s,...P.person,...P.bottle,HX.amber,2); edgeLabel(s,11.25,2.6,"holds",HX.amber);
  line(s,...P.person,...P.table,HX.teal,2); edgeLabel(s,10.1,3.25,"stands by",HX.teal);
  line(s,...P.bottle,...P.table,HX.peri,2); edgeLabel(s,11.05,3.95,"on",HX.peri);
  line(s,...P.table,...P.dog,HX.coral,2); edgeLabel(s,10.9,5.25,"near",HX.coral);
  line(s,...P.person,...P.dog,HX.mute,1.25,"dash"); 
  node(s,...P.person,1.1,"person",HX.amber,13); node(s,...P.bottle,1.0,"bottle",HX.peri,13);
  node(s,...P.table,1.0,"table",HX.teal,13); node(s,...P.dog,1.0,"dog",HX.coral,13);
  s.addText("illustrative scene graph", { x:9.2, y:6.7, w:4, h:0.3, fontSize:11, italic:true, color: C.text2, align:"right", margin:0, isTextBox:true });
  s.addNotes("Title. Project: dynamic scene graph generation on VidOR. Review 3 update: 5 of 8 dataset parts complete, parts 6-8 are training.");
}

// ---------- 2 PROBLEM ----------
{
  const s = content("01  ·  THE PROBLEM", "A video is a story of who does what to whom", "Scene graph = objects as nodes, relations as edges. We want this for every moment of a video.");
  s.addImage({ path:"images/val_batch1_pred.jpg", x:0.6, y:1.9, w:6.3, h:3.61, altText:"Validation frames with detected objects" });
  s.addText("Real validation frames with detector output (run 3)", { x:0.6, y:5.6, w:6.3, h:0.4, fontSize:12, italic:true, color: C.text2, margin:0, isTextBox:true });
  // triplets
  s.addText("OUTPUT PER FRAME  (illustrative)", { x:7.5, y:1.9, w:5, h:0.3, fontSize:12, bold:true, color: C.text2, charSpacing:3, margin:0, isTextBox:true });
  const tr = [["adult","holds","baby",HX.amber],["baby","looks at","adult",HX.coral],["adult","next to","baby",HX.teal]];
  tr.forEach((t,i)=>{ const y=2.4+i*1.0; card(s,7.5,y,5.2,0.8);
    s.addText([{text:t[0],options:{bold:true,color:HX.text}},{text:"   "+t[1]+"   ",options:{italic:true,color:t[3]}},{text:t[2],options:{bold:true,color:HX.text}}],{x:7.75,y,w:4.8,h:0.8,fontSize:20,valign:"middle",margin:0,isTextBox:true}); });
  s.addText("Doing this for every frame, with objects that keep their identity over time, is this project's task.", { x:7.5, y:5.5, w:5.2, h:0.9, fontSize:16, color: C.text2, margin:0, isTextBox:true });
}

// ---------- 3 STATIC VS DYNAMIC ----------
{
  const s = content("02  ·  WHY DYNAMIC", "Per-frame graphs miss what changes over time", "A static scene graph treats each frame alone. A dynamic one keeps identities and records when relations start and stop.");
  card(s,0.6,1.9,5.9,2.2); card(s,6.8,1.9,5.9,2.2);
  s.addText("Static (per frame)", { x:0.9, y:2.05, w:5.3, h:0.4, fontSize:20, bold:true, color: C.text1, margin:0, isTextBox:true });
  s.addText([{text:"Each frame labelled on its own",options:{bullet:true,breakLine:true}},{text:"No identity link between frames",options:{bullet:true,breakLine:true}},{text:"Relations flicker on and off",options:{bullet:true}}], { x:0.9, y:2.55, w:5.3, h:1.4, fontSize:15, color: C.text2, paraSpaceAfter:6, margin:0, isTextBox:true });
  s.addText("Dynamic (this project)", { x:7.1, y:2.05, w:5.3, h:0.4, fontSize:20, bold:true, color: C.accent1, margin:0, isTextBox:true });
  s.addText([{text:"Tracked objects keep one ID",options:{bullet:true,breakLine:true}},{text:"Relations carry a start and an end",options:{bullet:true,breakLine:true}},{text:"Context from neighbouring frames",options:{bullet:true}}], { x:7.1, y:2.55, w:5.3, h:1.4, fontSize:15, color: C.text1, paraSpaceAfter:6, margin:0, isTextBox:true });
  // timeline
  s.addText("RELATIONS AS INTERVALS", { x:0.6, y:4.45, w:5, h:0.3, fontSize:12, bold:true, color: C.text2, charSpacing:3, margin:0, isTextBox:true });
  const x0=2.8, step=2.0;
  ["t0","t1","t2","t3","t4"].forEach((t,i)=>{ s.addText(t,{x:x0+i*step-0.3,y:4.85,w:0.6,h:0.3,fontSize:12,align:"center",color:C.text2,margin:0,isTextBox:true}); line(s,x0+i*step,5.2,x0+i*step,6.55,HX.grid,0.75,"dash"); });
  const rows=[["person - holds - ball",1,4,HX.amber],["dog - near - person",0,3,HX.teal],["dog - looks at - ball",2,4,HX.coral]];
  rows.forEach((r,i)=>{ const y=5.3+i*0.42; s.addText(r[0],{x:0.6,y,w:2.1,h:0.32,fontSize:12,color:C.text1,valign:"middle",margin:0,isTextBox:true});
    s.addShape(S.roundRect,{x:x0+r[1]*step,y:y+0.02,w:(r[2]-r[1])*step,h:0.28,rectRadius:0.14,fill:{color:r[3]},line:{color:r[3],width:0}}); });
}

// ---------- 4 DATASET ----------
{
  const s = content("03  ·  DATA", "VidOR comes in eight parts - five are trained", "VidOR training set is split into 8 parts. Parts 1-5 are complete. Parts 6-8 are training now because GPU access was limited. Stats from dataset_manifest.json (export snapshot).");
  const done=5;
  for (let i=0;i<8;i++){ const col=i%4,row=Math.floor(i/4); const x=0.6+col*1.75,y=1.95+row*1.75; const ok=i<done;
    s.addShape(S.roundRect,{x,y,w:1.55,h:1.55,rectRadius:0.12,fill:{color: ok?C.accent1:C.background2},line:{color: ok?C.accent1:C.accent6,width:1.5,dashType: ok?"solid":"dash"}});
    s.addText("PART",{x,y:y+0.25,w:1.55,h:0.3,align:"center",fontSize:11,bold:true,charSpacing:3,color: ok?C.background1:C.text2,margin:0,isTextBox:true});
    s.addText(String(i+1),{x,y:y+0.5,w:1.55,h:0.6,align:"center",fontSize:36,bold:true,fontFace:"Cambria",color: ok?C.background1:C.text1,margin:0,isTextBox:true});
    s.addText(ok?"done":"training",{x,y:y+1.12,w:1.55,h:0.3,align:"center",fontSize:12,color: ok?C.background1:C.text2,margin:0,isTextBox:true}); }
  s.addText("Filled = complete   ·   Dashed = in training", { x:0.6, y:5.6, w:6.8, h:0.3, fontSize:12, color: C.text2, margin:0, isTextBox:true });
  const stats=[["1 fps","frame sampling rate"],["33,277","training frames (YOLO format)"],["3,594","validation frames"],["250","videos per part"]];
  stats.forEach((t,i)=>{ const y=1.95+i*1.05; s.addText(t[0],{x:7.9,y,w:2.6,h:0.9,fontSize:38,bold:true,fontFace:"Cambria",color:C.accent1,valign:"middle",margin:0,isTextBox:true}); s.addText(t[1],{x:10.5,y,w:2.3,h:0.9,fontSize:14,color:C.text2,valign:"middle",margin:0,isTextBox:true}); });
  s.addText("Counts from the last dataset export manifest.", { x:7.9, y:6.25, w:4.9, h:0.3, fontSize:11, italic:true, color: C.text2, margin:0, isTextBox:true });
}

// ---------- 5 PIPELINE ----------
{
  const s = content("04  ·  PIPELINE", "Eight stages from raw video to scene graph", "Stages mirror the src/ folder of the project.");
  const st=[["preprocessing","Clean and prepare frames"],["dataset","Parse VidOR labels, sample at 1 fps"],["detection","YOLOv8s finds objects per frame"],["tracking","Link detections into tracks"],["features","Appearance and geometry per object"],["graph","Build a graph for each frame"],["temporal","Connect graphs across time"],["relations","Predict the predicate between pairs"]];
  const w=1.4,g=0.127; const y=3.1;
  line(s,0.6+w/2,y,0.6+7*(w+g)+w/2,y,HX.accent6?"5B6573":"5B6573",2);
  st.forEach((t,i)=>{ const x=0.6+i*(w+g); const hot=(i===2);
    s.addShape(S.ellipse,{x:x+w/2-0.3,y:y-0.3,w:0.6,h:0.6,fill:{color: hot?C.accent1:C.background2},line:{color: hot?C.accent1:C.accent2,width:2}});
    s.addText(String(i+1),{x:x+w/2-0.3,y:y-0.3,w:0.6,h:0.6,align:"center",valign:"middle",fontSize:16,bold:true,color: hot?C.background1:C.text1,margin:0,isTextBox:true});
    s.addText(t[0],{x,y:y+0.5,w,h:0.35,align:"center",fontSize:12,bold:true,fontFace:"Courier New",color: hot?C.accent1:C.text1,margin:0,isTextBox:true});
    s.addText(t[1],{x,y:y+0.9,w,h:1.2,align:"center",valign:"top",fontSize:13,color:C.text2,margin:0,isTextBox:true}); });
  s.addText("Trained and evaluated so far: stage 3, detection", { x:0.6, y:5.9, w:7, h:0.4, fontSize:16, bold:true, color: C.accent1, margin:0, isTextBox:true });
  s.addText("The remaining stages consume its boxes, so detector quality sets the ceiling.", { x:0.6, y:6.3, w:9, h:0.4, fontSize:14, color: C.text2, margin:0, isTextBox:true });
}

// ---------- 6 EXPORT FOLDER ----------
{
  const s = content("05  ·  DATA EXPORT", "What the dataset export contains", "dataset_export on Drive.");
  const it=[["annotations","VidOR labels for train and val"],["videos","Source clips"],["yolo_format","Frames and boxes in YOLO layout"],["tracks","Per-video object tracks"],["features","Extracted object features"],["relation_dataset","Object-pair relation samples"],["relation_dataset_corrected","Revised relation samples"],["dataset_manifest.json","Parts, fps and counts"]];
  it.forEach((t,i)=>{ const col=i%2,row=Math.floor(i/2); const x=0.6+col*6.15,y=1.95+row*1.15; card(s,x,y,5.95,0.95);
    s.addShape(S.ellipse,{x:x+0.25,y:y+0.28,w:0.4,h:0.4,fill:{color:[C.accent1,C.accent2,C.accent4,C.accent3][row]},line:{color:C.background2,width:0}});
    s.addText(t[0],{x:x+0.85,y:y+0.1,w:5,h:0.4,fontSize:15,bold:true,fontFace:"Courier New",color:C.text1,margin:0,isTextBox:true});
    s.addText(t[1],{x:x+0.85,y:y+0.5,w:5,h:0.35,fontSize:13,color:C.text2,margin:0,isTextBox:true}); });
}

// ---------- 7 DETECTOR SETUP ----------
{
  const s = content("06  ·  DETECTOR", "YOLOv8s fine-tuned in three runs", "Config from args.yaml of each run. Only run 3 has saved metrics.");
  const spec=[["Base model","yolov8s.pt, pretrained"],["Input size","640 px"],["Epochs","10 per run"],["Optimizer","auto, lr0 0.01"],["Augmentation","mosaic, flip 0.5, HSV"],["Precision","mixed (AMP), Colab GPU"]];
  spec.forEach((t,i)=>{ const y=1.95+i*0.68; s.addText(t[0],{x:0.6,y,w:2.3,h:0.55,fontSize:14,color:C.text2,valign:"middle",margin:0,isTextBox:true}); s.addText(t[1],{x:2.9,y,w:3.7,h:0.55,fontSize:16,bold:true,color:C.text1,valign:"middle",margin:0,isTextBox:true}); line(s,0.6,y+0.62,6.6,y+0.62,HX.grid,0.75); });
  const runs=[["Run 1","batch 16","weights saved",false],["Run 2","batch 32","weights saved",false],["Run 3","batch 32","metrics + curves logged",true]];
  runs.forEach((r,i)=>{ const y=1.95+i*1.4; card(s,7.2,y,5.5,1.2, r[3]?C.background2:C.background2);
    s.addShape(S.roundRect,{x:7.2,y,w:5.5,h:1.2,rectRadius:0.12,fill:{color:C.background2,transparency:100},line:{color: r[3]?C.accent1:C.accent6,width: r[3]?2:1,dashType:r[3]?"solid":"dash"}});
    s.addText(r[0],{x:7.5,y:y+0.15,w:2,h:0.45,fontSize:22,bold:true,fontFace:"Cambria",color: r[3]?C.accent1:C.text1,margin:0,isTextBox:true});
    s.addText(r[1],{x:7.5,y:y+0.65,w:2,h:0.4,fontSize:14,color:C.text2,margin:0,isTextBox:true});
    s.addText(r[2],{x:9.5,y:y+0.15,w:3,h:0.9,fontSize:15,color: r[3]?C.text1:C.text2,valign:"middle",align:"right",margin:0,isTextBox:true}); });
}

// ---------- 8 CURVES ----------
{
  const s = content("07  ·  RESULTS", "mAP stalls near 0.25 as validation loss stays flat", "Run 3, 10 epochs, from results.csv. Classification loss: train 2.97 to 1.04, validation 2.10 to 2.11.");
  s.addChart(pres.charts.LINE, [{name:"mAP50",labels:EP,values:D.map50},{name:"mAP50-95",labels:EP,values:D.map5095}], { x:0.6,y:1.8,w:6.0,h:4.3, ...chartBase("Mean average precision by epoch"), chartColors:[HX.amber,HX.teal], valAxisMinVal:0.1, valAxisMaxVal:0.3, valAxisLabelFormatCode:"0.00", catAxisTitle:"epoch", showCatAxisTitle:true, catAxisTitleColor:HX.mute, catAxisTitleFontSize:11 });
  s.addChart(pres.charts.LINE, [{name:"train cls loss",labels:EP,values:D.tcls},{name:"val cls loss",labels:EP,values:D.vcls}], { x:6.9,y:1.8,w:5.8,h:4.3, ...chartBase("Classification loss by epoch"), chartColors:[HX.coral,HX.peri], valAxisMinVal:0.5, valAxisMaxVal:3.5, valAxisLabelFormatCode:"0.0", catAxisTitle:"epoch", showCatAxisTitle:true, catAxisTitleColor:HX.mute, catAxisTitleFontSize:11 });
  s.addText("Training loss keeps falling while validation loss stays near 2.1 - a sign of over-fitting. Full run: 10 epochs, about 79 minutes on one Colab GPU.", { x:0.6, y:6.25, w:12.1, h:0.6, fontSize:14, color: C.text2, margin:0, valign:"top", isTextBox:true });
}

// ---------- 9 STATS + PR ----------
{
  const s = content("08  ·  RESULTS", "Peak 0.263 mAP50 at epoch 7", "Best mAP values across 10 epochs; precision and recall are final-epoch values. PR curve image is from the run-3 folder.");
  const st=[["0.263","mAP50, best (epoch 7)",C.accent1],["0.153","mAP50-95, best (epoch 7)",C.accent2],["0.363","precision, final epoch",C.accent3],["0.251","recall, final epoch",C.accent4]];
  st.forEach((t,i)=>{ const x=0.6+(i%2)*2.85,y=1.95+Math.floor(i/2)*2.3; card(s,x,y,2.6,2.05);
    s.addText(t[0],{x,y:y+0.2,w:2.6,h:1.0,align:"center",valign:"middle",fontSize:44,bold:true,fontFace:"Cambria",color:t[2],margin:0,isTextBox:true});
    s.addText(t[1],{x:x+0.15,y:y+1.25,w:2.3,h:0.6,align:"center",valign:"top",fontSize:13,color:C.text2,margin:0,isTextBox:true}); });
  s.addShape(S.roundRect,{x:6.6,y:1.95,w:6.1,h:4.07,rectRadius:0.1,fill:{color:"FFFFFF"},line:{color:"FFFFFF",width:0}});
  s.addImage({ path:"images/BoxPR_curve.png", x:6.6, y:1.95, w:6.1, h:4.07, altText:"Precision-recall curve, all classes, mAP@0.5 0.263" });
  s.addText("Precision-recall for all classes (blue) and each class (grey). Precision falls steeply as recall rises.", { x:6.6, y:6.15, w:6.1, h:0.6, fontSize:13, color: C.text2, margin:0, valign:"top", isTextBox:true });
}

// ---------- 10 CONFUSION + PREDICTIONS ----------
{
  const s = content("09  ·  ERROR ANALYSIS", "Strong on adults, thin on most other classes", "Confusion matrix and validation predictions from the run-3 folder. The adult class dominates; many classes barely register. Predicted boxes on validation frames show adult and baby at confidence 0.3 to 0.5.");
  s.addShape(S.roundRect,{x:0.6,y:1.95,w:5.0,h:3.75,rectRadius:0.1,fill:{color:"FFFFFF"},line:{color:"FFFFFF",width:0}});
  s.addImage({ path:"images/confusion_matrix.png", x:0.6, y:1.95, w:5.0, h:3.75, altText:"Confusion matrix over VidOR classes; the adult class dominates" });
  s.addImage({ path:"images/val_batch0_pred.jpg", x:5.9, y:1.95, w:6.8, h:3.9, altText:"Validation frames with predicted boxes labelled adult and baby with confidence scores" });
  s.addText("Confusion matrix: a dark adult cell, faint scatter everywhere else.", { x:0.6, y:5.85, w:5.0, h:0.7, fontSize:13, color: C.text2, margin:0, valign:"top", isTextBox:true });
  s.addText("Validation predictions: adult and baby are found, but confidence stays at 0.3 to 0.5.", { x:5.9, y:6.0, w:6.8, h:0.6, fontSize:13, color: C.text2, margin:0, valign:"top", isTextBox:true });
}

// ---------- 11 TEMPORAL GRAPH ----------
{
  const s = content("10  ·  TEMPORAL GRAPH", "Tracks turn per-frame boxes into a temporal graph", "Tracking assigns persistent IDs; graph and temporal modules connect nodes across frames.");
  const cols=[2.4,6.4,10.4]; const ids=[["person #1",HX.amber,2.4],["ball #2",HX.peri,4.1],["dog #3",HX.coral,5.8]];
  cols.forEach((cx,ci)=>{ s.addText("frame t"+(ci?"+"+ci:""),{x:cx-0.9,y:1.8,w:1.8,h:0.3,align:"center",fontSize:13,bold:true,color:C.text2,margin:0,isTextBox:true}); });
  // temporal links
  for (let c=0;c<2;c++) ids.forEach(r=>line(s,cols[c]+0.5,r[2],cols[c+1]-0.5,r[2],r[1],1.5,"dash"));
  cols.forEach((cx,ci)=>{
    line(s,cx,2.4,cx,4.1,HX.teal,2);
    if(ci!==1) line(s,cx,4.1,cx,5.8,HX.mute,1.25); 
    if(ci===1) line(s,cx,4.1,cx,5.8,HX.coral,2);
    ids.forEach(r=>node(s,cx,r[2],1.0,r[0],r[1],11));
  });
  s.addText("holds",{x:cols[0]+0.1,y:3.1,w:0.9,h:0.28,fontSize:11,italic:true,color:HX.teal,margin:0,isTextBox:true});
  s.addText("looks at",{x:cols[1]+0.1,y:4.8,w:1.0,h:0.28,fontSize:11,italic:true,color:HX.coral,margin:0,isTextBox:true});
  s.addText("Solid edges: relations inside a frame.   Dashed edges: the same object linked through time.", { x:0.6, y:6.45, w:12, h:0.4, fontSize:14, color: C.text2, margin:0, isTextBox:true });
}

// ---------- 12 RELATION DATA ----------
{
  const s = content("11  ·  RELATION DATA", "Relation samples come from tracks and features", "Data flow inferred from the dataset_export folders.");
  const bx=[["annotations","tracks","features"],["relation_dataset"],["relation_dataset_corrected"],["relation model"]];
  const cx=[0.6,5.0,7.6,10.3], cw=[3.3,2.2,2.3,2.4];
  // sources stacked
  ["annotations","tracks","features"].forEach((t,i)=>{ const y=2.1+i*1.2; card(s,0.6,y,2.9,0.95); s.addText(t,{x:0.6,y,w:2.9,h:0.95,align:"center",valign:"middle",fontSize:16,bold:true,fontFace:"Courier New",color:C.text1,margin:0,isTextBox:true}); line(s,3.5,y+0.47,4.3,3.55,HX.mute,1.5); });
  s.addShape(S.roundRect,{x:4.3,y:2.9,w:2.5,h:1.3,rectRadius:0.12,fill:{color:C.background2},line:{color:C.accent2,width:2}});
  s.addText("relation_dataset",{x:4.3,y:2.9,w:2.5,h:1.3,align:"center",valign:"middle",fontSize:14,bold:true,fontFace:"Courier New",color:C.text1,margin:0,isTextBox:true});
  line(s,6.8,3.55,7.3,3.55,HX.amber,2);
  s.addShape(S.roundRect,{x:7.3,y:2.9,w:3.1,h:1.3,rectRadius:0.12,fill:{color:C.background2},line:{color:C.accent1,width:2}});
  s.addText("relation_dataset_corrected",{x:7.3,y:2.9,w:3.1,h:1.3,align:"center",valign:"middle",fontSize:12,bold:true,fontFace:"Courier New",color:C.accent1,margin:0,isTextBox:true});
  line(s,10.4,3.55,10.9,3.55,HX.amber,2);
  s.addShape(S.roundRect,{x:10.9,y:2.9,w:1.8,h:1.3,rectRadius:0.12,fill:{color:C.background2},line:{color:C.accent4,width:2}});
  s.addText("predicate model",{x:10.9,y:2.9,w:1.8,h:1.3,align:"center",valign:"middle",fontSize:14,bold:true,color:C.text1,margin:0,isTextBox:true});
  s.addText("Two exports exist: the first relation set and a corrected version. The corrected one is the cleaner input for relation learning.", { x:4.3, y:4.9, w:8.4, h:0.9, fontSize:16, color: C.text2, margin:0, isTextBox:true });
}

// ---------- 13 CHALLENGES ----------
{
  const s = content("12  ·  LESSONS", "What slowed us down and what we learned", "GPU access limited the number of parts we could train before this review.");
  const it=[["GPU access","Limited Colab time capped how many parts we could train, so parts 6-8 waited.",C.accent1],["Short runs","Ten epochs per run is a small budget for 33k frames.",C.accent2],["Over-fitting","Validation loss stays flat while training loss falls.",C.accent3],["Low recall","The detector misses objects, which limits every later stage.",C.accent4]];
  it.forEach((t,i)=>{ const col=i%2,row=Math.floor(i/2); const x=0.6+col*6.15,y=1.95+row*2.3; card(s,x,y,5.95,2.05);
    s.addShape(S.ellipse,{x:x+0.3,y:y+0.3,w:0.6,h:0.6,fill:{color:t[2]},line:{color:t[2],width:0}});
    s.addText(String(i+1),{x:x+0.3,y:y+0.3,w:0.6,h:0.6,align:"center",valign:"middle",fontSize:18,bold:true,color:C.background1,margin:0,isTextBox:true});
    s.addText(t[0],{x:x+1.15,y:y+0.28,w:4.5,h:0.45,fontSize:20,bold:true,fontFace:"Cambria",color:C.text1,margin:0,isTextBox:true});
    s.addText(t[1],{x:x+1.15,y:y+0.85,w:4.5,h:1.0,fontSize:14,color:C.text2,valign:"top",margin:0,isTextBox:true}); });
}

// ---------- 14 REVIEW 3 ----------
{
  const s = content("REVIEW 3  ·  VIDOR PROGRESS", "VidOR training: five of eight parts complete", "Review 3 slide. Parts 1-5 are complete and reported. Parts 6-8 are currently training; their results will be added after evaluation.");
  s.addChart(pres.charts.DOUGHNUT, [{name:"Parts",labels:["Complete","In training"],values:[5,3]}], { x:0.5,y:1.8,w:4.4,h:4.4, holeSize:68, chartColors:[HX.amber,HX.grid], showLegend:false, showTitle:false, showValue:false, showPercent:false, showLabel:false, dataBorder:{pt:2,color:HX.bg} });
  s.addText("5/8", { x:0.5, y:3.45, w:4.4, h:0.9, align:"center", valign:"middle", fontSize:54, bold:true, fontFace:"Cambria", color: C.text1, margin:0, isTextBox:true });
  s.addText("parts complete", { x:0.5, y:4.3, w:4.4, h:0.4, align:"center", fontSize:14, color: C.text2, margin:0, isTextBox:true });
  const rows=[["Parts","Status","Result"],["1 - 5","Complete","Detector: best mAP50 0.263, mAP50-95 0.153"],["6","Training","Pending evaluation"],["7","Training","Pending evaluation"],["8","Training","Pending evaluation"]];
  rows.forEach((r,i)=>{ const y=1.95+i*0.72; const hdr=i===0; const done=i===1;
    if(!hdr) card(s,5.3,y,7.4,0.62, C.background2);
    s.addText(r[0],{x:5.5,y,w:1.0,h:0.62,valign:"middle",fontSize:hdr?12:16,bold:true,charSpacing:hdr?3:0,color: hdr?C.text2:C.text1,margin:0,isTextBox:true});
    s.addText(r[1],{x:6.6,y,w:1.5,h:0.62,valign:"middle",fontSize:hdr?12:15,bold:true,charSpacing:hdr?3:0,color: hdr?C.text2:(done?C.accent1:C.accent2),margin:0,isTextBox:true});
    s.addText(r[2],{x:8.2,y,w:4.4,h:0.62,valign:"middle",fontSize:hdr?12:14,charSpacing:hdr?3:0,color: hdr?C.text2:C.text1,margin:0,isTextBox:true}); });
  s.addText([{text:"Next in this review: ",options:{bold:true,color:HX.amber}},{text:"parts 6-8 results, compared against the parts 1-5 baseline of 0.263 mAP50, go in as each run finishes.",options:{color:HX.text}}], { x:5.3, y:5.75, w:7.4, h:0.8, fontSize:15, margin:0, valign:"top", isTextBox:true });
}

// ---------- 15 NEXT ----------
{
  const s = content("13  ·  NEXT", "Finish the parts, then close the loop", "Roadmap.");
  const st=[["Finish parts 6-8","Complete training and evaluate each part",HX.amber],["Retrain on all eight","Re-fit the detector with the full, more varied data",HX.teal],["Longer runs","More epochs, watch validation loss for early stopping",HX.peri],["Run the full graph","Detection to tracking to temporal relations, end to end",HX.coral]];
  line(s,1.3,3.2,11.8,3.2,"5B6573",2);
  st.forEach((t,i)=>{ const x=0.6+i*3.1; 
    s.addShape(S.ellipse,{x:x+0.35,y:2.85,w:0.7,h:0.7,fill:{color:t[2]},line:{color:t[2],width:0}});
    s.addText(String(i+1),{x:x+0.35,y:2.85,w:0.7,h:0.7,align:"center",valign:"middle",fontSize:20,bold:true,color:C.background1,margin:0,isTextBox:true});
    s.addText(t[0],{x,y:3.85,w:2.8,h:0.8,fontSize:20,bold:true,fontFace:"Cambria",color:C.text1,valign:"top",margin:0,isTextBox:true});
    s.addText(t[1],{x,y:4.7,w:2.8,h:1.2,fontSize:14,color:C.text2,valign:"top",margin:0,isTextBox:true}); });
  s.addText("Thank you", { x:0.6, y:6.1, w:6, h:0.6, fontSize:28, bold:true, fontFace:"Cambria", color: C.accent1, margin:0, isTextBox:true });
}

(async()=>{
  await pres.writeFile({ fileName: "Dynamic_Scene_Graph_Review3.pptx" });
  await applyTheme("Dynamic_Scene_Graph_Review3.pptx", THEME);
  console.log("done", n, "slides");
})();
