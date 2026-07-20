const pptxgen = require("pptxgenjs");

const ASSETS = __dirname + "/assets";

const COLORS = {
  bg: "FFFFFF",
  dark: "1B2430",
  accent: "2E6FE6",
  accentDark: "1D4FB8",
  fake: "E74C3C",
  real: "27AE60",
  gray: "6B7684",
  cardBg: "F4F7FC",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in

function baseSlide() {
  const s = pres.addSlide();
  s.background = { color: COLORS.bg };
  return s;
}

function titleText(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.6, y: 0.4, w: 12.1, h: 0.9,
    fontSize: 30, bold: true, color: COLORS.dark, fontFace: "Calibri",
    ...opts,
  });
}

function footer(slide, pageNum) {
  slide.addText("Deepfake Detection using Vision Transformer (ViT)", {
    x: 0.6, y: 7.15, w: 8, h: 0.3, fontSize: 10, color: COLORS.gray,
  });
  slide.addText(String(pageNum), {
    x: 12.4, y: 7.15, w: 0.4, h: 0.3, fontSize: 10, color: COLORS.gray, align: "right",
  });
}

// ---------------------------------------------------------------------- //
// Slide 1: Title
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  s.background = { color: COLORS.dark };
  s.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: COLORS.dark },
  });
  s.addText("DEEPFAKE DETECTION", {
    x: 0.8, y: 2.3, w: 11.7, h: 1.0, fontSize: 44, bold: true, color: "FFFFFF",
  });
  s.addText("using Vision Transformer (ViT-B/16)", {
    x: 0.8, y: 3.2, w: 11.7, h: 0.7, fontSize: 24, color: COLORS.accent,
  });
  s.addText("MCA Major Project  |  Transfer Learning for Real vs. Fake Image Classification", {
    x: 0.8, y: 4.1, w: 11.7, h: 0.5, fontSize: 15, color: "C7CDD6",
  });
  s.addText("[Student Name]   ·   [Institute Name]   ·   [Month, Year]", {
    x: 0.8, y: 6.4, w: 11.7, h: 0.5, fontSize: 13, color: "8A93A0",
  });
}

// ---------------------------------------------------------------------- //
// Slide 2: Problem Statement
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Problem Statement");
  s.addText(
    "Generative models (GANs, diffusion, autoencoder face-swapping) can now produce highly " +
    "convincing fake facial imagery, enabling misinformation, fraud and reputational harm.",
    { x: 0.6, y: 1.5, w: 12.1, h: 1.0, fontSize: 16, color: COLORS.gray }
  );

  const cards = [
    { t: "Scale", d: "Manual review cannot keep pace with the volume of synthetic media generated daily." },
    { t: "Generalisation", d: "CNN-only detectors often overfit to artefacts of a specific manipulation method." },
    { t: "Trust", d: "Detections need to be explainable, not just a black-box score." },
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.8, w: 3.85, h: 3.3, rectRadius: 0.12,
      fill: { color: COLORS.cardBg }, line: { type: "none" },
    });
    s.addText(c.t, { x: x + 0.3, y: 3.05, w: 3.25, h: 0.5, fontSize: 18, bold: true, color: COLORS.accentDark });
    s.addText(c.d, { x: x + 0.3, y: 3.65, w: 3.25, h: 2.2, fontSize: 13.5, color: COLORS.dark });
  });
  footer(s, 2);
}

// ---------------------------------------------------------------------- //
// Slide 3: Objectives
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Objectives");
  const objectives = [
    "Fine-tune a pretrained ViT-B/16 for Real vs. Fake classification via transfer learning",
    "Build a modular, reproducible, fully-configurable training & evaluation pipeline",
    "Perform thorough EDA and apply targeted data augmentation",
    "Evaluate using accuracy, precision, recall, F1 and ROC-AUC",
    "Provide explainability via Grad-CAM and attention rollout",
    "Deploy an interactive Streamlit web application",
  ];
  const textItems = objectives.map((o, i) => ({
    text: o, options: { bullet: { code: "25CF" }, breakLine: i !== objectives.length - 1, color: COLORS.dark, fontSize: 16 },
  }));
  s.addText(textItems, { x: 0.7, y: 1.7, w: 11.6, h: 5, paraSpaceAfter: 16, valign: "top" });
  footer(s, 3);
}

// ---------------------------------------------------------------------- //
// Slide 4: Dataset
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Dataset");
  s.addText("Standard binary ImageFolder layout — compatible with FaceForensics++, Celeb-DF, DFDC, 140k Real & Fake Faces, etc.", {
    x: 0.6, y: 1.4, w: 12.1, h: 0.6, fontSize: 15, color: COLORS.gray,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 2.2, w: 6.0, h: 4.2, rectRadius: 0.1,
    fill: { color: "0F1720" }, line: { type: "none" },
  });
  s.addText(
    [
      { text: "dataset/\n", options: { color: "7FD1FF", bold: true } },
      { text: "├── train/{real, fake}\n", options: { color: "E6E6E6" } },
      { text: "├── validation/{real, fake}\n", options: { color: "E6E6E6" } },
      { text: "└── test/{real, fake}\n", options: { color: "E6E6E6" } },
    ],
    { x: 0.9, y: 2.5, w: 5.4, h: 1.5, fontSize: 15, fontFace: "Consolas" }
  );
  s.addText("Recommended split: 70% train / 15% validation / 15% test, class-balanced.", {
    x: 0.9, y: 4.1, w: 5.4, h: 1.8, fontSize: 13, color: "AAB4C0", fontFace: "Consolas",
  });

  const facts = [
    ["Image size", "224 x 224 (resized)"],
    ["Classes", "Real, Fake"],
    ["Normalisation", "Backbone-matched mean/std"],
    ["Augmentation", "Flip, rotate, crop, jitter, erasing"],
  ];
  let fy = 2.3;
  facts.forEach(([k, v]) => {
    s.addText(k, { x: 7.0, y: fy, w: 2.4, h: 0.6, fontSize: 14, bold: true, color: COLORS.accentDark });
    s.addText(v, { x: 9.4, y: fy, w: 3.2, h: 0.6, fontSize: 14, color: COLORS.dark });
    fy += 0.95;
  });
  footer(s, 4);
}

// ---------------------------------------------------------------------- //
// Slide 5: Model Architecture
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Model Architecture");
  s.addImage({ path: `${ASSETS}/architecture_diagram.png`, x: 0.6, y: 1.6, w: 12.1, h: 4.63 });
  footer(s, 5);
}

// ---------------------------------------------------------------------- //
// Slide 6: Methodology Flowchart
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Project Methodology");
  s.addImage({ path: `${ASSETS}/methodology_flowchart.png`, x: 3.9, y: 1.35, w: 3.6, h: 5.6 });
  const notes = [
    "Data collection & organisation",
    "EDA & preprocessing",
    "Augmentation pipeline",
    "Transfer-learning fine-tuning",
    "Evaluation & explainability",
    "Streamlit deployment",
  ];
  s.addText(notes.map((n, i) => ({ text: n, options: { bullet: { code: "2022" }, breakLine: i !== notes.length - 1 } })),
    { x: 8.0, y: 1.9, w: 4.7, h: 4, fontSize: 15, color: COLORS.dark, paraSpaceAfter: 14 });
  footer(s, 6);
}

// ---------------------------------------------------------------------- //
// Slide 7: Training Configuration
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Training Configuration");
  const rows = [
    [{ text: "Component", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accentDark } } },
     { text: "Choice", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accentDark } } }],
    ["Backbone", "ViT-B/16, pretrained (ImageNet-21k), backbone frozen initially"],
    ["Optimizer", "AdamW, discriminative LR (head vs. backbone)"],
    ["Loss", "Cross-Entropy with label smoothing"],
    ["Scheduler", "Cosine Annealing LR"],
    ["Precision", "Automatic Mixed Precision (AMP) when CUDA available"],
    ["Regularisation", "Early stopping (patience-based) + best-checkpoint saving"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.6, w: 12.1, h: 4.8,
    colW: [3.6, 8.5],
    fontSize: 14, color: COLORS.dark, border: { type: "solid", color: "DCE3EC", pt: 1 },
    autoPage: false,
  });
  footer(s, 7);
}

// ---------------------------------------------------------------------- //
// Slide 8: Evaluation Metrics
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Evaluation Metrics");
  const metrics = [
    ["Accuracy", "Overall proportion of correct predictions"],
    ["Precision", "Trustworthiness of a 'real' prediction"],
    ["Recall", "Coverage of actually-real images correctly found"],
    ["F1 Score", "Harmonic mean of precision and recall"],
    ["ROC-AUC", "Discrimination ability across all thresholds"],
    ["Confusion Matrix", "Full breakdown of TP / FP / TN / FN"],
  ];
  metrics.forEach((m, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15;
    const y = 1.6 + row * 1.7;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: 5.85, h: 1.45, rectRadius: 0.08,
      fill: { color: COLORS.cardBg }, line: { type: "none" },
    });
    s.addText(m[0], { x: x + 0.25, y: y + 0.12, w: 5.35, h: 0.4, fontSize: 15, bold: true, color: COLORS.accentDark });
    s.addText(m[1], { x: x + 0.25, y: y + 0.55, w: 5.35, h: 0.8, fontSize: 12.5, color: COLORS.dark });
  });
  footer(s, 8);
}

// ---------------------------------------------------------------------- //
// Slide 9: Explainable AI
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Explainable AI");
  s.addText("Grad-CAM", { x: 0.9, y: 1.7, w: 5.6, h: 0.5, fontSize: 20, bold: true, color: COLORS.accentDark });
  s.addText(
    "Gradients of the predicted class w.r.t. the last encoder block's patch-token activations " +
    "are reshaped into a 14x14 heatmap and upsampled onto the original image.",
    { x: 0.9, y: 2.3, w: 5.6, h: 2.2, fontSize: 14, color: COLORS.dark }
  );
  s.addText("Attention Rollout", { x: 6.9, y: 1.7, w: 5.6, h: 0.5, fontSize: 20, bold: true, color: COLORS.accentDark });
  s.addText(
    "Multiplies [CLS]-token attention weights across every encoder layer to reveal which image " +
    "patches most influenced the model's final decision.",
    { x: 6.9, y: 2.3, w: 5.6, h: 2.2, fontSize: 14, color: COLORS.dark }
  );
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 4.6, w: 11.6, h: 2.0, rectRadius: 0.1,
    fill: { color: "FDF3E3" }, line: { type: "none" },
  });
  s.addText("Both overlays render live inside the Streamlit app for every uploaded image, so users can see exactly why a prediction was made.", {
    x: 1.3, y: 5.1, w: 10.8, h: 1.0, fontSize: 14.5, color: COLORS.dark, italic: true,
  });
  footer(s, 9);
}

// ---------------------------------------------------------------------- //
// Slide 10: Streamlit Application
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Streamlit Web Application");
  const features = [
    "Drag-and-drop image / video upload with preview",
    "Real / Fake prediction with confidence score & probability chart",
    "Grad-CAM and attention-rollout heatmap overlays",
    "Loading spinners and graceful error handling",
    "Optional frame-wise video analysis with overall verdict",
  ];
  s.addText(features.map((f, i) => ({ text: f, options: { bullet: { code: "25CF" }, breakLine: i !== features.length - 1 } })),
    { x: 0.7, y: 1.7, w: 11.6, h: 4.5, fontSize: 17, color: COLORS.dark, paraSpaceAfter: 18 });
  footer(s, 10);
}

// ---------------------------------------------------------------------- //
// Slide 11: Results (template)
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Results");
  s.addText(
    "This project ships complete, runnable code without a bundled dataset. Populate the figures " +
    "below with your own outputs/test_metrics.json after running train.py and test.py.",
    { x: 0.6, y: 1.4, w: 12.1, h: 0.9, fontSize: 14, color: COLORS.gray, italic: true }
  );
  const rows = [
    [{ text: "Metric", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accentDark } } },
     { text: "Value", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accentDark } } }],
    ["Accuracy", "[fill after training]"],
    ["Precision", "[fill after training]"],
    ["Recall", "[fill after training]"],
    ["F1 Score", "[fill after training]"],
    ["ROC-AUC", "[fill after training]"],
  ];
  s.addTable(rows, {
    x: 2.9, y: 2.5, w: 7.5, h: 3.6,
    colW: [3.75, 3.75],
    fontSize: 16, color: COLORS.dark, border: { type: "solid", color: "DCE3EC", pt: 1 },
    autoPage: false,
  });
  footer(s, 11);
}

// ---------------------------------------------------------------------- //
// Slide 12: Future Scope
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  titleText(s, "Future Scope");
  const items = [
    "Native video deepfake detection (Video ViT / 3D-CNN)",
    "Robustness evaluation against compression & adversarial noise",
    "Ensembling ViT with CNN backbones (e.g., EfficientNet)",
    "Automatic face detection & alignment for in-the-wild inference",
    "REST API deployment (FastAPI) alongside the Streamlit demo",
    "Active learning to continuously retrain on hard examples",
  ];
  items.forEach((item, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 1.7 + row * 1.6;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: 5.85, h: 1.35, rectRadius: 0.08,
      fill: { color: COLORS.cardBg }, line: { type: "none" },
    });
    s.addText(item, { x: x + 0.3, y: y + 0.2, w: 5.25, h: 0.95, fontSize: 14, color: COLORS.dark, valign: "middle" });
  });
  footer(s, 12);
}

// ---------------------------------------------------------------------- //
// Slide 13: Thank You
// ---------------------------------------------------------------------- //
{
  const s = baseSlide();
  s.background = { color: COLORS.dark };
  s.addText("Thank You", { x: 0.8, y: 2.9, w: 11.7, h: 1.0, fontSize: 40, bold: true, color: "FFFFFF" });
  s.addText("Questions & Discussion", { x: 0.8, y: 3.9, w: 11.7, h: 0.6, fontSize: 18, color: COLORS.accent });
  s.addText("Deepfake Detection using Vision Transformer (ViT-B/16)  ·  MCA Major Project", {
    x: 0.8, y: 6.6, w: 11.7, h: 0.4, fontSize: 12, color: "8A93A0",
  });
}

pres.writeFile({ fileName: __dirname + "/Project_Presentation.pptx" }).then(() => {
  console.log("Presentation written to reports/Project_Presentation.pptx");
});
