const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, ImageRun,
  PageBreak, BorderStyle, TableOfContents, LevelFormat,
} = require("docx");

const ASSETS = __dirname + "/assets";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function numbered(text, instance) {
  return new Paragraph({
    text,
    numbering: { reference: "numbered-list", level: 0 },
    spacing: { after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}
function image(path, width, height) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 150, after: 80 },
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function simpleTable(headerRow, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const mkCell = (text, bold, shade) =>
    new TableCell({
      width: { size: 0, type: WidthType.AUTO },
      shading: shade ? { type: ShadingType.CLEAR, fill: "D9E8F5" } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text: String(text), bold })] })],
    });

  const header = new TableRow({
    children: headerRow.map((t) => mkCell(t, true, true)),
  });
  const body = rows.map(
    (r) => new TableRow({ children: r.map((t) => mkCell(t, false, false)) })
  );
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [header, ...body],
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "numbered-list",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START }],
      },
    ],
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        // ---------------- Title Page ----------------
        new Paragraph({ text: "", spacing: { before: 1800 } }),
        new Paragraph({
          children: [new TextRun({ text: "DEEPFAKE DETECTION USING", bold: true, size: 44 })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "VISION TRANSFORMER (ViT)", bold: true, size: 44 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 400 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "A Major Project Report", size: 28, italics: true })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Submitted in partial fulfilment of the requirements for the degree of", size: 22 })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "Master of Computer Applications (MCA)", size: 24, bold: true })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 600 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "[Student Name]", size: 22 })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "[University / Institute Name]", size: 22 })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "[Month, Year]", size: 22 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),
        pageBreak(),

        // ---------------- Table of Contents ----------------
        h1("Table of Contents"),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        pageBreak(),

        // ---------------- Abstract ----------------
        h1("1. Abstract"),
        p(
          "Deepfakes — synthetically generated or manipulated facial imagery — pose an escalating " +
          "threat to information integrity, personal privacy and public trust. This project presents " +
          "a complete, end-to-end deepfake image classification system built on a pretrained Vision " +
          "Transformer (ViT-B/16), fine-tuned using transfer learning rather than trained from scratch. " +
          "The pipeline covers dataset preparation, exploratory data analysis, preprocessing, " +
          "augmentation, model fine-tuning, comprehensive evaluation, explainable-AI visualisation " +
          "(Grad-CAM and attention rollout), and deployment through an interactive Streamlit web " +
          "application. The system classifies an input face image (or sampled video frames) as " +
          "Real or Fake, reporting a confidence score alongside a visual explanation of which image " +
          "regions influenced the decision."
        ),

        // ---------------- Problem Statement ----------------
        h1("2. Problem Statement"),
        p(
          "The rapid advancement of generative models (GANs, diffusion models, autoencoder-based face " +
          "swapping) has made it trivial to produce highly convincing fake images and videos of real " +
          "people. Such content can be weaponised for misinformation, fraud, harassment and reputational " +
          "damage. Manual detection is unreliable at scale, and many existing automated detectors rely on " +
          "convolutional architectures that struggle to generalise across manipulation methods. There is " +
          "a need for an accurate, explainable, and easily deployable deepfake detector that can be " +
          "trained efficiently even with a moderate amount of labelled data, by leveraging large-scale " +
          "pretrained vision models."
        ),

        // ---------------- Objectives ----------------
        h1("3. Objectives"),
        bullet("To fine-tune a pretrained Vision Transformer (ViT-B/16) for binary Real vs. Fake image classification using transfer learning."),
        bullet("To design a modular, reproducible training and evaluation pipeline with centralised configuration."),
        bullet("To perform thorough exploratory data analysis and apply appropriate data augmentation to improve generalisation."),
        bullet("To evaluate the trained model using a comprehensive suite of classification metrics."),
        bullet("To provide model explainability via Grad-CAM and attention-rollout visualisations."),
        bullet("To deploy the trained model through an interactive, user-friendly Streamlit web application."),
        bullet("To optionally extend detection to video by aggregating frame-level predictions."),

        // ---------------- Literature Review ----------------
        h1("4. Literature Review"),
        p(
          "Early deepfake detectors predominantly used convolutional neural networks (CNNs) such as " +
          "XceptionNet and EfficientNet, trained on datasets like FaceForensics++, exploiting low-level " +
          "artefacts introduced by GAN-based generation (e.g., blending boundaries, frequency-domain " +
          "inconsistencies, and unnatural textures). While effective within-dataset, these CNN-based " +
          "approaches often generalise poorly to manipulation methods unseen during training."
        ),
        p(
          "Vision Transformers (ViT), introduced by Dosovitskiy et al. (2020), model an image as a " +
          "sequence of fixed-size patches processed through self-attention layers, enabling them to " +
          "capture long-range dependencies across the entire image rather than local receptive fields " +
          "alone. Subsequent work has shown that ViT and hybrid CNN-Transformer architectures, especially " +
          "when pretrained on large image corpora (ImageNet-21k) and fine-tuned via transfer learning, " +
          "achieve competitive or superior deepfake-detection accuracy while requiring comparatively less " +
          "task-specific data than training a transformer from scratch."
        ),
        p(
          "Explainability techniques such as Grad-CAM (Selvaraju et al., 2017), originally developed for " +
          "CNNs, have been adapted to transformer architectures via gradient-weighted patch-token " +
          "activations, while attention-rollout (Abnar & Zuidema, 2020) aggregates multi-layer " +
          "self-attention maps to highlight the image regions a ViT attends to most strongly — both are " +
          "used in this project to make the model's predictions interpretable to end users."
        ),

        // ---------------- Dataset ----------------
        h1("5. Dataset"),
        p(
          "The system expects a binary-labelled face-image dataset organised in the standard " +
          "torchvision ImageFolder layout, split into train / validation / test partitions, each " +
          "containing a 'real' and a 'fake' subfolder:"
        ),
        new Paragraph({
          children: [new TextRun({
            text: "dataset/{train, validation, test}/{real, fake}/*.jpg",
            font: "Consolas", size: 20,
          })],
          spacing: { after: 200 },
        }),
        p(
          "This structure is compatible with widely used public deepfake datasets (e.g., FaceForensics++, " +
          "Celeb-DF, DFDC, 140k Real and Fake Faces) after reorganising their images into the layout " +
          "above. A recommended split ratio is 70% train / 15% validation / 15% test, with balanced " +
          "class counts to avoid biasing the classifier."
        ),
        h2("5.1 Preprocessing"),
        bullet("Resize every image to 224 x 224 pixels (matching ViT-B/16's patch grid of 14 x 14 patches of size 16 x 16)."),
        bullet("Normalise pixel values using the mean/std expected by the pretrained backbone."),
        bullet("Convert to PyTorch tensors for model consumption."),
        h2("5.2 Data Augmentation (training split only)"),
        bullet("Random Horizontal Flip"),
        bullet("Random Rotation"),
        bullet("Random Crop (with padding)"),
        bullet("Color Jitter (brightness, contrast, saturation, hue)"),
        bullet("Random Erasing"),

        // ---------------- Architecture Diagram ----------------
        pageBreak(),
        h1("6. System / Model Architecture"),
        p(
          "The classifier reuses a pretrained ViT-B/16 backbone (google/vit-base-patch16-224) with its " +
          "original ImageNet classification head removed. The backbone is initially frozen so that only " +
          "a lightweight custom classifier head is trained; the last few transformer blocks can " +
          "optionally be unfrozen for a subsequent fine-tuning stage."
        ),
        image(`${ASSETS}/architecture_diagram.png`, 600, 230),
        caption("Figure 1: Model architecture — pretrained ViT-B/16 with a custom classifier head."),

        // ---------------- Methodology ----------------
        h1("7. Methodology"),
        p(
          "The project follows a standard supervised transfer-learning workflow, illustrated in Figure " +
          "2 below and implemented across config.py, src/dataset.py, src/model.py, train.py, test.py, " +
          "predict.py and app.py."
        ),
        image(`${ASSETS}/methodology_flowchart.png`, 330, 500),
        caption("Figure 2: End-to-end project methodology."),

        // ---------------- Algorithms ----------------
        h1("8. Algorithms"),
        h2("8.1 Transfer Learning Fine-Tuning"),
        numbered("Load ViT-B/16 pretrained on ImageNet-21k."),
        numbered("Freeze all backbone parameters; attach a new randomly-initialised classifier head."),
        numbered("Train the head with a higher learning rate using AdamW."),
        numbered("Optionally unfreeze the last N transformer blocks and continue training at a lower learning rate for full fine-tuning."),
        h2("8.2 Optimisation"),
        bullet("Optimizer: AdamW with discriminative learning rates (head vs. backbone) and weight decay."),
        bullet("Loss: Cross-Entropy Loss with label smoothing."),
        bullet("Scheduler: Cosine Annealing Learning Rate."),
        bullet("Mixed precision (AMP) training automatically enabled when a CUDA GPU is available."),
        bullet("Early stopping on validation loss with patience, plus best-checkpoint saving."),
        h2("8.3 Explainability"),
        bullet("Grad-CAM: gradients of the predicted class w.r.t. the last encoder block's patch-token activations, reshaped into a 14x14 heatmap and upsampled to the original image size."),
        bullet("Attention Rollout: multiplies the [CLS]-token attention weights across all encoder layers to visualise which patches most influenced the final representation."),

        // ---------------- Flowchart (referenced above) ----------------
        h1("9. Flowchart"),
        p("See Figure 2 (Section 7) for the complete methodology flowchart, from dataset collection through to Streamlit deployment."),

        // ---------------- Screenshots ----------------
        pageBreak(),
        h1("10. Screenshots"),
        p(
          "[Insert screenshots here after running `streamlit run app.py` on your trained model: " +
          "(1) the image-upload screen, (2) the Real/Fake prediction card with confidence score, " +
          "(3) the Grad-CAM / attention-rollout explainability panel, and (4) the optional video-analysis view.]",
          { italics: true }
        ),

        // ---------------- Results ----------------
        h1("11. Results"),
        p(
          "This repository ships complete, runnable code but intentionally does not bundle a dataset " +
          "or pretrained weights. After placing your dataset inside dataset/ and running `python " +
          "train.py` followed by `python test.py`, populate the table and figures below with the " +
          "values written to outputs/test_metrics.json and outputs/plots/."
        ),
        simpleTable(
          ["Metric", "Value"],
          [
            ["Accuracy", "[fill after training]"],
            ["Precision", "[fill after training]"],
            ["Recall", "[fill after training]"],
            ["F1 Score", "[fill after training]"],
            ["ROC-AUC", "[fill after training]"],
          ],
          [5000, 5000]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        // ---------------- Graphs ----------------
        h1("12. Graphs"),
        p(
          "[Insert outputs/plots/training_curves.png, outputs/plots/confusion_matrix.png, " +
          "outputs/plots/roc_curve.png and outputs/plots/pr_curve.png here once training/evaluation " +
          "has been run on your dataset.]",
          { italics: true }
        ),

        // ---------------- Evaluation Metrics explanation ----------------
        h1("13. Evaluation Metrics"),
        bullet("Accuracy: proportion of correctly classified images overall."),
        bullet("Precision: of the images predicted 'real', the fraction that are actually real (i.e., how trustworthy a 'real' prediction is)."),
        bullet("Recall: of all actually-real images, the fraction correctly identified (i.e., how many real images are not missed)."),
        bullet("F1 Score: harmonic mean of precision and recall, balancing both concerns."),
        bullet("ROC-AUC: area under the Receiver Operating Characteristic curve, summarising the model's ability to discriminate real from fake across all classification thresholds."),
        bullet("Confusion Matrix: breakdown of true/false positives and negatives, useful for spotting class-specific weaknesses."),

        // ---------------- Future Scope ----------------
        h1("14. Future Scope"),
        bullet("Extend to native video deepfake detection using temporal/3D architectures (e.g., Video ViT, 3D-CNN)."),
        bullet("Evaluate and improve robustness against compression, noise and adversarial perturbations."),
        bullet("Explore ensembling ViT with CNN backbones (e.g., EfficientNet) for improved generalisation."),
        bullet("Integrate automatic face detection/alignment (e.g., MTCNN) for robust in-the-wild inference."),
        bullet("Expose the model via a REST API (FastAPI) for integration into third-party platforms."),
        bullet("Incorporate active learning to continuously retrain on newly labelled hard examples."),

        // ---------------- References ----------------
        h1("15. References"),
        p("1. Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929."),
        p("2. Rossler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV."),
        p("3. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV."),
        p("4. Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL."),
        p("5. Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR."),
        p("6. Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. HuggingFace / EMNLP System Demonstrations."),
        p("7. Dolhansky, B. et al. (2020). The DeepFake Detection Challenge (DFDC) Dataset. arXiv:2006.07397."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(__dirname + "/Project_Report.docx", buffer);
  console.log("Report written to reports/Project_Report.docx");
});
