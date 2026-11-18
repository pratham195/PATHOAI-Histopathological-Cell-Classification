import { useState } from "react";
import "./ModelEvaluation.css";

const SUMMARY = [
  {
    key: "binary",
    title: "CNN · isCancerous",
    tag: "BINARY",
    accent: "cyan",
    accuracy: "91%",
    macroF1: 0.890,
    stats: [
      ["Test macro-F1", "89.0%"],
      ["Best validation F1", "89.5%"],
      ["Best epoch", "5 / 15"],
      ["Test samples", "5,112"],
    ],
    note: "Cancerous precision, recall and F1-score are approximately 0.84.",
  },
  {
    key: "cell",
    title: "CNN · Cell Phenotyping",
    tag: "5 CLASSES",
    accent: "violet",
    accuracy: "76%",
    macroF1: 0.635,
    stats: [
      ["Test macro-F1", "63.5%"],
      ["Best validation F1", "71.7%"],
      ["Best epoch", "15 / 15"],
      ["Test samples", "5,112"],
    ],
    note: "Dead and Inflammatory have weaker F1-scores and require careful interpretation.",
  },
];

const ARCH_BLOCKS = [
  { label: "INPUT", value: "64 × 64 × 3", sub: "RGB image" },
  { label: "BLOCK 01", value: "16 channels", sub: "32 × 32" },
  { label: "BLOCK 02", value: "32 channels", sub: "16 × 16" },
  { label: "BLOCK 03", value: "64 channels", sub: "8 × 8" },
  { label: "FLATTEN", value: "4,096", sub: "64 × 8 × 8" },
  { label: "FC + DROPOUT", value: "64 neurons", sub: "ReLU · 0.3" },
  { label: "OUTPUT", value: "2 or 5", sub: "Softmax classes" },
];

const HYPERPARAMS = [
  ["Optimizer", "Adam"],
  ["Learning rate", "0.001"],
  ["Batch size", "64"],
  ["Epoch limit", "15"],
  ["Dropout", "0.3"],
  ["Loss function", "Weighted cross-entropy"],
];

const TRAINING_CONFIG = [
  ["Architecture", "Custom SmallCNN", "Custom SmallCNN"],
  ["Output classes", "2", "5"],
  ["Training samples", "25,338", "25,338"],
  ["Validation samples", "5,261", "5,261"],
  ["Test samples", "5,112", "5,112"],
  ["Augmentation", "Flips + rotation ≤ 20°", "Flips + rotation ≤ 20°"],
  ["Checkpoint selection", "Best validation macro-F1", "Best validation macro-F1"],
  ["Best checkpoint", "Epoch 5", "Epoch 15"],
  ["Saved model", "cnn_iscancerous.pt", "cnn_celltype.pt"],
  ["Label encoder", "—", "celltype_label_encoder.joblib"],
];

const BINARY_RESULTS = [
  { name: "Non-cancerous", precision: 0.94, recall: 0.94, f1: 0.94, n: 3712, color: "#34d399" },
  { name: "Cancerous", precision: 0.84, recall: 0.84, f1: 0.84, n: 1400, color: "#fb7185" },
];

const CELL_RESULTS = [
  { name: "Connective", precision: 0.75, recall: 0.82, f1: 0.79, n: 1874, color: "#a78bfa" },
  { name: "Dead", precision: 0.09, recall: 0.62, f1: 0.16, n: 21, color: "#94a3b8" },
  { name: "Epithelial", precision: 0.83, recall: 0.76, f1: 0.79, n: 811, color: "#34d399" },
  { name: "Inflammatory", precision: 0.76, recall: 0.50, f1: 0.60, n: 1006, color: "#fbbf24" },
  { name: "Neoplastic", precision: 0.81, recall: 0.87, f1: 0.84, n: 1400, color: "#fb7185" },
];

const CLASSES = [
  "Connective",
  "Dead",
  "Epithelial",
  "Inflammatory",
  "Neoplastic",
];

const BINARY_CNN_CONFUSION = [
  { actual: "Non-cancerous", counts: [3483, 229] },
  { actual: "Cancerous", counts: [219, 1181] },
];

const CELL_CNN_CONFUSION = [
  { actual: "Connective", counts: [1540, 54, 62, 111, 107] },
  { actual: "Dead", counts: [7, 13, 1, 0, 0] },
  { actual: "Epithelial", counts: [38, 0, 617, 25, 131] },
  { actual: "Inflammatory", counts: [356, 63, 30, 505, 52] },
  { actual: "Neoplastic", counts: [105, 15, 34, 27, 1219] },
];

const RF_RESULTS = [
  { name: "Non-cancerous", precision: 0.87, recall: 0.76, f1: 0.81, n: 3712, color: "#34d399" },
  { name: "Cancerous", precision: 0.53, recall: 0.71, f1: 0.60, n: 1400, color: "#fb7185" },
];

const RF_CONFUSION = [
  { actual: "Non-cancerous", counts: [2816, 896] },
  { actual: "Cancerous", counts: [409, 991] },
];

const RF_PARAMS = [
  ["Model", "Random Forest"],
  ["Trees", "200"],
  ["Max depth", "12"],
  ["Features per split", "√6 ≈ 2"],
  ["Class weights", "Balanced"],
  ["Split criterion", "Gini impurity"],
  ["Random seed", "42"],
  ["Parallelism", "All CPU cores"],
];

// importance values come from the trained Random Forest (feature_importances_).
// nonCancerousMean / cancerousMean are optional: fill them in with the class
// averages from your training features. While they are null, those two
// columns stay hidden automatically.
const FEATURES = [
  { name: "Mean intensity", description: "Average grayscale intensity", role: "Relative brightness", importance: 0.321, nonCancerousMean: null, cancerousMean: null },
  { name: "Texture contrast", description: "GLCM contrast (distance 1, angle 0)", role: "Local texture variation", importance: 0.241, nonCancerousMean: null, cancerousMean: null },
  { name: "Perimeter", description: "Length of the region boundary", role: "Boundary extent", importance: 0.119, nonCancerousMean: null, cancerousMean: null },
  { name: "Area", description: "Pixels in the selected region", role: "Region size", importance: 0.118, nonCancerousMean: null, cancerousMean: null },
  { name: "Solidity", description: "Region area ÷ convex hull area", role: "Boundary irregularity", importance: 0.102, nonCancerousMean: null, cancerousMean: null },
  { name: "Eccentricity", description: "Elongation relative to an ellipse", role: "Shape", importance: 0.099, nonCancerousMean: null, cancerousMean: null },
];

// These match describe_shape() in inference.py exactly. If you change the
// thresholds there, change them here too.
const DESCRIPTION_RULES = [
  { measurement: "Eccentricity", ifCondition: "> 0.7", ifText: "Elongated or irregular", elseCondition: "≤ 0.7", elseText: "Round or regular" },
  { measurement: "Solidity", ifCondition: "< 0.85", ifText: "Jagged border", elseCondition: "≥ 0.85", elseText: "Smooth border" },
  { measurement: "Mean intensity", ifCondition: "< 100", ifText: "Darker staining", elseCondition: "≥ 100", elseText: "Lighter staining" },
  { measurement: "Texture contrast", ifCondition: "> 100", ifText: "Coarse or rough texture", elseCondition: "≤ 100", elseText: "Fine or uniform texture" },
];

const DISTRIBUTION = [
  { name: "Connective", n: 7850, color: "#a78bfa" },
  { name: "Inflammatory", n: 6795, color: "#fbbf24" },
  { name: "Neoplastic", n: 6391, color: "#fb7185" },
  { name: "Epithelial", n: 3879, color: "#34d399" },
  { name: "Dead", n: 423, color: "#94a3b8" },
];

function Heading({ number, eyebrow, title, description }) {
  return (
    <div className="section-heading">
      <div className="section-number">{number}</div>
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}

function SummaryCard({ card }) {
  return (
    <article className={`panel eval-card eval-${card.accent}`}>
      <div className="eval-card-top">
        <h3>{card.title}</h3>
        <span className="eval-tag">{card.tag}</span>
      </div>

      <div className="eval-accuracy">
        <span>Test accuracy</span>
        <strong>{card.accuracy}</strong>
      </div>

      <div className="eval-stats">
        {card.stats.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <div className="eval-meter-head">
        <span>Test macro-F1</span>
        <span>{card.macroF1.toFixed(3)}</span>
      </div>

      <div className="eval-meter">
        <div style={{ width: `${card.macroF1 * 100}%` }} />
      </div>

      <p className="eval-note">{card.note}</p>
    </article>
  );
}

function ConfusionMatrix({ title, classes, rows }) {
  return (
    <>
      <div className="eval-subhead">
        <span className="eyebrow">{title}</span>
        <span className="muted-label">ROWS = ACTUAL · COLUMNS = PREDICTED</span>
      </div>

      <div className="eval-table-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th>Actual \ Predicted</th>
              {classes.map((name) => <th key={name}>{name}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.actual}>
                <td>{row.actual}</td>
                {row.counts.map((count, index) => (
                  <td
                    key={classes[index]}
                    style={index === classes.indexOf(row.actual) ? { fontWeight: 800 } : undefined}
                  >
                    {count.toLocaleString()}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ConfusionStatCards({ stats, subtitle }) {
  const cards = [
    { key: "tp", label: "True Positive", short: "TP", value: stats.tp, description: "Correctly identified positive cases", tone: "tp" },
    { key: "fp", label: "False Positive", short: "FP", value: stats.fp, description: "Negative cases incorrectly flagged positive", tone: "fp" },
    { key: "tn", label: "True Negative", short: "TN", value: stats.tn, description: "Correctly identified negative cases", tone: "tn" },
    { key: "fn", label: "False Negative", short: "FN", value: stats.fn, description: "Positive cases incorrectly missed", tone: "fn" },
  ];

  return (
    <div className="eval-cm-stats">
      <div className="eval-cm-stats-heading">
        <span className="eyebrow">CONFUSION MATRIX BREAKDOWN</span>
        {subtitle && <span className="muted-label">{subtitle}</span>}
      </div>

      <div className="eval-cm-stat-grid">
        {cards.map((card) => (
          <article className={`eval-cm-stat eval-cm-${card.tone}`} key={card.key}>
            <span className="eval-cm-stat-label">
              <span className="eval-cm-stat-code">{card.short}</span>
              {card.label}
            </span>
            <strong className="eval-cm-stat-value">
              {card.value.toLocaleString()}
            </strong>
            <p>{card.description}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function getOneVsRestStats(rows, classes, selectedClass) {
  const classIndex = classes.indexOf(selectedClass);
  if (classIndex === -1) return { tp: 0, fp: 0, tn: 0, fn: 0 };

  const total = rows.reduce(
    (sum, row) => sum + row.counts.reduce((a, b) => a + b, 0),
    0
  );

  const tp = rows[classIndex].counts[classIndex];
  const actualPositive = rows[classIndex].counts.reduce((sum, n) => sum + n, 0);
  const predictedPositive = rows.reduce((sum, row) => sum + row.counts[classIndex], 0);
  const fn = actualPositive - tp;
  const fp = predictedPositive - tp;
  const tn = total - tp - fn - fp;

  return { tp, fp, tn, fn };
}

function MulticlassConfusionStats({ classes, rows }) {
  const [selectedClass, setSelectedClass] = useState(classes[0]);
  const stats = getOneVsRestStats(rows, classes, selectedClass);

  return (
    <div className="eval-multiclass-stats">
      <div className="eval-cm-stats-heading">
        <span className="eyebrow">PER-CLASS ERROR BREAKDOWN</span>
        <span className="muted-label">ONE-VS-REST</span>
      </div>

      <label className="eval-cm-select-label" htmlFor="eval-cm-class">
        Choose a cell class
      </label>

      <select
        id="eval-cm-class"
        className="eval-cm-select"
        value={selectedClass}
        onChange={(event) => setSelectedClass(event.target.value)}
      >
        {classes.map((name) => <option value={name} key={name}>{name}</option>)}
      </select>

      <p className="eval-cm-explanation">
        <strong>{selectedClass}</strong> is treated as positive. Every other
        class is treated as negative for this calculation.
      </p>

      <ConfusionStatCards
        stats={stats}
        subtitle={`Selected class: ${selectedClass}`}
      />
    </div>
  );
}

function ResultsTable({ rows }) {
  return (
    <div className="eval-table-wrap">
      <table className="eval-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-score</th>
            <th className="eval-bar-col">F1</th>
            <th>Test samples</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>
                <i className="eval-dot" style={{ background: r.color }} />
                {r.name}
              </td>
              <td>{r.precision.toFixed(2)}</td>
              <td>{r.recall.toFixed(2)}</td>
              <td>{r.f1.toFixed(2)}</td>
              <td className="eval-bar-col">
                <div className="eval-mini-track">
                  <div style={{ width: `${r.f1 * 100}%`, background: r.color }} />
                </div>
              </td>
              <td>{r.n.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureTable({ features }) {
  const hasMeans = features.some(
    (f) => f.nonCancerousMean !== null || f.cancerousMean !== null
  );
  const maxImportance = Math.max(...features.map((f) => f.importance));

  return (
    <div className="eval-table-wrap">
      <table className="eval-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Description</th>
            <th>Role</th>
            {hasMeans && <th>Non-cancerous (mean)</th>}
            {hasMeans && <th>Cancerous (mean)</th>}
            <th>RF importance</th>
            <th className="eval-bar-col">Share</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => (
            <tr key={f.name}>
              <td>{f.name}</td>
              <td className="eval-muted">{f.description}</td>
              <td className="eval-muted">{f.role}</td>
              {hasMeans && <td>{f.nonCancerousMean ?? "—"}</td>}
              {hasMeans && <td>{f.cancerousMean ?? "—"}</td>}
              <td>{f.importance.toFixed(3)}</td>
              <td className="eval-bar-col">
                <div className="eval-mini-track">
                  <div
                    style={{
                      width: `${(f.importance / maxImportance) * 100}%`,
                      background: "#67e8f9",
                    }}
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RulesTable({ rules }) {
  return (
    <>
      <div className="eval-subhead">
        <span className="eyebrow">RULE-BASED DESCRIPTIONS</span>
      </div>

      <div className="eval-table-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th>Measurement</th>
              <th>Condition</th>
              <th>Description</th>
              <th>Condition</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r) => (
              <tr key={r.measurement}>
                <td>{r.measurement}</td>
                <td><code>{r.ifCondition}</code></td>
                <td className="eval-muted">{r.ifText}</td>
                <td><code>{r.elseCondition}</code></td>
                <td className="eval-muted">{r.elseText}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Params({ items }) {
  return (
    <div className="eval-params">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

export default function ModelEvaluation() {
  const maxCount = Math.max(...DISTRIBUTION.map((d) => d.n));

  return (
    <main className="app-shell">
      <div className="background-grid" aria-hidden="true" />
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <header className="topbar">
        <a className="brand" href="#/" aria-label="PathoAI home">
          <span className="brand-mark">
            <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
              <path d="M20 5v30M5 20h30" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
              <circle cx="20" cy="20" r="12" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="20" cy="20" r="4" fill="currentColor" />
            </svg>
          </span>
          <span className="brand-copy">
            <strong>PATHO<span>AI</span></strong>
            <small>COMPUTATIONAL PATHOLOGY</small>
          </span>
        </a>

        <nav className="topnav">
          <a className="nav-link" href="#/">Analysis</a>
          <a className="nav-link nav-link-active" href="#/evaluation">Model evaluation</a>
        </nav>
      </header>

      <section className="eval-hero">
        <div className="eyebrow">MODEL VALIDATION</div>
        <h1>Model performance &amp; architecture</h1>
        <p>
          Evaluation results from the trained models on the held-out test
          dataset. These are model-level metrics, not predictions for an
          uploaded image.
        </p>
      </section>

      <section className="eval-summary-grid">
        {SUMMARY.map((card) => <SummaryCard key={card.key} card={card} />)}
      </section>

      <section className="panel eval-section">
        <Heading
          number="01"
          eyebrow="NETWORK CONFIGURATION"
          title="SmallCNN architecture"
          description="Both networks share the same feature-extraction layers. Only the final classification layer differs."
        />

        <div className="eval-arch">
          {ARCH_BLOCKS.map((b, i) => (
            <div className="eval-arch-item" key={b.label}>
              <div className="eval-arch-box">
                <span>{b.label}</span>
                <strong>{b.value}</strong>
                <small>{b.sub}</small>
              </div>
              {i < ARCH_BLOCKS.length - 1 && <span className="eval-arrow">→</span>}
            </div>
          ))}
        </div>

        <p className="eval-caption">
          Each convolutional block applies Conv2D, batch normalization, ReLU
          and max pooling. Softmax converts output scores into probabilities
          when probabilities are needed.
        </p>
        <Params items={HYPERPARAMS} />
      </section>

      <section className="panel eval-section">
        <Heading
          number="02"
          eyebrow="TRAINING SETUP"
          title="Training configuration"
          description="Settings used in the reported training run."
        />

        <div className="eval-table-wrap">
          <table className="eval-table">
            <thead>
              <tr><th>Parameter</th><th>Binary CNN</th><th>Cell-type CNN</th></tr>
            </thead>
            <tbody>
              {TRAINING_CONFIG.map(([p, a, b]) => (
                <tr key={p}><td className="eval-muted">{p}</td><td>{a}</td><td>{b}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel eval-section">
        <Heading
          number="03"
          eyebrow="BINARY CNN"
          title="Neoplastic vs. other classes"
          description="Test accuracy 91% · macro-F1 0.890. The positive class is the project-defined Neoplastic label, not an independent clinical diagnosis."
        />

        <ResultsTable rows={BINARY_RESULTS} />

        <ConfusionMatrix
          title="BINARY CNN · TEST CONFUSION MATRIX"
          classes={["Non-cancerous", "Cancerous"]}
          rows={BINARY_CNN_CONFUSION}
        />

        <ConfusionStatCards
          stats={{ tp: 1181, fp: 229, tn: 3483, fn: 219 }}
          subtitle="Positive class: Cancerous"
        />

        <p className="eval-caption">
          Of 3,712 actual non-cancerous test samples, 3,483 were correctly
          classified and 229 were predicted as cancerous. Of 1,400 actual
          cancerous samples, 1,181 were correctly classified and 219 were
          missed.
        </p>
      </section>

      <section className="panel eval-section">
        <Heading
          number="04"
          eyebrow="CELL-TYPE CNN"
          title="Five-class cell phenotyping"
          description="Test accuracy 76% · test macro-F1 0.635 · best validation macro-F1 0.717."
        />

        <ResultsTable rows={CELL_RESULTS} />

        <ConfusionMatrix
          title="CELL-TYPE CNN · TEST CONFUSION MATRIX"
          classes={CLASSES}
          rows={CELL_CNN_CONFUSION}
        />

        <MulticlassConfusionStats
          classes={CLASSES}
          rows={CELL_CNN_CONFUSION}
        />

        <p className="eval-caption">
          The largest confusion shown is 356 actual Inflammatory samples
          predicted as Connective. Dead recall is 13/21, but its small test
          sample count makes the estimate unstable.
        </p>

        <div className="eval-subhead">
          <span className="eyebrow">TRAINING DISTRIBUTION</span>
          <span className="muted-label">25,338 SAMPLES</span>
        </div>

        <div className="eval-dist">
          {DISTRIBUTION.map((d) => (
            <div className="eval-dist-row" key={d.name}>
              <span>{d.name}</span>
              <div className="eval-mini-track eval-dist-track">
                <div
                  style={{
                    width: `${(d.n / maxCount) * 100}%`,
                    background: d.color,
                  }}
                />
              </div>
              <strong>{d.n.toLocaleString()}</strong>
            </div>
          ))}
        </div>

        <p className="eval-caption">
          The training set is imbalanced: Dead has only 423 training samples.
          Weighted loss can help but does not guarantee equal performance.
        </p>
      </section>

      <section className="panel eval-section">
        <Heading
          number="05"
          eyebrow="CLASSICAL PIPELINE"
          title="Morphological features"
          description="Six features are extracted from an estimated nucleus region. RF importance shows how much the Random Forest relied on each one."
        />

        <FeatureTable features={FEATURES} />

        <p className="eval-caption">
          RF importance values sum to 1 and come from the trained forest. They
          show how much the model used each feature, not that a feature causes
          or indicates cancer.
        </p>

        <RulesTable rules={DESCRIPTION_RULES} />

        <p className="eval-caption">
          These thresholds describe the extracted measurements. They are not
          explanations of the CNN's internal decisions, and the region is an
          estimate rather than a verified nucleus segmentation.
        </p>
      </section>

      <section className="panel eval-section">
        <Heading
          number="06"
          eyebrow="CLASSICAL MODEL · RANDOM FOREST"
          title="Random Forest evaluation"
          description="A 200-tree forest predicts the project-defined binary label from morphological features."
        />

        <Params items={RF_PARAMS} />

        <ResultsTable rows={RF_RESULTS} />

        <ConfusionMatrix
          title="RANDOM FOREST · TEST CONFUSION MATRIX"
          classes={["Non-cancerous", "Cancerous"]}
          rows={RF_CONFUSION}
        />

        <ConfusionStatCards
          stats={{ tp: 991, fp: 896, tn: 2816, fn: 409 }}
          subtitle="Positive class: Cancerous"
        />

        <p className="eval-caption">
          Test accuracy is approximately 74%, with macro-F1 0.707. The forest
          identifies 991 cancerous cases but also produces 896 false positives.
          Its output should not be treated as a clinical diagnosis.
        </p>
      </section>

      <section className="panel eval-section">
        <Heading
          number="07"
          eyebrow="OVERALL INTERPRETATION"
          title="Findings and limitations"
          description="Interpret model metrics together with class-level errors and the limits of the test dataset."
        />

        <div className="eval-table-wrap">
          <table className="eval-table">
            <thead><tr><th>Area</th><th>Interpretation</th></tr></thead>
            <tbody>
              <tr>
                <td>Binary CNN</td>
                <td>91% accuracy and 0.890 macro-F1 on the reported test set.</td>
              </tr>
              <tr>
                <td>Cell-type CNN</td>
                <td>76% accuracy and 0.635 macro-F1, with weaker results for Dead and Inflammatory.</td>
              </tr>
              <tr>
                <td>Class imbalance</td>
                <td>Dead is underrepresented, which makes its evaluation metrics less stable.</td>
              </tr>
              <tr>
                <td>Model comparison</td>
                <td>The binary CNN has a higher reported macro-F1 than the Random Forest baseline.</td>
              </tr>
              <tr>
                <td>Evaluation scope</td>
                <td>These results do not establish generalization to other laboratories, staining protocols, or patient-level diagnoses.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <div className="reference-note eval-disclaimer">
        <span>ⓘ</span>
        <p>
          Research prototype. Metrics come from the reported held-out test
          evaluation and do not establish clinical diagnostic reliability.
          Model confidence is not a calibrated probability of correctness.
        </p>
      </div>

      <footer className="app-footer">
        <a className="footer-brand" href="#/">PATHO<span>AI</span></a>
        <span>MODEL EVALUATION · RESEARCH PROTOTYPE</span>
        <a href="#/" className="back-to-top">BACK TO ANALYSIS ←</a>
      </footer>
    </main>
  );
}