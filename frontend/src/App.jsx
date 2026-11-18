import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/predict/";

// NOTE for developers: confirm these names and their order match the
// classes in your trained label encoder (celltype_label_encoder.joblib).
const CELL_TYPES = [
  {
    name: "Neoplastic",
    short: "NEO",
    color: "#fb7185",
    description: "Abnormal cell growth",
    plain:
      "Cells that are growing in an abnormal way. A doctor needs to check whether they are harmless or harmful.",
    meaning:
      "Tumour cells that grow abnormally and out of control. In colorectal tissue they usually arise from the gland-lining epithelium.",
    appearance:
      "Nuclei are often enlarged, dark and irregular in shape, with crowded or disorganised arrangement.",
    caution:
      "Neoplastic describes abnormal growth and does not by itself say whether it is benign or malignant. It can be confused with normal epithelial cells because both come from the same tissue.",
  },
  {
    name: "Inflammatory",
    short: "INF",
    color: "#fbbf24",
    description: "Immune response cells",
    plain:
      "Cells from your body's defence system. They gather when tissue is irritated or fighting something. Finding them is common and does not mean cancer.",
    meaning:
      "Immune cells such as lymphocytes, plasma cells and neutrophils that gather where tissue is irritated, infected or reacting to a tumour.",
    appearance:
      "Typically small, round, densely stained nuclei that often sit in clusters.",
    caution:
      "Inflammation is common in healthy and diseased tissue alike, so its presence does not indicate cancer. Small dark nuclei of other types can look similar.",
  },
  {
    name: "Connective",
    short: "CON",
    color: "#a78bfa",
    description: "Connective and stromal cells",
    plain:
      "Supporting cells that hold tissue together, a bit like scaffolding. They are a normal part of healthy tissue.",
    meaning:
      "Supporting cells of the stroma, such as fibroblasts, smooth muscle and blood-vessel lining cells, which form the framework around glands.",
    appearance:
      "Often elongated or spindle-shaped nuclei lying in bundles or thin lines.",
    caution:
      "This is a broad group of different cell kinds. Tangentially cut tissue or stretched cells can resemble other categories, especially inflammatory cells.",
  },
  {
    name: "Dead",
    short: "DEA",
    color: "#94a3b8",
    description: "Dead-cell regions",
    plain:
      "Cells that have died. This can happen in damaged tissue or in fast-growing growths. A doctor decides whether it matters.",
    meaning:
      "Cells that have died through necrosis or apoptosis. Dead cells are often found inside fast-growing tumours or damaged tissue.",
    appearance:
      "Shrunken, very dark or fragmented nuclei, or pale, washed-out areas where structure has been lost.",
    caution:
      "This is the rarest category in the training data, so it is the least reliable one. Treat a Dead prediction as a weak hint and check it by eye.",
  },
  {
    name: "Epithelial",
    short: "EPI",
    color: "#34d399",
    description: "Epithelial tissue cells",
    plain:
      "Cells that line the inside of the bowel and form its glands. They are normal, but many bowel cancers start in these cells, so doctors look at them closely.",
    meaning:
      "Cells that line surfaces and form glands. In the colon they make up the gland structures that absorb water and secrete mucus.",
    appearance:
      "Usually oval nuclei of fairly uniform size, lined up along the edge of glands.",
    caution:
      "Healthy epithelium and neoplastic epithelium are hard to tell apart. Do not assume an Epithelial label means the tissue is normal.",
  },
];

const READING_GUIDE = [
  "The model gives one cell-type label for the whole image tile, while real tissue contains a mix of cell types.",
  "Confidence is the model's own score. It is not a guarantee that the prediction is correct.",
  "Use these outputs as a research aid and have a qualified pathologist review any result that matters.",
];

// Concern levels are a guide to how much attention a finding usually needs.
// They are NOT a health status or a diagnosis.
const CONCERN_LEVELS = {
  low: { label: "Usually normal", color: "#34d399" },
  mid: { label: "Needs context", color: "#fbbf24" },
  high: { label: "Higher concern", color: "#fb7185" },
};

const CELL_LEVEL = {
  Connective: "low",
  Epithelial: "low",
  Inflammatory: "mid",
  Dead: "mid",
  Neoplastic: "high",
};

const OVERALL_LEVELS = {
  low: {
    label: "Lower concern",
    color: "#34d399",
    text: "No patterns linked to cancer were seen. This is a good sign, but it is not the same as being healthy.",
  },
  mid: {
    label: "Needs a specialist's review",
    color: "#fbbf24",
    text: "The result is unclear, or the two methods disagreed. A specialist should look at the sample.",
  },
  high: {
    label: "Higher concern",
    color: "#fb7185",
    text: "Patterns linked to cancer were seen. This is not a diagnosis, but it is a good reason to see a specialist soon.",
  },
};

const FEATURE_LABELS = {
  area: "Nuclear area",
  eccentricity: "Eccentricity",
  solidity: "Solidity",
  perimeter: "Perimeter",
  mean_intensity: "Mean intensity",
  texture_contrast: "Texture contrast",
};

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "—";

  const number = Number(value);
  if (!Number.isFinite(number)) return "—";

  const percentage = number <= 1 ? number * 100 : number;
  return `${percentage.toFixed(1)}%`;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  return String(value);
}

function confidenceInWords(value) {
  const number = Number(value);
  if (value === null || value === undefined || value === "" || !Number.isFinite(number)) {
    return null;
  }

  const percentage = number <= 1 ? number * 100 : number;
  if (percentage >= 90) return "very sure";
  if (percentage >= 70) return "fairly sure";
  return "not very sure";
}

function normalizeProbabilities(probabilities) {
  if (!probabilities) return [];

  if (Array.isArray(probabilities)) {
    return probabilities
      .map((item) => {
        if (typeof item === "number") {
          return { name: "Class", value: item };
        }

        return {
          name: item?.label ?? item?.name ?? item?.class ?? "Unknown",
          value: Number(item?.probability ?? item?.confidence ?? item?.value ?? 0),
        };
      })
      .filter((item) => Number.isFinite(item.value));
  }

  if (typeof probabilities === "object") {
    return Object.entries(probabilities).map(([name, value]) => ({
      name,
      value: Number(
        typeof value === "object"
          ? value?.probability ?? value?.confidence ?? value?.value ?? 0
          : value
      ),
    })).filter((item) => Number.isFinite(item.value));
  }

  return [];
}

function getCellTypeInfo(name) {
  const normalized = String(name ?? "").toLowerCase().replace(/[_-]/g, " ");

  return CELL_TYPES.find(
    (type) => type.name.toLowerCase() === normalized
  ) ?? {
    name: name || "Unknown",
    short: "CELL",
    color: "#67e8f9",
    description: "Predicted cell category",
    plain: "",
    meaning: "",
    appearance: "",
    caution: "",
  };
}

function LevelTag({ level }) {
  const info = CONCERN_LEVELS[level];
  if (!info) return null;

  return (
    <span className="level-tag" style={{ "--level-color": info.color }}>
      <i />
      {info.label}
    </span>
  );
}

function StatusBadge({ positive, children }) {
  return (
    <span className={`status-badge ${positive ? "status-positive" : "status-negative"}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

function SectionHeading({ number, eyebrow, title, description }) {
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

function ProbabilityBar({ name, value, color }) {
  const percentage = Math.max(0, Math.min(100, value <= 1 ? value * 100 : value));

  return (
    <div className="probability-row">
      <div className="probability-label">
        <span>{name}</span>
        <strong>{percentage.toFixed(1)}%</strong>
      </div>
      <div className="probability-track">
        <div
          className="probability-fill"
          style={{
            width: `${percentage}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(selectedFile) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setError("");
  }

  function handleFileInput(event) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  function resetAnalysis() {
    setFile(null);
    setResult(null);
    setError("");
    setLoading(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function analyzeImage() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.error ||
          data.detail ||
          data.message ||
          `Analysis failed with status ${response.status}.`
        );
      }

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message?.includes("Failed to fetch")
          ? "Could not connect to Django. Make sure the backend is running and CORS is configured if needed."
          : err.message || "Something went wrong while analyzing the image."
      );
    } finally {
      setLoading(false);
    }
  }

  const cellType = result?.cell_type;
  const cellInfo = getCellTypeInfo(cellType?.predicted);
  const probabilities = normalizeProbabilities(cellType?.probabilities)
    .sort((a, b) => b.value - a.value);

  const features = result?.features && typeof result.features === "object"
    ? Object.entries(result.features)
    : [];

  const classical = result?.classical_model;
  const isCancerous = result?.is_cancerous === true;
  const isNonCancerous = result?.is_cancerous === false;
  const binaryLabel = isCancerous
    ? "Cancer detected"
    : isNonCancerous
      ? "Non-cancerous"
      : "Result unavailable";

  // Random Forest result. It is only "available" if the backend sent a real
  // true/false prediction (it will not if feature extraction failed).
  const rfAvailable = typeof classical?.is_cancerous === "boolean";
  const cnnAvailable = typeof result?.is_cancerous === "boolean";

  // Agreement: work it out here by comparing the two predictions, and only
  // fall back to the backend's flag if we could not compare them ourselves.
  let agreement;
  if (rfAvailable && cnnAvailable) {
    agreement = classical.is_cancerous === result.is_cancerous;
  } else if (typeof classical?.agrees_with_cnn === "boolean") {
    agreement = classical.agrees_with_cnn;
  }

  const modelName = classical?.model_name || "Random Forest";
  const cancerWords = confidenceInWords(result?.confidence);
  const cellWords = confidenceInWords(cellType?.confidence);
  const overallLevel = isCancerous
    ? "high"
    : isNonCancerous && agreement !== false
      ? "low"
      : "mid";
  const overall = OVERALL_LEVELS[overallLevel];

  return (
    <main className="app-shell">
      <div className="background-grid" aria-hidden="true" />
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <header className="topbar">
        <a className="brand" href="#top" aria-label="PathoAI home">
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
          <a className="nav-link nav-link-active" href="#/">Analysis</a>
          <a className="nav-link" href="#/evaluation">Model evaluation</a>
        </nav>

        <div className="topbar-status">
          <span className="live-indicator" />
          <span>LOCAL INFERENCE SYSTEM</span>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="hero-kicker">
            <span className="kicker-line" />
            AI-ASSISTED HISTOPATHOLOGY
          </div>

          <h1>
            See beyond
            <br />
            <span>the tissue.</span>
          </h1>

          <p className="hero-description">
            Analyze colorectal tissue imagery with deep learning.
            Classify cancer status, explore cell-type phenotypes, and
            inspect extracted morphological features in one workspace.
          </p>

          <div className="hero-tags">
            <span><i /> CNN CLASSIFICATION</span>
            <span><i /> RANDOM FOREST</span>
            <span><i /> CELL PHENOTYPING</span>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="visual-orbit orbit-one" />
          <div className="visual-orbit orbit-two" />
          <div className="visual-core">
            <div className="core-crosshair crosshair-horizontal" />
            <div className="core-crosshair crosshair-vertical" />
            <div className="core-ring ring-one" />
            <div className="core-ring ring-two" />
            <div className="core-cell cell-one" />
            <div className="core-cell cell-two" />
            <div className="core-cell cell-three" />
            <div className="core-cell cell-four" />
            <div className="core-cell cell-five" />
            <div className="core-cell cell-six" />
            <div className="core-center" />
          </div>
          <span className="visual-label label-top">TISSUE ANALYSIS</span>
          <span className="visual-label label-bottom">VISION MODEL / ACTIVE</span>
          <span className="visual-coordinates">23.04° N<br />72.57° E</span>
        </div>
      </section>

      <div className="workflow-line">
        <div className="workflow-step active"><span>01</span> IMAGE INPUT</div>
        <span className="workflow-connector" />
        <div className={`workflow-step ${result ? "active" : ""}`}><span>02</span> CANCER CLASSIFICATION</div>
        <span className="workflow-connector" />
        <div className={`workflow-step ${cellType ? "active" : ""}`}><span>03</span> CELL PHENOTYPING</div>
      </div>

      <section className="panel upload-panel">
        <SectionHeading
          number="01"
          eyebrow="INPUT / SPECIMEN"
          title="Image acquisition"
          description="Upload a histopathology image to begin the analysis pipeline."
        />

        <div className="upload-layout">
          <div
            className={`dropzone ${dragging ? "dropzone-dragging" : ""} ${previewUrl ? "dropzone-has-image" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) {
                setDragging(false);
              }
            }}
            onDrop={handleDrop}
          >
            {previewUrl ? (
              <div className="preview-stage">
                <img src={previewUrl} alt="Selected histopathology specimen preview" />
                <div className="preview-overlay">
                  <span className="preview-chip"><span /> SPECIMEN LOADED</span>
                  <button className="icon-button" type="button" onClick={resetAnalysis} aria-label="Remove image">
                    ×
                  </button>
                </div>
                <div className="preview-corner corner-tl" />
                <div className="preview-corner corner-tr" />
                <div className="preview-corner corner-bl" />
                <div className="preview-corner corner-br" />
              </div>
            ) : (
              <div className="dropzone-empty">
                <div className="upload-symbol">
                  <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
                    <path d="M24 31V9m0 0-8 8m8-8 8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M12 27v10a3 3 0 0 0 3 3h18a3 3 0 0 0 3-3V27" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </div>
                <h3>Drop your specimen here</h3>
                <p>Drag an image into this area, or browse your files.</p>
                <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}>
                  Browse images <span>↗</span>
                </button>
                <span className="upload-formats">PNG · JPG · JPEG · TIFF*</span>
              </div>
            )}
          </div>

          <aside className="specimen-info">
            <div className="info-card-label">SPECIMEN DETAILS</div>
            <div className="specimen-file-icon">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <rect x="4" y="3" width="16" height="18" rx="2" stroke="currentColor" strokeWidth="1.5" />
                <circle cx="9" cy="9" r="2" stroke="currentColor" strokeWidth="1.5" />
                <path d="m5 17 5-4 3 2 3-3 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <span className="detail-label">FILE NAME</span>
            <strong className="file-name">{file?.name || "No specimen selected"}</strong>
            <div className="detail-divider" />
            <div className="detail-pair">
              <span>File size</span>
              <strong>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "—"}</strong>
            </div>
            <div className="detail-pair">
              <span>File type</span>
              <strong>{file?.type?.split("/")[1]?.toUpperCase() || "—"}</strong>
            </div>
            <div className="detail-pair">
              <span>Analysis status</span>
              <strong className={result ? "text-success" : loading ? "text-cyan" : ""}>
                {loading ? "Processing…" : result ? "Complete" : file ? "Ready" : "Awaiting input"}
              </strong>
            </div>

            <button
              className="primary-button"
              type="button"
              onClick={analyzeImage}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="button-spinner" />
                  Analyzing specimen
                </>
              ) : (
                <>
                  Run analysis <span className="button-arrow">↗</span>
                </>
              )}
            </button>

            <p className="privacy-note">
              <span className="lock-symbol">⌑</span>
              Image sent to your local Django inference API.
            </p>
          </aside>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={handleFileInput}
        />

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon">!</span>
            <div>
              <strong>Analysis could not be completed</strong>
              <p>{error}</p>
            </div>
            <button type="button" className="error-dismiss" onClick={() => setError("")} aria-label="Dismiss error">×</button>
          </div>
        )}

        {loading && (
          <div className="processing-banner" role="status">
            <div className="processing-animation"><span /><span /><span /></div>
            <div>
              <strong>Processing tissue image</strong>
              <p>Running classification models and extracting morphological features…</p>
            </div>
          </div>
        )}
      </section>

      {result && (
        <>
          <div className="results-heading">
            <div>
              <div className="eyebrow">ANALYSIS REPORT / 01</div>
              <h2>Specimen intelligence</h2>
              <p>Model outputs generated from the selected tissue image.</p>
            </div>
            <button className="text-button" type="button" onClick={resetAnalysis}>
              ↻ New analysis
            </button>
          </div>

          <section className="results-grid">
            <article className="panel cancer-panel">
              <SectionHeading
                number="02"
                eyebrow="BINARY CLASSIFICATION"
                title="Cancer classification"
                description="CNN-based prediction of cancer status."
              />

              <div className={`diagnosis-card ${isCancerous ? "diagnosis-cancer" : "diagnosis-clear"}`}>
                <div className="diagnosis-topline">
                  <span className="diagnosis-icon">
                    {isCancerous ? "!" : isNonCancerous ? "✓" : "?"}
                  </span>
                  <span className="diagnosis-label">CNN MODEL OUTPUT</span>
                  <span className="diagnosis-indicator" />
                </div>
                <h3>{binaryLabel}</h3>
                <p>{result.explanation || "The model returned a classification result for the uploaded specimen."}</p>
                <div className="confidence-block">
                  <div className="confidence-heading">
                    <span>Prediction confidence</span>
                    <strong>{formatPercent(result.confidence)}</strong>
                  </div>
                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{ width: `${Math.min(100, Math.max(0, Number(result.confidence) <= 1 ? Number(result.confidence) * 100 : Number(result.confidence) || 0))}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="model-comparison">
                <div className="comparison-heading">
                  <span className="eyebrow">CROSS-MODEL VALIDATION</span>
                  {typeof agreement === "boolean" ? (
                    <StatusBadge positive={agreement}>
                      {agreement ? "Models agree" : "Models differ"}
                    </StatusBadge>
                  ) : (
                    <StatusBadge positive={false}>Second model unavailable</StatusBadge>
                  )}
                </div>

                <div className="model-row">
                  <div className="model-avatar cnn-avatar">C</div>
                  <div className="model-row-copy">
                    <strong>Convolutional Neural Network</strong>
                    <span>Deep visual feature learning</span>
                  </div>
                  <div className="model-result">
                    <strong>{isCancerous ? "Cancerous" : isNonCancerous ? "Non-cancerous" : "—"}</strong>
                    <span>{formatPercent(result.confidence)}</span>
                  </div>
                </div>

                <div className="model-row">
                  <div className="model-avatar rf-avatar">RF</div>
                  <div className="model-row-copy">
                    <strong>{modelName}</strong>
                    <span>Handcrafted morphology features</span>
                  </div>
                  <div className="model-result">
                    {rfAvailable ? (
                      <>
                        <strong>{classical.is_cancerous ? "Cancerous" : "Non-cancerous"}</strong>
                        <span>{formatPercent(classical.confidence)}</span>
                      </>
                    ) : (
                      <>
                        <strong>Unavailable</strong>
                        <span>No features measured</span>
                      </>
                    )}
                  </div>
                </div>

                <p className="model-note">
                  {typeof agreement === "boolean"
                    ? agreement
                      ? `The ${modelName} model agrees with the CNN. Both models predict this image is ${isCancerous ? "cancerous" : "non-cancerous"}.`
                      : `The ${modelName} model does not agree with the CNN. The CNN predicts ${isCancerous ? "cancerous" : "non-cancerous"}, while the ${modelName} predicts ${classical?.is_cancerous ? "cancerous" : "non-cancerous"}.`
                    : "The Random Forest could not run for this image because no morphology features were measured, so there is nothing to compare with the CNN."}
                </p>

                {classical?.explanation && (
                  <p className="model-note">{classical.explanation}</p>
                )}
              </div>
              <p className="disclaimer">
                Research prototype only. This output is not a clinical diagnosis.
              </p>
            </article>

            <article className="panel phenotype-panel">
              <SectionHeading
                number="03"
                eyebrow="MULTI-CLASS CLASSIFICATION"
                title="Cell-type phenotyping"
                description="CNN prediction across the configured cell categories."
              />

              <div className="phenotype-featured">
                <div className="cell-orb" style={{ "--cell-color": cellInfo.color }}>
                  <span>{cellInfo.short}</span>
                  <i className="orb-ring orb-ring-a" />
                  <i className="orb-ring orb-ring-b" />
                </div>
                <div className="phenotype-prediction">
                  <span className="detail-label">PREDICTED CELL TYPE</span>
                  <h3>{cellInfo.name}</h3>
                  <p>{cellInfo.description}</p>
                  <LevelTag level={CELL_LEVEL[cellInfo.name]} />
                  <div className="phenotype-confidence">
                    <span>Confidence</span>
                    <strong>{formatPercent(cellType?.confidence)}</strong>
                  </div>
                </div>
              </div>

              {cellInfo.caution && (
                <div className="caution-box">
                  <span className="caution-mark">!</span>
                  <p>
                    <strong>Caution:</strong> {cellInfo.caution}
                  </p>
                </div>
              )}

              {cellType?.explanation && (
                <div className="explanation-box">
                  <span className="explanation-mark">↳</span>
                  <p>{cellType.explanation}</p>
                </div>
              )}

              <div className="probabilities-heading">
                <span className="eyebrow">CLASS PROBABILITIES</span>
                <span className="muted-label">{probabilities.length} classes returned</span>
              </div>

              <div className="probability-list">
                {probabilities.length > 0 ? probabilities.map((item) => {
                  const info = getCellTypeInfo(item.name);
                  return (
                    <ProbabilityBar
                      key={item.name}
                      name={info.name}
                      value={item.value}
                      color={info.color}
                    />
                  );
                }) : (
                  <p className="empty-message">No class probability values were returned by the API.</p>
                )}
              </div>
            </article>
          </section>

          <section className="panel features-panel">
            <div className="features-header">
              <SectionHeading
                number="04"
                eyebrow="MORPHOLOGICAL MEASUREMENTS"
                title="Extracted features"
                description="Feature values returned by the classical analysis pipeline."
              />
              <span className="feature-count">{features.length.toString().padStart(2, "0")} FEATURES</span>
            </div>

            {features.length > 0 ? (
              <div className="feature-grid">
                {features.map(([key, value], index) => (
                  <div className="feature-card" key={key}>
                    <span className="feature-index">F{String(index + 1).padStart(2, "0")}</span>
                    <span className="feature-name">{FEATURE_LABELS[key] || key.replace(/_/g, " ")}</span>
                    <strong>{formatValue(value)}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-message">No morphological feature values were returned.</p>
            )}
          </section>

          <section className="panel plain-panel">
            <SectionHeading
              number="05"
              eyebrow="IN SIMPLE WORDS"
              title="What this result means"
              description="A plain-language summary of the results above. It is general information, not medical advice."
            />

            <div className="plain-notice">
              <span className="caution-mark">!</span>
              <p>
                <strong>Please remember:</strong> this tool cannot tell you that you are healthy,
                and it cannot tell you that you have cancer. Only a doctor who examines the whole
                tissue sample can do that.
              </p>
            </div>

            <div className="level-banner" style={{ "--level-color": overall.color }}>
              <span className="detail-label">OVERALL CONCERN LEVEL</span>
              <strong>{overall.label}</strong>
              <p>{overall.text}</p>
            </div>

            <div className="plain-grid">
              <div className="plain-card">
                <span className="detail-label">THE IMAGE OVERALL</span>
                <h3>
                  {isCancerous
                    ? "Patterns linked to cancer were seen"
                    : isNonCancerous
                      ? "No patterns linked to cancer were seen"
                      : "No clear result"}
                </h3>
                <p>
                  {isCancerous
                    ? "The tool found shapes and textures in this image that it connects with cancer. This is a reason to have the sample looked at by a specialist, not a final answer."
                    : isNonCancerous
                      ? "The tool did not find shapes and textures that it connects with cancer in this image. This does not rule cancer out, because the tool can miss things and only looked at this one image."
                      : "The tool could not give a clear answer for this image."}
                  {cancerWords && ` The tool is ${cancerWords} about this, but being sure is not the same as being right.`}
                </p>
              </div>

              <div className="plain-card">
                <span className="detail-label">THE TYPE OF CELLS</span>
                <h3>{cellType?.predicted ? `Looks like ${cellInfo.name.toLowerCase()} cells` : "No cell type returned"}</h3>
                <p>
                  {cellInfo.plain || "No simple description is available for this cell type."}
                  {cellWords && ` The tool is ${cellWords} about this label.`}
                </p>
              </div>

              <div className="plain-card">
                <span className="detail-label">A SECOND CHECK</span>
                <h3>
                  {typeof agreement === "boolean"
                    ? agreement
                      ? "Both methods gave the same answer"
                      : "The two methods did not agree"
                    : "No second check available"}
                </h3>
                <p>
                  {typeof agreement === "boolean"
                    ? agreement
                      ? "A second, different method reached the same result. That adds a little more support, but it is still not proof."
                      : "A second, different method gave another answer. This is a good reason to have the sample reviewed carefully by a specialist."
                    : "The second method could not run for this image, so only the main result is shown."}
                </p>
              </div>

              <div className="plain-card plain-card-next">
                <span className="detail-label">WHAT TO DO NEXT</span>
                <h3>Talk to your doctor</h3>
                <p>
                  Share this report with your doctor or pathologist and ask them to explain it.
                  Please do not make health decisions based on this page alone. If you feel
                  unwell or are worried, contact a doctor straight away.
                </p>
              </div>
            </div>
          </section>

          <div className="report-footer">
            <span>PATHOAI / ANALYSIS COMPLETE</span>
            <span>AI-ASSISTED RESEARCH TOOL · NOT FOR CLINICAL USE</span>
          </div>
        </>
      )}

      <section className="reference-section" id="cell-guide">
        <div className="reference-header">
          <div>
            <div className="eyebrow">CELL-TYPE GUIDE</div>
            <h2>Understanding the cell types</h2>
          </div>
          <p>What each category means, what it looks like, and the cautions that apply.</p>
        </div>

        <div className="cell-reference-grid">
          {CELL_TYPES.map((type, index) => (
            <div className="cell-reference-card" key={type.name}>
              <div className="reference-card-top">
                <span className="reference-index">0{index + 1}</span>
                <span className="reference-dot" style={{ background: type.color, boxShadow: `0 0 12px ${type.color}66` }} />
              </div>
              <div className="reference-cell-icon" style={{ "--cell-color": type.color }}>
                <span>{type.short}</span>
              </div>
              <h3>{type.name}</h3>
              <p className="reference-summary">{type.description}</p>
              <LevelTag level={CELL_LEVEL[type.name]} />

              <dl className="reference-detail">
                <div className="reference-plain">
                  <dt>In simple words</dt>
                  <dd>{type.plain}</dd>
                </div>
                <div>
                  <dt>What it means</dt>
                  <dd>{type.meaning}</dd>
                </div>
                <div>
                  <dt>What it looks like</dt>
                  <dd>{type.appearance}</dd>
                </div>
                <div className="reference-caution">
                  <dt>Caution</dt>
                  <dd>{type.caution}</dd>
                </div>
              </dl>
            </div>
          ))}
        </div>

        <div className="reference-guidance">
          <div className="eyebrow">NOTES ON READING THE RESULTS</div>
          <ul>
            {READING_GUIDE.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <footer className="app-footer">
        <a className="footer-brand" href="#top">PATHO<span>AI</span></a>
        <span>COLORECTAL HISTOPATHOLOGY · RESEARCH PROTOTYPE</span>
        <a href="#top" className="back-to-top">BACK TO TOP ↑</a>
      </footer>
    </main>
  );
}

export default App;