# Module 8: Evaluation & Analysis

## Limitations

**Dataset substitution.** The originally proposed CRCHistoPhenotypes dataset was replaced with
PanNuke after Warwick's TIA Centre confirmed access would be delayed past this project's timeline
(correspondence on record). PanNuke is larger and includes real segmentation masks, but this
substitution is a documented deviation from the original proposal, not an equivalent swap.

**Image-level, not patient-level, split.** PanNuke provides no patient identifiers — only source
images. The train/validation/test split groups by image, guaranteeing no image's nuclei appear in
more than one split, but this is a weaker generalization guarantee than a true patient-level split
would be.

**Severe class imbalance on "Dead."** The Dead category has only 423 training examples against
7,850 for Connective — roughly 18x fewer. Every model trained on cell-type classification
(classical and CNN) struggles specifically on this class; this is a direct, expected consequence of
data scarcity, not a bug, and is reflected honestly in every cell-type confusion matrix and
classification report produced.

**Independent single-task models can disagree.** The isCancerous CNN, the cell-type CNN, and the
classical isCancerous model were trained completely separately. Nothing constrains them to agree
with each other, and in practice they sometimes don't — e.g. the isCancerous model predicting
"non-cancerous" while the cell-type model's top pick is "Neoplastic" on the same image. Rather than
suppress or paper over this, the web interface surfaces disagreement explicitly. This is treated as
a genuine finding about the limits of using separate single-task models, not an error to hide.

**Nucleus-mask estimation for new images.** Training features are computed from PanNuke's exact,
hand-drawn nucleus masks. A new uploaded photo has no such mask, so the system estimates one using
Otsu thresholding — a real approximation that can fail in specific, identifiable ways (for example,
mistakenly selecting a block of image padding instead of the actual nucleus when a crop sits near a
tile's edge). This was found, reproduced, and documented during testing rather than only described
in the abstract. The CNNs mitigate this specific issue for prediction (they read pixels directly,
no mask needed), but the classical models and the feature-based explanations still depend on it.

## Real-World Adequacy

This system is a research prototype, evaluated on a public research dataset, and is **not**
clinically validated or deployable as a diagnostic tool. Its current performance (CNN isCancerous:
macro-F1 ≈ 0.88 on held-out PanNuke data; cell-type CNN: macro-F1 ≈ 0.61, weighed down by the Dead
class) represents a reasonable research-stage result, not a clinical-grade one. Real deployment
would require: validation on external, multi-institution data; clinician-in-the-loop review rather
than autonomous decision-making; formal regulatory evaluation; and resolution of the Dead-class
performance gap, among other requirements well beyond this project's scope.

## Ethical and Professional Considerations

**Patient privacy.** PanNuke is a public, de-identified research dataset; this project introduces
no new patient data and performs no re-identification. Any future use of patient-derived data in a
similar system would require informed consent and institutional ethical approval, which this
project does not claim to have obtained or need, since no real patient data is used.

**Risk asymmetry in false negatives vs. false positives.** In a cancer-screening context, a missed
cancerous cell (false negative) is generally more harmful than a false alarm (false positive),
since a false alarm is typically caught at a later review stage while a false negative may not be.
This project reports precision and recall separately for the cancerous class specifically (rather
than only an aggregate accuracy figure) so that this asymmetry is visible, not hidden behind a
single summary number.

**Risk of overclaiming.** Throughout this project, results are reported with their actual
confidence values, and known weaknesses (the Dead class, the Otsu-estimation limitation, model
disagreement) are stated plainly rather than omitted. This is treated as a professional obligation,
not an optional nicety — presenting a prototype's limitations honestly is part of responsible AI
development in a healthcare-adjacent context.

**Scope of training data and generalization.** The model is trained exclusively on PanNuke's
Colon-tissue subset. It has not been evaluated on other tissue types, other staining protocols, other
scanners, or populations outside PanNuke's source data, and no claim is made about its performance
outside that scope.