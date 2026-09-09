# AccentSense Experimental Protocol & Benchmarks (`EXPERIMENTS.md`)

This document records the experimental methodology, baseline evaluations, ablation studies, and evaluation metrics for **AccentSense**.

---

## 📈 Evaluation Metrics

Because speech datasets possess intrinsic class imbalance, we do **not** rely solely on raw accuracy.

1. **Balanced Accuracy**:
   $$\text{Balanced Acc} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
2. **Macro-Averaged F1 Score** (Primary Metric):
   $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c$$
3. **Expected Calibration Error (ECE)**:
   Measures whether model confidence matches true empirical accuracy:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
4. **Explanation Faithfulness (Area Under Deletion Curve - AUDC)**:
   Progressively masks the top $k\%$ most salient frames ($20\text{ ms}$) and measures how rapidly target class probability drops compared to random frame masking.
5. **Relative Word Error Rate Reduction (WERR)**:
   $$\text{WERR} = \frac{\text{WER}_{\text{baseline}} - \text{WER}_{\text{adapted}}}{\text{WER}_{\text{baseline}}} \times 100\%$$

---

## 🔬 Benchmark Comparison Matrix

| Model Tier | Backbone | Features | Pooling | Trainable Params | Test Macro-F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1 Baseline** | Classical SVM (RBF) | 39-dim MFCCs + $F_0$ + Energy | Mean/Std Summary | $\sim 1\text{k}$ | **0.2146** | ✅ Verified |
| **Phase 1 Baseline** | Random Forest (200 trees) | 39-dim MFCCs + $F_0$ + Energy | Mean/Std Summary | $\sim 10\text{k}$ | **0.3207** | ✅ Verified |
| **Phase 2 Deep** | Wav2Vec 2.0 Base | 768-dim SSL (Frozen) | Mean Pooling | $\sim 3\text{k}$ | *Planned* | 🟡 In Progress |
| **Phase 3 Deep (SOTA)**| **WavLM Base+** | **768-dim SSL (Frozen)** | **Attentive Stats (ASP)** | **$\sim 495\text{k}$** | *Target: >0.70* | 🟡 Initialized |
| **Phase 3 Ablation** | IndicWav2Vec | 768-dim Indian SSL (Frozen) | Attentive Stats (ASP) | $\sim 495\text{k}$ | *Target: >0.68* | ⚪ Future Ablation |

---

## 🔍 Explanation Faithfulness (AUDC) Benchmarks

*Evaluated on progressive deletion ($0\%$ to $50\%$ in $10\%$ increments) across the 4 Regional Anchors:*

| Attribution Method | Mean $\text{AUDC}_{\text{salient}}$ | Mean $\text{AUDC}_{\text{random}}$ | Mean $\Delta\text{AUDC}$ | Faithfulness Pass Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Integrated Gradients (Captum)** | **0.1305** | **0.1352** | **+0.0046** | **100.0% (4/4)** | ✅ Verified |
| Vanilla Input Gradient | 0.1289 | 0.1294 | +0.0005 | 75.0% (3/4) | ✅ Verified |

*Criterion*: A faithful explanation degrades model confidence faster under salient frame masking than under random frame masking ($\text{AUDC}_{\text{salient}} < \text{AUDC}_{\text{random}}$, i.e., $\Delta\text{AUDC} > 0$). Captum Integrated Gradients demonstrates superior stability across all 4 regional speech distributions.

## 🧪 Ablation Plan

1. **Ablation 1 (Pooling Mechanism)**:
   Mean Pooling vs. Max Pooling vs. Attentive Statistics Pooling (ASP).
   *Hypothesis*: ASP will yield higher recall on transient retroflex bursts ($[ʈ]$) due to standard-deviation capture.
2. **Ablation 2 (Encoder Freezing Depth)**:
   Frozen Backbone vs. Unfreezing Top-2 Layers vs. Full Fine-Tuning.
   *Hypothesis*: Unfreezing Top-2 layers will achieve optimal balance between domain adaptation and speaker overfitting.
3. **Ablation 3 (Pretraining Corpus)**:
   WavLM Base+ (English + Noise) vs. IndicWav2Vec (40 Indian languages).
   *Hypothesis*: Evaluates whether Indian phonetic pre-training outperforms global English pre-training.
