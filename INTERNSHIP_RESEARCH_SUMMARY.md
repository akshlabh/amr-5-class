# Internship Research Summary — AMR with MCLDNN, Attention, Physical Features, Differential Attention, and SNR-aware Hybrid Models

Generated: 2026-07-09  
Repository root: `D:\ism-new\AMR`

This document summarizes the research and implementation work done in the `src/` folder during the internship. It also lists the corresponding configs, notebooks, result files, and graph locations.

---

## 1. Research Objective

The main objective was automatic modulation recognition (AMR) on RadioML2016.10a IQ signals. The work started from an MCLDNN-style baseline and gradually explored:

- LSTM capacity changes.
- Self-attention as a replacement for LSTM.
- Cost-sensitive training for QAM16/QAM64 confusion.
- Physical signal features such as amplitude, phase, PAR, boundary crossing, and excess amplitude.
- Differential attention for noisy/low-SNR robustness.
- SNR-aware hybrid attention.
- Automatic SNR-gated hybrid attention.
- Extension from 5 classes to all 11 RadioML2016.10a classes.

Primary 5-class set:

```text
BPSK, QPSK, 8PSK, QAM16, QAM64
```

Full 11-class set:

```text
8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
```

---

## 2. Core Dataset and Training Infrastructure

### Dataset loader

Main file:

- `src/dataset.py`

Important functionality added/used:

- Class filtering through `selected_classes`.
- Defined subsets:
  - `FIVE_CLASS`
  - `FOUR_CLASS`
  - `ALL_CLASSES`
- Per-sample RMS normalization:

```text
X_normalized = X / RMS(X)
```

- Optional SNR filtering using `snr_range=(lo, hi)`.
- Optional global shuffle split using `shuffle_split=True`.
- Default split remains deterministic per `(modulation, SNR)` block:

```text
60% train / 20% validation / 20% test
```

Why this mattered:

- All models could be trained and compared on the same class set and split logic.
- SNR-wise accuracy curves could be generated consistently.
- Normalization removed absolute received power, forcing models and SNR gate features to learn structure/noise behaviour instead of raw magnitude scale.

### Main training script

Main file:

- `src/train.py`

Role:

- Reads YAML configs from `configs/`.
- Loads data using `src/dataset.py`.
- Builds the selected model.
- Trains with callbacks.
- Saves:
  - checkpoints
  - training curves
  - test score
  - confusion matrices
  - accuracy vs SNR plots

Common output structure:

```text
experiments/<experiment_name>/
  checkpoints/best_model.weights.h5
  logs/
  figures/
  results/
```

---

## 3. Baseline MCLDNN and LSTM Variants

### Files

- `src/models/mcldnn.py`
- `src/models/mcldnn_lstm1.py`
- `src/models/mcldnn_lstm64.py`
- `configs/exp_5class_baseline.yaml`
- `configs/exp_5class_lstm1.yaml`
- `configs/exp_5class_lstm64.yaml`
- `notebooks/lstm1-vs-lstm64.ipynb`

### Theory

The original MCLDNN combines:

1. 2D convolution over the joint IQ frame.
2. 1D convolution over I and Q channels separately.
3. LSTM layers for temporal sequence modelling.
4. Dense classifier for modulation prediction.

The LSTM variants tested whether recurrent model capacity affected performance:

- `LSTM-1`: much smaller recurrent layer.
- `LSTM-64`: reduced recurrent capacity compared with the larger baseline.

### Main results

| Model | Test accuracy |
|---|---:|
| Baseline MCLDNN | 59.80% |
| LSTM-1 | 56.93% |
| LSTM-64 | 57.96% |

### Graphs and results

- Baseline:
  - Results: `experiments/5class_baseline/results/`
  - Graphs: `experiments/5class_baseline/figures/`
- LSTM-1:
  - Results: `experiments/5class_lstm1/results/`
  - Graphs: `experiments/5class_lstm1/figures/`
- LSTM-64:
  - Results: `experiments/5class_lstm64/results/`
  - Graphs: `experiments/5class_lstm64/figures/`
- Four-way comparison:
  - `experiments/comparison_lstm_variants/fourway_acc_vs_snr.png`
  - `experiments/comparison_lstm_variants/param_efficiency.png`

### Conclusion

The normal attention model later outperformed the LSTM variants with fewer parameters, especially at medium/high SNR. The LSTM models were not the strongest direction for this dataset.

---

## 4. Normal Attention Model

### Files

- `src/models/mcldnn_attention.py`
- `configs/exp_5class_attention.yaml`
- `notebooks/attention-mech.ipynb`
- `notebooks/diffattention-train-compare.ipynb`

### Theory

The normal attention model keeps the MCLDNN convolutional front-end but replaces the LSTM temporal modelling block with self-attention.

Flow:

```text
IQ input
  -> CNN feature extraction
  -> sequence representation of length 124
  -> positional encoding
  -> multi-head self-attention
  -> residual + normalization
  -> feed-forward block
  -> mean/max temporal pooling
  -> dense classifier
```

Why attention was used:

- LSTM processes sequence step-by-step.
- Self-attention can compare all time steps directly.
- It can focus on relevant parts of the IQ sequence without relying only on recurrent memory.

### Main result

| Model | Test accuracy |
|---|---:|
| Normal attention | 61.23% |

### Graphs and results

- Results:
  - `experiments/5class_attention/results/test_score.csv`
  - `experiments/5class_attention/results/acc_per_snr.csv`
- Graphs:
  - `experiments/5class_attention/figures/acc_vs_snr.png`
  - `experiments/5class_attention/figures/acc_per_class_vs_snr.png`
  - `experiments/5class_attention/figures/confusion_all_snrs.png`
  - `experiments/5class_attention/figures/attn_profile_snr*.png`

### Conclusion

Normal attention became the strongest standard 5-class model among the early models and was used as the main baseline for later comparisons.

---

## 5. Cost-sensitive Attention for QAM Confusion

### Files

- `src/losses/cost_sensitive.py`
- `src/train_costsensitive.py`
- `configs/exp_5class_attention_costsensitive.yaml`
- `configs/exp_5class_attention_qam_shifted_cost.yaml`

### Theory

The goal was to reduce QAM16/QAM64 confusion by changing the loss cost:

- Correct QAM16/QAM64 prediction could be rewarded or cost-shifted.
- QAM16 predicted as QAM64 and QAM64 predicted as QAM16 were penalized more strongly.

The idea:

```text
standard cross entropy + custom class-pair penalty
```

This directly targets a known confusion pair instead of treating all mistakes equally.

### Graphs and results

- Cost-sensitive attention:
  - Results: `experiments/5class_attention_costsensitive/results/`
  - Graphs: `experiments/5class_attention_costsensitive/figures/`
- Shifted QAM cost:
  - Results: `experiments/5class_attention_qam_shifted_cost/results/`
  - Graphs: `experiments/5class_attention_qam_shifted_cost/figures/`
- Comparison:
  - `experiments/comparison_costsensitive/acc_vs_snr.png`
  - `experiments/comparison_costsensitive/confusion_comparison.png`
  - `experiments/comparison_costsensitive/snr_comparison.csv`

### Conclusion

Cost-sensitive training was useful for studying targeted QAM errors, but later physical-feature and hybrid approaches gave more interpretable explanations.

---

## 6. Physical Feature Attention: Amplitude and Phase

### Files

- `src/features/signal_features.py`
- `src/models/mcldnn_attention_phys.py`
- `configs/exp_5class_attention_phys.yaml`

### Theory

The professor suggested using physical properties of IQ signals:

```text
x(t) = I(t) + jQ(t)
```

Amplitude:

```text
A(t) = sqrt(I(t)^2 + Q(t)^2)
```

Differential phase:

```text
dphi(t) = angle(x(t) * conjugate(x(t-1)))
```

Implemented features:

Amplitude branch:

```text
A(t), A(t)^2, ΔA(t), |ΔA(t)|
```

Phase branch:

```text
cos(dphi), sin(dphi)
```

Fusion:

- Main IQ attention branch extracts learned temporal features.
- Physical branch extracts handcrafted amplitude/phase cues.
- The two are fused after attention pooling using a learned gate.

### Main result

| Model | Test accuracy |
|---|---:|
| Physical-feature attention | 60.88% |
| Normal attention reference | 61.23% |

### Observed behaviour

- Some improvement in transition SNR regions.
- QAM64 to QAM16 confusion reduced in one analysis.
- QAM16 to QAM64 and QPSK to 8PSK confusion could increase.
- Phase branch may be too strong or noisy for some SNRs.

### Graphs and results

- Results:
  - `experiments/5class_attention_phys/results/test_score.csv`
  - `experiments/5class_attention_phys/results/acc_per_snr.csv`
- Graphs:
  - `experiments/5class_attention_phys/figures/acc_vs_snr.png`
  - `experiments/5class_attention_phys/figures/acc_per_class_vs_snr.png`
  - `experiments/5class_attention_phys/figures/confusion_all_snrs.png`
  - `experiments/5class_attention_phys/figures/confusion_snr*.png`

### Conclusion

Amplitude and phase features gave a physically meaningful direction, but phase features were not always reliable. This led to the next stage: amplitude-only and PAR-focused models.

---

## 7. Amplitude-only, PAR, and QAM Boundary Features

### Files

- `src/features/signal_features.py`
- `src/models/mcldnn_attention_amp.py`
- `src/models/mcldnn_attention_amp_lite.py`
- `src/models/mcldnn_attention_amp_static.py`
- `src/models/mcldnn_diffattention_amp_focus.py`
- `configs/exp_5class_attention_amp.yaml`
- `configs/exp_5class_attention_amp_lite.yaml`
- `configs/exp_5class_attention_amp_static.yaml`
- `configs/exp_5class_diffattention_amp_focus.yaml`
- Notebooks:
  - `notebooks/amplitude_only.ipynb`
  - `notebooks/amplitude-par-done.ipynb`
  - `notebooks/amplitude-multiborder.ipynb`
  - `notebooks/amplitude_lite.ipynb`
  - `notebooks/amplitude_static.ipynb`
  - `notebooks/diffattention-amp-focus.ipynb`

### Theory

The QAM16 vs QAM64 confusion is strongly related to amplitude distribution.

QAM64 has more constellation points and more outer-ring amplitude levels than QAM16. Therefore, useful cues include:

- how often amplitude crosses the QAM16 outer boundary
- how large the excess amplitude is
- peak-to-average ratio
- high-percentile amplitude values

Implemented amplitude/PAR features included:

Sequence features:

```text
A(t)
A(t)^2
boundary_crossing(t)
excess_amplitude(t)
relative_excess(t)
relative_excess(t)^2
```

Global features:

```text
log1p(PAR)
peak_ratio
mean_excess
max_excess
amplitude_std
amplitude_p95
amplitude_p99
```

Where:

```text
PAR = max(A(t)^2) / mean(A(t)^2)
```

QAM16 boundary was estimated from training data only, usually using a high percentile of QAM16 amplitude.

### Main result

| Model | Test accuracy |
|---|---:|
| Diff attention + amplitude/PAR focus | 59.21% |
| Plain diff attention reference | 57.34% |
| Normal attention reference | 61.23% |

### Graphs and results

- Diff attention + amplitude focus:
  - Results: `experiments/5class_diffattention_amp_focus/results/`
  - Graphs: `experiments/5class_diffattention_amp_focus/figures/`
  - Comparison graphs:
    - `experiments/5class_diffattention_amp_focus/results/comparison_with_normal_and_diffattention/normal_plain_diff_vs_amp_focus_acc_vs_snr.png`
    - `experiments/5class_diffattention_amp_focus/results/comparison_with_normal_and_diffattention/normal_plain_diff_vs_amp_focus_summary.csv`

### Conclusion

PAR and boundary-crossing features improved the plain differential attention model but still did not beat normal attention overall. The most useful conclusion was interpretability: QAM64/QAM16 separation should rely more on stable amplitude-distribution statistics than noisy adjacent-step transitions.

---

## 8. QAM16 vs QAM64 Histogram and Raw Amplitude/Phase Analysis

### Files

- `notebooks/qam_amplitude_histograms.ipynb`

### Theory

The goal was to inspect raw amplitude and phase distributions of QAM16 and QAM64 at selected SNRs:

```text
-4 dB, -2 dB, 0 dB, +2 dB
```

The analysis used exact 20-bin histograms over the raw amplitude/phase range without normalization or mean subtraction, as requested.

For each SNR:

- Raw amplitude histograms.
- Raw phase histograms.
- Threshold/crossing count plots.
- All-signal aggregated counts.

### Important result observation

The histograms often overlapped strongly because:

- Dataset normalization and noise reduce absolute amplitude separation.
- At low SNR, noise dominates symbol-level amplitude structure.
- QAM16 and QAM64 both occupy similar observed amplitude ranges after channel/noise effects.
- Single raw histograms may not fully reveal constellation-ring differences.

This supported the later move toward PAR, percentiles, excess amplitude, and boundary-crossing features rather than only raw histogram overlap.

### Graphs and results

- Results:
  - `experiments/qams_histograms/results/`
- Graphs:
  - `experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_exact20bin_histograms.png`
  - `experiments/qams_histograms/figures/all_target_snrs_raw_phase_exact20bin_histograms.png`
  - `experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_crossing_counts.png`
  - `experiments/qams_histograms/figures/all_target_snrs_raw_phase_crossing_counts.png`
  - SNR-specific files:
    - `experiments/qams_histograms/figures/snr_minus_04db_*`
    - `experiments/qams_histograms/figures/snr_minus_02db_*`
    - `experiments/qams_histograms/figures/snr_plus_00db_*`
    - `experiments/qams_histograms/figures/snr_plus_02db_*`

---

## 9. Differential Attention

### Files

- `src/models/mcldnn_diffattention.py`
- `configs/exp_5class_diffattention.yaml`
- `configs/exp_5class_diffattention_lowsnr.yaml`
- `notebooks/diffattention-train-compare.ipynb`
- `notebooks/diffattention-lowsnr-vs-allsnr.ipynb`

### Theory

Normal self-attention computes one attention distribution:

```text
Attention = softmax(QK^T / sqrt(d)) V
```

Differential attention computes two attention distributions and subtracts one from the other:

```text
DiffAttention = (softmax(Q1K1^T / sqrt(d)) - λ softmax(Q2K2^T / sqrt(d))) V
```

Intuition:

- Both attention maps may capture common/noisy similarity.
- Subtraction can suppress common-mode attention noise.
- The remaining signal may focus on more discriminative time steps.

Implementation details:

- Keras 3 custom layer.
- Gated cancellation so the model starts close to normal attention.
- RMS normalization after differential attention.
- Same convolutional front-end as normal attention.

### Main result

| Model | Test accuracy | Mean SNR accuracy | Peak SNR accuracy |
|---|---:|---:|---:|
| Normal attention | 61.23% | 61.23% | 92.70% |
| Differential attention | 57.34% | 57.34% | 79.70% |

Differential attention helped in some low-SNR regions but was weaker at high SNR.

### Graphs and results

- Results:
  - `experiments/5class_diffattention/results/test_score.csv`
  - `experiments/5class_diffattention/results/acc_per_snr.csv`
- Graphs:
  - `experiments/5class_diffattention/figures/acc_vs_snr.png`
  - `experiments/5class_diffattention/figures/confusion_all_snrs.png`
  - `experiments/5class_diffattention/figures/confusion_snr*.png`
- Normal vs diff comparison:
  - `experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_acc_vs_snr.png`
  - `experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_summary.csv`

### Conclusion

Differential attention was not better overall, but it motivated the SNR-aware hybrid idea: use differential attention where noise is high and normal attention where signal structure is cleaner.

---

## 10. Manual Hybrid SNR-aware Attention

### Files

- `src/evaluate_hybrid_snr_aware_attention.py`
- `notebooks/hybrid-snr-aware-attention.ipynb`

### Theory

Manual hybrid combines two trained models:

```text
Normal attention model
Differential attention model
```

Routing is based on SNR:

```text
low/transition SNR -> differential attention
cleaner SNR        -> normal attention
```

The final prediction is selected at inference time. No new classifier is trained.

### Main result

| Model | Accuracy |
|---|---:|
| Normal attention | 61.23% |
| Manual hybrid SNR-aware attention | 62.35% |
| Improvement | +1.12 percentage points |

### Graphs and results

- Results:
  - `experiments/5class_hybrid_snr_aware_attention/results/test_score.csv`
  - `experiments/5class_hybrid_snr_aware_attention/results/acc_per_snr.csv`
  - `experiments/5class_hybrid_snr_aware_attention/results/hybrid_snr_selection.csv`
- Graphs:
  - `experiments/5class_hybrid_snr_aware_attention/figures/hybrid_vs_normal_acc_vs_snr.png`
  - `experiments/5class_hybrid_snr_aware_attention/figures/hybrid_minus_normal_delta_by_snr.png`
  - `experiments/5class_hybrid_snr_aware_attention/figures/confusion_all_snrs.png`
  - `experiments/5class_hybrid_snr_aware_attention/figures/confusion_normal_all_snrs.png`

### Conclusion

The manual hybrid improved over normal attention by using differential attention selectively. This supported the hypothesis that differential attention is mainly useful in low/noisy SNR regimes.

---

## 11. Lightweight SNR Gate

### Files

- `src/models/snr_gate.py`
- `src/train_snr_gate.py`
- `src/features/signal_features.py`
- `configs/exp_snr_gate.yaml`
- `notebooks/snr-gate-lightweight-train.ipynb`

### Theory

The manual hybrid needs the true SNR label. In a practical receiver, true SNR may not be available. Therefore, a lightweight SNR gate was trained to predict:

```text
class 0: SNR <= 0 dB
class 1: SNR > 0 dB
```

The SNR gate does not classify modulation. It only decides the routing region.

Input features are extracted from IQ samples:

```text
x(t) = I(t) + jQ(t)
A(t) = sqrt(I(t)^2 + Q(t)^2)
```

Feature groups:

- Amplitude spread.
- Peak-to-average ratio.
- Amplitude percentiles.
- Temporal roughness.
- Phase variation.
- Short-lag correlation.

Because the dataset loader RMS-normalizes each IQ window, the SNR gate does not simply use absolute signal power. It learns noise/cleanliness patterns from the signal structure.

### Main result

| Model | Accuracy |
|---|---:|
| Lightweight SNR gate | 99.16% |

Routing summary:

```text
threshold_db = 0.0
snr_gate_route_accuracy = 0.9916
manual_diff_route_count = 11000
manual_normal_route_count = 9000
gated_diff_route_count = 11102
gated_normal_route_count = 8898
```

### Graphs and results

- Results:
  - `experiments/snr_gate_lightweight/results/test_score.csv`
  - `experiments/snr_gate_lightweight/results/snr_gate_acc_per_snr.csv`
  - `experiments/snr_gate_lightweight/results/confusion_normalized.csv`
  - `experiments/snr_gate_lightweight/results/snr_gate_feature_names.csv`
  - `experiments/snr_gate_lightweight/results/snr_gate_metadata.json`
- Graphs:
  - `experiments/snr_gate_lightweight/figures/snr_gate_accuracy_vs_snr.png`
  - `experiments/snr_gate_lightweight/figures/snr_gate_p_low_vs_snr.png`
  - `experiments/snr_gate_lightweight/figures/snr_gate_confusion.png`
  - `experiments/snr_gate_lightweight/logs/snr_gate_accuracy.png`
  - `experiments/snr_gate_lightweight/logs/snr_gate_loss.png`

### Conclusion

The SNR gate successfully automated the low/high SNR decision, enabling a practical gated hybrid system without giving the true SNR directly to the modulation classifier.

---

## 12. Gated Hybrid SNR-aware Attention

### Files

- `src/evaluate_gated_hybrid_snr_aware_attention.py`
- `notebooks/gated-hybrid-snr-aware-attention.ipynb`

### Theory

The gated hybrid system uses:

```text
IQ signal
  -> SNR gate predicts low/high SNR
  -> low predicted: use differential attention
  -> high predicted: use normal attention
  -> final modulation prediction
```

Important:

- True SNR is not passed to the gated hybrid for routing.
- True SNR is used only afterward for evaluation and plotting.
- The selected normal/diff model receives only IQ inputs, not SNR.

### Main result

| Model | Accuracy |
|---|---:|
| Normal attention | 61.23% |
| Differential attention | 57.34% |
| Manual hybrid using true SNR | 62.16% |
| Gated hybrid using predicted SNR | 62.19% |

### Graphs and results

- Results:
  - `experiments/5class_gated_hybrid_snr_aware_attention/results/test_score.csv`
  - `experiments/5class_gated_hybrid_snr_aware_attention/results/routing_summary.csv`
  - `experiments/5class_gated_hybrid_snr_aware_attention/results/manual_vs_gated_acc_per_snr.csv`
  - `experiments/5class_gated_hybrid_snr_aware_attention/results/snr_gate_route_confusion_normalized.csv`
- Graphs:
  - `experiments/5class_gated_hybrid_snr_aware_attention/figures/manual_vs_gated_hybrid_acc_vs_snr.png`
  - `experiments/5class_gated_hybrid_snr_aware_attention/figures/gated_minus_manual_delta_by_snr.png`
  - `experiments/5class_gated_hybrid_snr_aware_attention/figures/snr_gate_route_confusion.png`
  - `experiments/5class_gated_hybrid_snr_aware_attention/figures/confusion_manual_hybrid_all_snrs.png`
  - `experiments/5class_gated_hybrid_snr_aware_attention/figures/confusion_gated_hybrid_all_snrs.png`

### Conclusion

The gated hybrid nearly matched the manual hybrid while removing the need for true SNR at inference time. This is a more practical system design.

---

## 13. 11-class Attention and Manual Hybrid Extension

### Files

- `configs/exp_11class_attention.yaml`
- `configs/exp_11class_diffattention.yaml`
- `src/evaluate_11class_manual_hybrid_snr_aware_attention.py`
- `notebooks/11class-attention-diff-manual-hybrid.ipynb`

### Theory

The 5-class attention experiments were extended to all 11 RadioML2016.10a classes.

Models trained:

1. 11-class normal attention.
2. 11-class differential attention.
3. 11-class manual SNR-aware hybrid.

Manual hybrid rule:

```text
SNR <= 0 dB -> differential attention
SNR > 0 dB  -> normal attention
```

No gated model was used in this 11-class experiment.

### Main result

| Model | Accuracy |
|---|---:|
| 11-class normal attention | 64.09% |
| 11-class differential attention | 62.80% |
| 11-class manual hybrid SNR-aware | 64.18% |

### SNR-wise observation

The 11-class differential model was better than normal attention at several low/transition SNR values, for example around `-14 dB` to `0 dB`, while normal attention remained stronger at some very low and high SNR values. The manual hybrid gave a small overall gain over normal attention.

### Graphs and results

- Normal attention:
  - Results: `experiments/11class_attention/results/`
  - Graphs: `experiments/11class_attention/figures/`
- Differential attention:
  - Results: `experiments/11class_diffattention/results/`
  - Graphs: `experiments/11class_diffattention/figures/`
- Manual hybrid:
  - Results: `experiments/11class_manual_hybrid_snr_aware_attention/results/`
  - Graphs:
    - `experiments/11class_manual_hybrid_snr_aware_attention/figures/normal_vs_diff_vs_manual_hybrid_acc_vs_snr.png`
    - `experiments/11class_manual_hybrid_snr_aware_attention/figures/manual_hybrid_minus_normal_delta_by_snr.png`
    - `experiments/11class_manual_hybrid_snr_aware_attention/figures/confusion_normal_attention_all_snrs.png`
    - `experiments/11class_manual_hybrid_snr_aware_attention/figures/confusion_diff_attention_all_snrs.png`
    - `experiments/11class_manual_hybrid_snr_aware_attention/figures/confusion_manual_hybrid_snr_aware_all_snrs.png`

### Conclusion

The 11-class extension confirmed that the same attention and hybrid framework can scale beyond the 5-class PSK/QAM subset. The manual hybrid gave a modest improvement over normal attention.

---

## 14. Branch Ablation and K-fold Work

### Files

- `src/models/mcldnn_ablation.py`
- `src/analyze_branch_ablation.py`
- `src/train_kfold.py`
- `configs/branch_ablation/`
- `configs/exp_5class_attention_kfold.yaml`
- `configs/exp_5class_baseline_kfold.yaml`
- `notebooks/train_4class_ablation.ipynb`

### Theory

Branch ablation tested whether I-only, Q-only, or IQ-combined branches were responsible for most of the discriminative power.

K-fold experiments tested stability across folds instead of relying on only one train/validation/test split.

### Graphs and results

- Branch results:
  - `experiments/branch_5class/I_only/`
  - `experiments/branch_5class/Q_only/`
  - `experiments/branch_5class/IQ_only/`
- K-fold comparison:
  - `experiments/comparison_kfold/`
  - `experiments/5class_attention_kfold/kfold/`
  - `experiments/5class_baseline_kfold/kfold/`

### Conclusion

These experiments supported fairer model comparison and helped verify that changes were not only due to a lucky split.

---

## 15. Summary of Key Results

| Experiment | Accuracy |
|---|---:|
| Baseline MCLDNN | 59.80% |
| LSTM-1 | 56.93% |
| LSTM-64 | 57.96% |
| Normal attention | 61.23% |
| Physical-feature attention | 60.88% |
| Differential attention | 57.34% |
| Diff attention + amplitude/PAR focus | 59.21% |
| Manual hybrid SNR-aware attention | 62.35% |
| SNR gate route classifier | 99.16% |
| Gated hybrid SNR-aware attention | 62.19% |
| 11-class normal attention | 64.09% |
| 11-class differential attention | 62.80% |
| 11-class manual hybrid SNR-aware attention | 64.18% |

---

## 16. Outcome Comparison: Which Approach Was Better and Why

This section directly compares the major experiments by metric and states the outcome.

### 16.1 Baseline MCLDNN vs Normal Attention

Metric:

```text
Overall test accuracy
```

| Model | Accuracy |
|---|---:|
| Baseline MCLDNN | 59.80% |
| Normal attention | 61.23% |

Outcome:

```text
Normal attention was better by +1.43 percentage points.
```

Reason:

- The attention model replaces recurrent temporal compression with direct time-step comparison.
- This helped capture repeated IQ temporal patterns better than the LSTM baseline.
- It also used fewer parameters than the larger LSTM baseline in the comparison work.

Result files:

```text
experiments/5class_baseline/results/test_score.csv
experiments/5class_attention/results/test_score.csv
experiments/comparison_lstm_variants/fourway_acc_vs_snr.png
```

Conclusion:

```text
Normal attention became the main 5-class baseline.
```

### 16.2 LSTM-1 vs LSTM-64 vs Normal Attention

Metric:

```text
Overall test accuracy and SNR-wise accuracy curve
```

| Model | Accuracy |
|---|---:|
| LSTM-1 | 56.93% |
| LSTM-64 | 57.96% |
| Normal attention | 61.23% |

Outcome:

```text
Normal attention was better than LSTM-1 by +4.30 percentage points.
Normal attention was better than LSTM-64 by +3.28 percentage points.
```

Reason:

- The smaller LSTM variants reduced temporal modelling capacity.
- Attention was better at using long-range relationships across the 124-step feature sequence.

Graph:

```text
experiments/comparison_lstm_variants/fourway_acc_vs_snr.png
```

Conclusion:

```text
Reducing LSTM size did not improve performance. Attention was the stronger direction.
```

### 16.3 Normal Attention vs Physical-feature Attention

Metric:

```text
Overall test accuracy, confusion behaviour, and SNR-wise accuracy
```

| Model | Accuracy |
|---|---:|
| Normal attention | 61.23% |
| Physical-feature attention | 60.88% |

Outcome:

```text
Normal attention was slightly better overall by +0.35 percentage points.
```

However, physical-feature attention gave useful interpretability.

Reason:

- Amplitude and phase features made the physical basis of classification more explicit.
- But phase features could also increase QPSK/8PSK confusion.
- Overall performance was close but not better than normal attention.

Result files:

```text
experiments/5class_attention/results/test_score.csv
experiments/5class_attention_phys/results/test_score.csv
experiments/5class_attention_phys/figures/confusion_all_snrs.png
```

Conclusion:

```text
Physical features were valuable for analysis, but the full amplitude+phase model was not the best final classifier.
```

### 16.4 Normal Attention vs Differential Attention

Metric:

```text
Overall test accuracy, mean SNR accuracy, peak SNR accuracy
```

| Model | Test accuracy | Mean SNR accuracy | Peak SNR accuracy |
|---|---:|---:|---:|
| Normal attention | 61.23% | 61.23% | 92.70% |
| Differential attention | 57.34% | 57.34% | 79.70% |

Outcome:

```text
Normal attention was better overall by +3.90 percentage points.
```

Reason:

- Differential attention suppresses common attention patterns, which can help in noisy regions.
- But in high SNR, the normal attention model better exploited clean signal structure.
- Differential attention saturated earlier at high SNR.

Result files:

```text
experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_summary.csv
experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_acc_vs_snr.png
```

Conclusion:

```text
Differential attention alone was not the best overall model, but it motivated SNR-aware hybrid routing.
```

### 16.5 Plain Differential Attention vs Differential Attention + Amplitude/PAR Focus

Metric:

```text
Overall test accuracy and peak SNR accuracy
```

| Model | Accuracy | Peak SNR accuracy |
|---|---:|---:|
| Plain differential attention | 56.13% to 57.34% depending run/comparison file |
| Differential attention + amplitude/PAR focus | 59.21% | 84.50% |

Outcome:

```text
Amplitude/PAR focus improved the differential-attention branch.
```

From the comparison summary:

```text
Diff attention + amplitude/PAR focus improved over the plain diff-attention comparison run by about +3.08 percentage points.
```

Reason:

- QAM16/QAM64 confusion is physically linked to amplitude distribution.
- Boundary crossing, excess amplitude, PAR, and p95 amplitude gave the model explicit QAM separation cues.

Result files:

```text
experiments/5class_diffattention_amp_focus/results/test_score.csv
experiments/5class_diffattention_amp_focus/results/comparison_with_normal_and_diffattention/normal_plain_diff_vs_amp_focus_summary.csv
experiments/5class_diffattention_amp_focus/results/comparison_with_normal_and_diffattention/normal_plain_diff_vs_amp_focus_acc_vs_snr.png
```

Conclusion:

```text
Amplitude/PAR features were useful for improving differential attention, but still did not beat normal attention overall.
```

### 16.6 Normal Attention vs Manual Hybrid SNR-aware Attention

Metric:

```text
Overall test accuracy and accuracy-vs-SNR curve
```

| Model | Accuracy |
|---|---:|
| Normal attention | 61.23% |
| Manual hybrid SNR-aware attention | 62.35% |

Outcome:

```text
Manual hybrid was better by +1.12 percentage points.
```

Reason:

- Normal attention is stronger at cleaner SNR.
- Differential attention can be useful in noisy/transition SNR.
- Manual hybrid combines them by selecting the model depending on SNR.

Result files:

```text
experiments/5class_hybrid_snr_aware_attention/results/test_score.csv
experiments/5class_hybrid_snr_aware_attention/results/comparison_with_normal_attention/normal_vs_hybrid_summary.csv
experiments/5class_hybrid_snr_aware_attention/figures/hybrid_vs_normal_acc_vs_snr.png
experiments/5class_hybrid_snr_aware_attention/figures/hybrid_minus_normal_delta_by_snr.png
```

Conclusion:

```text
Manual SNR-aware routing was the best 5-class approach when true SNR is available.
```

### 16.7 Manual Hybrid vs Gated Hybrid

Metric:

```text
Overall test accuracy and SNR-gate route accuracy
```

| Model | Accuracy |
|---|---:|
| Manual hybrid using true SNR | 62.16% |
| Gated hybrid using predicted SNR | 62.19% |

SNR gate:

| Metric | Value |
|---|---:|
| SNR route accuracy | 99.16% |

Outcome:

```text
Gated hybrid approximately matched the manual hybrid and was slightly higher in this run by +0.04 percentage points.
```

Reason:

- The SNR gate predicted the low/high SNR route very accurately.
- Therefore, the gated hybrid behaved almost the same as the manual hybrid.
- Unlike manual hybrid, gated hybrid does not require true SNR during inference.

Result files:

```text
experiments/snr_gate_lightweight/results/test_score.csv
experiments/snr_gate_lightweight/results/snr_gate_acc_per_snr.csv
experiments/5class_gated_hybrid_snr_aware_attention/results/test_score.csv
experiments/5class_gated_hybrid_snr_aware_attention/results/routing_summary.csv
experiments/5class_gated_hybrid_snr_aware_attention/figures/manual_vs_gated_hybrid_acc_vs_snr.png
experiments/5class_gated_hybrid_snr_aware_attention/figures/snr_gate_route_confusion.png
```

Conclusion:

```text
Gated hybrid is the most practical 5-class system because it automates model selection without using true SNR as input.
```

### 16.8 Normal 11-class Attention vs Differential 11-class Attention vs Manual 11-class Hybrid

Metric:

```text
Overall test accuracy and SNR-wise accuracy
```

| Model | Accuracy |
|---|---:|
| 11-class normal attention | 64.09% |
| 11-class differential attention | 62.80% |
| 11-class manual hybrid SNR-aware | 64.18% |

Outcome:

```text
11-class manual hybrid was best overall.
It improved over normal attention by +0.08 percentage points.
It improved over differential attention by +1.38 percentage points.
```

Reason:

- Differential attention helped at some low/transition SNR values.
- Normal attention was stronger overall and at cleaner SNR.
- Manual hybrid recovered some of the low-SNR benefit while preserving high-SNR normal attention behaviour.

Result files:

```text
experiments/11class_attention/results/test_score.csv
experiments/11class_diffattention/results/test_score.csv
experiments/11class_manual_hybrid_snr_aware_attention/results/test_score.csv
experiments/11class_manual_hybrid_snr_aware_attention/results/normal_diff_manual_hybrid_acc_per_snr.csv
experiments/11class_manual_hybrid_snr_aware_attention/figures/normal_vs_diff_vs_manual_hybrid_acc_vs_snr.png
```

Conclusion:

```text
The hybrid idea also scaled to all 11 classes, but the gain was smaller than in the 5-class case.
```

### 16.9 QAM Histogram Analysis Outcome

Metric:

```text
Raw amplitude/phase histogram overlap and outer-bin crossing counts
```

Outcome:

```text
Raw amplitude histograms of QAM16 and QAM64 overlapped strongly.
```

Reason:

- The dataset uses channel effects and noise.
- The loader applies RMS normalization.
- At low SNR, noise dominates amplitude distribution.
- Therefore, raw amplitude alone is not enough to cleanly separate QAM16 and QAM64.

Useful result:

```text
The overlap justified using more robust amplitude statistics:
PAR, p95/p99 amplitude, peak ratio, boundary crossing, and excess amplitude.
```

Result files:

```text
experiments/qams_histograms/results/raw_amplitude_highest_bin_outer_region_counts.csv
experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_exact20bin_histograms.png
experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_crossing_counts.png
```

Conclusion:

```text
Histogram analysis was mainly diagnostic. It showed why simple raw amplitude plots were insufficient and why robust amplitude/PAR features were needed.
```

### 16.10 Final Ranking by Practical Usefulness

For the 5-class work:

| Rank | Approach | Reason |
|---:|---|---|
| 1 | Gated hybrid SNR-aware attention | Nearly matches manual hybrid and does not need true SNR at inference |
| 2 | Manual hybrid SNR-aware attention | Best when true SNR is available |
| 3 | Normal attention | Strongest single model |
| 4 | Physical/amplitude/PAR variants | Best for interpretability and QAM analysis |
| 5 | Plain differential attention | Useful low-SNR idea but weak overall |
| 6 | LSTM variants | Lower performance than attention |

For the 11-class work:

| Rank | Approach | Reason |
|---:|---|---|
| 1 | 11-class manual hybrid | Highest 11-class accuracy |
| 2 | 11-class normal attention | Strongest single 11-class model |
| 3 | 11-class differential attention | Useful at some SNRs but weaker overall |

Overall conclusion:

```text
The best research outcome was not replacing normal attention entirely.
The best outcome was SNR-aware routing: use differential attention only where it helps,
and use normal attention where the signal is cleaner.
```

---

## 17. Most Important Graph Locations

### 5-class attention

```text
experiments/5class_attention/figures/acc_vs_snr.png
experiments/5class_attention/figures/confusion_all_snrs.png
```

### Normal vs differential attention

```text
experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_acc_vs_snr.png
experiments/5class_diffattention/results/comparison_with_normal_attention/normal_vs_diffattention_summary.csv
```

### Manual hybrid

```text
experiments/5class_hybrid_snr_aware_attention/figures/hybrid_vs_normal_acc_vs_snr.png
experiments/5class_hybrid_snr_aware_attention/figures/hybrid_minus_normal_delta_by_snr.png
```

### SNR gate

```text
experiments/snr_gate_lightweight/figures/snr_gate_accuracy_vs_snr.png
experiments/snr_gate_lightweight/figures/snr_gate_p_low_vs_snr.png
experiments/snr_gate_lightweight/figures/snr_gate_confusion.png
```

### Gated hybrid

```text
experiments/5class_gated_hybrid_snr_aware_attention/figures/manual_vs_gated_hybrid_acc_vs_snr.png
experiments/5class_gated_hybrid_snr_aware_attention/figures/gated_minus_manual_delta_by_snr.png
experiments/5class_gated_hybrid_snr_aware_attention/figures/snr_gate_route_confusion.png
```

### 11-class comparison

```text
experiments/11class_manual_hybrid_snr_aware_attention/figures/normal_vs_diff_vs_manual_hybrid_acc_vs_snr.png
experiments/11class_manual_hybrid_snr_aware_attention/figures/manual_hybrid_minus_normal_delta_by_snr.png
```

### QAM16/QAM64 histogram analysis

```text
experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_exact20bin_histograms.png
experiments/qams_histograms/figures/all_target_snrs_raw_phase_exact20bin_histograms.png
experiments/qams_histograms/figures/all_target_snrs_raw_amplitude_crossing_counts.png
experiments/qams_histograms/figures/all_target_snrs_raw_phase_crossing_counts.png
```

### LSTM and attention comparison

```text
experiments/comparison_lstm_variants/fourway_acc_vs_snr.png
experiments/comparison_lstm_variants/param_efficiency.png
```

---

## 18. Overall Internship Conclusion

The strongest practical direction was not a single replacement model, but a physically motivated SNR-aware system:

1. Normal attention is strong overall and especially at cleaner SNRs.
2. Differential attention can be useful in noisy or transition SNR regions.
3. Physical amplitude/PAR features improved interpretability for QAM16/QAM64 confusion.
4. Raw amplitude histograms alone were insufficient because normalized/noisy signals overlap heavily.
5. Manual hybrid attention improved the 5-class result from `61.23%` to `62.35%`.
6. The SNR gate made the hybrid automatic by predicting low/high SNR from IQ features.
7. The gated hybrid nearly matched the manual hybrid without passing true SNR to the modulation classifier.
8. The 11-class extension showed the framework can scale to the full RadioML2016.10a class set.

The final research contribution can be described as:

```text
An MCLDNN-based attention framework with physical-feature analysis,
differential attention for low-SNR robustness, and SNR-aware hybrid routing
for improved and more interpretable AMR performance.
```
