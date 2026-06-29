QAM16 vs QAM64 amplitude histogram analysis

Target SNR values: [-4, -2, 0, 2] dB
Random samples per modulation/SNR: 3
Random seed: 2026

Main figures:
- figures/*_random_3_samples_normalized_histograms.png
  Random single-window normalized amplitude histograms.
- figures/*_random_3_samples_normalized_amplitude_time_traces.png
  Random single-window normalized amplitude A(t)/mean(A) time traces.
- figures/*_random_3_samples_raw_amplitude_time_traces.png
  Random single-window raw amplitude A(t) time traces.
- figures/*_average_normalized_overlay.png
  Average normalized amplitude histogram overlay for QAM16 vs QAM64 at one SNR.
- figures/*_average_normalized_amplitude_time_trace.png
  Average normalized amplitude A(t)/mean(A) trace overlay for QAM16 vs QAM64 at one SNR.
- figures/*_average_raw_amplitude_time_trace.png
  Average raw amplitude A(t) trace overlay for QAM16 vs QAM64 at one SNR.
- figures/all_target_snrs_average_normalized_overlay.png
  Combined average normalized histogram overlays for all target SNRs.
- figures/all_target_snrs_average_normalized_amplitude_time_traces.png
  Combined average normalized amplitude A(t)/mean(A) trace overlays for all target SNRs.
- figures/all_target_snrs_average_raw_amplitude_time_traces.png
  Combined average raw amplitude A(t) trace overlays for all target SNRs.
- figures/all_target_snrs_average_raw_overlay.png
  Combined raw-amplitude overlays for reference.

Main tables:
- results/selected_random_samples.csv
  Exact randomly selected sample indices inside each RML2016.10a modulation/SNR block.
- results/amplitude_summary_statistics.csv
  Mean/std/p95/p99/PAR summary values.
- results/average_normalized_histogram_values.csv
  Average normalized histogram density values for reproducible plotting.
- results/average_normalized_amplitude_time_trace_values.csv
  Average normalized A(t)/mean(A) values for reproducible time-trace plotting.
- results/average_raw_amplitude_time_trace_values.csv
  Average raw A(t) values for reproducible time-trace plotting.
- results/histogram_settings.json
  Bins, seed, and normalization details.

Normalization:
1. Per-window RMS normalization of IQ.
2. A(t) = sqrt(I(t)^2 + Q(t)^2).
3. A_norm(t) = A(t) / mean(A(t)) per 128-sample window.