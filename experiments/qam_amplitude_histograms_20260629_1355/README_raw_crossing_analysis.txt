QAM16 vs QAM64 raw crossing analysis

Target SNR values: [-4, -2, 0, 2] dB
Bins per histogram: 20
Signals used: all available signals for each modulation/SNR block

Processing rules:
- No random sample selection.
- No normalization.
- No mean subtraction.
- Raw amplitude: A(t) = sqrt(I(t)^2 + Q(t)^2).
- Raw phase: phase(t) = atan2(Q(t), I(t)), in radians.
- For each SNR, QAM16 and QAM64 share the exact combined min-to-max range split into 20 equal bins.
- Crossing counts report the number of raw time samples greater than or equal to each threshold.

Main figures:
- figures/*_all_signals_raw_amplitude_exact20bins.png
- figures/*_raw_amplitude_crossing_counts.png
- figures/*_all_signals_raw_phase_exact20bins.png
- figures/*_raw_phase_crossing_counts.png
- figures/all_target_snrs_raw_amplitude_crossing_counts.png
- figures/all_target_snrs_raw_amplitude_exact20bin_histograms.png
- figures/all_target_snrs_raw_phase_crossing_counts.png
- figures/all_target_snrs_raw_phase_exact20bin_histograms.png

Main tables:
- results/raw_amplitude_phase_summary_all_signals.csv
- results/all_signals_raw_amplitude_exact20bin_histogram_counts.csv
- results/all_signals_raw_phase_exact20bin_histogram_counts.csv
- results/all_signals_raw_amplitude_threshold_crossing_counts.csv
- results/all_signals_raw_phase_threshold_crossing_counts.csv
- results/raw_amplitude_highest_bin_outer_region_counts.csv

The highest-amplitude-bin table is the direct outer-region count summary: it shows how many raw time samples fall in the largest amplitude bucket for QAM16 and QAM64 at each SNR.