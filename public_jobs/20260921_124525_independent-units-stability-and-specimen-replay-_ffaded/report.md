# Independent discrepancy audit

All 22 measured inputs were replayed at three timesteps. No fitted parameters were changed.

- measured_persistent_count: 11
- predicted_persistent_count: 0
- late_mean_voltage_RMSE_mV: 45.33282150422331
- maximum_late_mean_change_025_to_0125_mV: 0.2660387507407336
- minimum_major_branch_Ccrit_pF: 2.7959080422173814
- all_major_branch_equilibria_locally_stable_at_039pF: True
- conditional_active_volume_um3_at_cv_3e6: 0.015957745333333332
- conditional_width_um_at_gap_02um_thickness_015um: 0.5319248444444444

The stability table linearizes each fresh major branch with hysteresis memory frozen. It is neither a full hysteretic bifurcation proof nor a calibrated replacement capacitance. The active-volume calculation assumes all fitted heat capacity belongs to VO2 with volumetric heat capacity 3 MJ/m³/K; it is a geometric plausibility check only. Source hashes and the dirty-tree patch preserve the examined state. See docs/DISCREPANCY_AUDIT_20260921.md for derivation, literature, bug scope, and experiments.
