VO2 SETTLED OSCILLATION SEARCH — running locally

Started: 22 September 2026, 20:51 Israel time.
Nominal completion: 22:51 Israel time. One active simulation may extend the limit slightly.
PID: 31846. No ChatGPT or API calls are made by this job.
The process is detached and uses a frozen source snapshot. Idle sleep is inhibited
only for the job lifetime; keep the Mac powered on. Closing this chat will not stop it.

Target: original source labels 300,350,400,450,500,550,600,650,700.
These correspond to actual currents approximately 228–533 µA.
Score window: 150–250 ns. Startup, phase and DC offset are excluded.
Frequency, amplitude and late persistence are optimized with a static R(T)
log10 RMSE ceiling of 0.05. Eight DE/Powell restarts use different starts;
two keep the major-loop parameters fixed, six allow constrained changes.
The original eef073 and static reference are preserved for final comparison.

Open Live status.json for progress. Results/checkpoint.json saves the best values
and elites; Results/optimization_history.jsonl saves every unique evaluation.
After completion inspect Results/summary.csv, parameters.csv, verification.csv
and figures/settled_comparison.png. Compare the finest timestep before accepting
a fit. Near-perfect agreement is a goal, not a promised result.

To stop gracefully, create an empty file named STOP inside the Results folder.
The stop is checked between simulations and may skip unfinished verification.

Run directory: /Users/gabrielberezovsky/Documents/Projects/single-neuristor/runs/20260922_175152_background-settled-oscillations-for-original-lab_f426de
Frozen sources: /Users/gabrielberezovsky/Documents/Projects/single-neuristor/outputs/steady_background_20260922_205149/source
