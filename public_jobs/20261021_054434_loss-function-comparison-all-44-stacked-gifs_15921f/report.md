# All-input loss-function animations

One GIF for every experimental input under each retained loss candidate. No optimization was repeated.
The original finest verification timestep and full measured current prehistory are replayed.
Every replay must agree with its saved voltage to within 0.000001 mV over the complete saved acquisition grid.
Parent files are unchanged; their hashes and the exact parameter vectors are included.

The top panel is the baseline-corrected measured current, not the acquisition filename number.
Titles use its 50–250 ns average. The middle panel overlays baseline-corrected measured output and
simulated device voltage without phase alignment or rescaling. Both measured channels keep their
original acquisition times; native solver voltage is shown without temporal decimation.
Playback shows −30 to 250 ns; earlier measured current prehistory still drives the integration.

The bottom panel keeps every native simulated resistance/temperature endpoint, with heating and
cooling major branches as guides. Large per-step jumps (>0.25 K or >0.08 in ln R) are broken rather
than connected by invented chords; counts appear in animation_index.csv. Branch guides stop at the
measured temperature domain. Some trajectories leave that domain, explicitly flagged in the viewer.
R and T are simulated, not inferred from measured V/I. Compressed NPZ files retain native plotted
simulation samples and both complete acquisition channels. Full-trace PNGs accompany every GIF.

These are diagnostic candidates, not accepted specimen calibrations. Neither objective produced
a satisfactory all-current fit; displaying all 22 records includes failures and stable controls.
GIF axes fit each record, so use the labeled scales when comparing amplitudes between records.
The animations slow physical time; frame count changes playback only, never integration.
