Wave Atlas v1 Release Notes

Tag: wave-atlas-v1
Commit: e49cc81e9479f3fa8252232c3b29eea8c6cc6fe6
PDF size: 924862 bytes
PDF SHA256: 75EA5DF4D63D24B3C695482751ED6EF631E5CF0F35FA03527FA0E39E65E42A00

Milestones:
- M1: baseline wave atlas generator (occupancy, values, diagonals).
- M2: wave metrics (column periodicity, q-rays, diagonal impulses).
- M3: baselines and scaling checks (H_K/K, 1/k, q-ray formula).
- M4: wheel overlay (LCM centers vs wave features).
- M5: twin signal on t-axis (FFT, autocorr, residue heatmaps).
- M6: wheel-shift control (11↔13 signature shift).
- M7: detrend and conditioning (remove p0 signature).
- M8: sequential conditioning (layered sieve decay).
- M9: GT bridge (consecutive rows).
- M9b: GT bridge on wheel-lattice rows (restored variability).

Build:
cd docs && lualatex wave_atlas.tex && lualatex wave_atlas.tex

Artifacts:
out/wave_atlas/{m1..m9b}/...

Wave Atlas v1.1 Release Notes

Tag: wave-atlas-v1.1
Commit: 7dbc915c30a4fba5129b85a9b1b6a7024b2457c3
Changes: add ROADMAP.md and Appendix Roadmap/Future Work in wave_atlas.tex.
PDF size: 926879 bytes
PDF SHA256: 1BDB7586A4ABD156783006C8F1570AD8FFC785B1C25421548AB55490A7798456
