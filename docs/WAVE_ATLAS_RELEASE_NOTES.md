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

Wave Atlas v1.2 Release Notes

Tag: wave-atlas-v1.2
Commit: d2201adb9b0b6a6e6f5ef2e5a6f3b2f3e62d12b7
Changes: M11 LaTeX table via \input, M12 residual process, M12b inhomogeneous null for dispersion; cleanup raw wheel_scan files.
PDF size: 1059651 bytes
PDF SHA256: A46DE0386126BBE31BE6A7AE0E92AC48970FA2507A85D7D8356A111961E26B45

Wave Atlas v1.3 Release Notes

Tag: wave-atlas-v1.3
Commit: 9cfa74a
Changes: M13 GT wheel-lattice sensitivity; triangles+isolates detect sieve layers; stable across eps.
PDF size: 1103519 bytes
PDF SHA256: 16AE7675EE475E954D825EFED94914E032CAA2A4984F1FAA64CF5DF0214E82C9

Wave Atlas v1.4 Release Notes

Tag: wave-atlas-v1.4
Commit: 27fb4fd
Changes: M14 wave-sieve candidate generator + figs survival/throughput/gaps.
PDF size: 1162068 bytes
PDF SHA256: 679866141B5A6E1CD9462E127DE8C96E048231B3C0224F7922A93772A98D5C2E

Wave Atlas v1.5 Release Notes

Tag: wave-atlas-v1.5
Commit: 4a6274f
Changes: M14b segmented layers efficiency table; depth extended to 150 layers.
PDF size: 1212398 bytes
PDF SHA256: 460E2EAC391E809C8B442D43BD1A5C9EBCF20337F982D54A8D4574A474AAD3A2

Wave Atlas v1.6 Release Notes

Tag: wave-atlas-v1.6
Commit: 3f67325
Changes: M15 Mersenne wave atlas + M16 budget model (test-cost bridge).
PDF size: 1407362 bytes
PDF SHA256: 1B8BBA573FC098853EA772E6FC69BD8BD72597835AC9DCB4518DDF47AB758742

Wave Atlas v1.6.1 Release Notes

Tag: wave-atlas-v1.6.1
Commit: b8a7ca7
Changes: M16 illustrative savings table.
PDF size: 1408026 bytes
PDF SHA256: 6EB6DD297248F0411CF7B25AF0A634ACD9ACDFB883E4FCFE741461898195A14E

Wave Atlas v1.7 Release Notes

Tag: wave-atlas-v1.7
Commit: c4bafeb
Changes: M17 two-stage budget model (cheap filter + expensive test).
PDF size: 1523307 bytes
PDF SHA256: 64FFA1C5ED1048051BA8893BAF4730B58CD5D0ADEE058167840E9A2EC5164286

Roadmap update note:
- ROADMAP updated: protocol added, M14b/M17 sections completed.

Wave Atlas v1.8 Release Notes

Tag: wave-atlas-v1.8
Commit: (tag wave-atlas-v1.8)
Changes: M18 spectral density (Q0 vs Q1); track M17 artifacts for reproducible PDF builds.
PDF size: 1593926 bytes
PDF SHA256: BD66FAC6972AFD31234DA38811A527ADB2BC86A84B822561EC38889916085D1A

Wave Atlas v1.9 Release Notes

Tag: wave-atlas-v1.9
Commit: (tag wave-atlas-v1.9)
Changes: M18b scaling/generalization + M19 weight tuning for spectral density.
PDF size: 1701909 bytes
PDF SHA256: 3FAB2A7EF7F86B03F350C82FD5A9D05FD7C8F67BAA6FD0BEC78A520087839DBF

Wave Atlas v1.9.1 Release Notes

Tag: wave-atlas-v1.9.1
Commit: (tag wave-atlas-v1.9.1)
Changes: M19 weight sanity check (q/rand) + M18b baseline-survival note in TeX.
PDF size: 1704048 bytes
PDF SHA256: A413EFA225F1C48605A4446B7E31204CBF24668A72E8C02D64A232CB382A513B

Wave Atlas v1.10 Release Notes

Tag: wave-atlas-v1.10
Commit: (tag wave-atlas-v1.10)
Changes: M20b all-p weight effects (resolve prime-mode degeneracy).
PDF size: 1768925 bytes
PDF SHA256: 429EA91A25C67CFD7177D267976FAF443EC3C72603AC95B52CA38AEC6B0BCB45

Wave Atlas v1.11 Release Notes

Tag: wave-atlas-v1.11
Commit: (tag wave-atlas-v1.11)
Changes: M21 hazard model for prime p (death in Q1\Q0).
PDF size: 1844115 bytes
PDF SHA256: 6ABAD243DF99952B3E6FD0085EEED7773F1A85095186A4B392E25A2B712E6878

Wave Atlas v1.12 Release Notes

Tag: wave-atlas-v1.12
Commit: (tag wave-atlas-v1.12)
Changes: M22 two-stage Mersenne navigator (M18 + M21 queue).
PDF size: 1890696 bytes
PDF SHA256: 2403C530FEEA3AE5F4A17DDFB2043E68A1129F848BC4F7F34AA26910EEC14AF6

Wave Atlas v1.12.1 Release Notes

Tag: wave-atlas-v1.12.1
Commit: (tag wave-atlas-v1.12.1)
Changes: M22 savings table (compute-seconds saved) + M21 interpretation note.
PDF size: 1891374 bytes
PDF SHA256: CFCF54311D1BCA54712DBB18EF8BCF694C575130388A02EAB172764DF708245C

Wave Atlas v1.13 Release Notes

Tag: wave-atlas-v1.13
Commit: (tag wave-atlas-v1.13)
Changes: M23 budgeted search simulation (random vs M18/M21/M22) + savings table.
PDF size: 1988103 bytes
PDF SHA256: 2CDB1BEC86BFA95578EE84726CC4A5FF36547615538F8EFE98AFB7BFA0C82918
