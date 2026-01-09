Roadmap (Wave Atlas)

## Протокол документации (обязательный)

После каждого завершённого шага Mx:

1) В `docs/ROADMAP.md`:
   - пометить `Status: Done (tag ...)`
   - дописать/обновить: **Цель**, **DoD**, **Команды**, **Риски/заметки**
2) В `docs/WAVE_ATLAS_RELEASE_NOTES.md`:
   - добавить строку "что добавили" + SHA/size PDF (если менялся)
3) В `docs/wave_atlas.tex`:
   - добавить/обновить подпункт Mx + `\includegraphics`/таблицу/1 абзац вывода
4) В `out/wave_atlas/mx/`:
   - должны лежать: минимум 1 PNG + 1 CSV/JSON (если есть метрики) + (опц.) `.tex` таблица
5) Команда воспроизводимости:
   - в ROADMAP должна быть **одна** команда запуска (копипаст-готовая).

P0 - Repro/CI polish

## M10 - Repro/CI polish
**Цель:** чистые, воспроизводимые сборки PDF.
**DoD:**
- скрипты сборки `make_wave_atlas.sh` и `python -m code.scripts.build_wave_atlas_all`
- README с чёткой инструкцией сборки
- фиксация SHA256 PDF в release notes
**Команды:**
```bash
cd docs && lualatex wave_atlas.tex && lualatex wave_atlas.tex
```

P1 - Выбор базы и осей

## M11 - Multi-wheel survey (m=8..14)
Status: Done (tag wave-atlas-v1.2)

**Цель:** сравнить спектры/метрики по колёсам и выбрать p0/модель.
**DoD:**
- out/wave_atlas/m11/*.png
- out/wave_atlas/m11/*.csv
- wave_atlas.tex: раздел M11

## M12 - Residual-as-process (снятие 3 слоёв)
Status: Done (tag wave-atlas-v1.2)

**Цель:** оценить остатки как процесс после снятия слоёв.
**DoD:**
- out/wave_atlas/m12/residual_gaps.png
- out/wave_atlas/m12/residual_acf.png
- out/wave_atlas/m12/residual_summary.json
- wave_atlas.tex: раздел M12

P2 - GT-метрики по базовой решётке

## M13 - GT wheel-lattice metrics
Status: Done (tag wave-atlas-v1.3)

**Цель:** найти граф-метрики на wheel-решётке, чувствительные к слоям.
**DoD:**
- сравнение метрик по слоям/mod в CSV/PNG
- 2-3 визуализации устойчивости
- wave_atlas.tex: раздел M13

P3 - Генератор кандидатов

## M14 - Candidate generator (wave-sieve accelerator)
Status: Done (tag wave-atlas-v1.4)

**Цель:** построить генератор кандидатов t=r+Lq по слоям.
**DoD:**
- survival/throughput vs layers
- CLI генератора
- wave_atlas.tex: раздел M14

## M14b - Segmented deep layers (ускоритель до 150 слоёв)
Status: Done (tag wave-atlas-v1.5)

**Цель:** поднять глубину слоёв (десятки-сотни простых) без взрыва модуля/CRT,
оценить survival/throughput на сегментном приближении.

**Гипотеза:** даже при падении throughput, любое снижение survival выгодно при дорогом тесте (LLR/PRP).

**DoD (артефакты):**
- out/wave_atlas/m14b/m14b_survival_vs_layers.png
- out/wave_atlas/m14b/m14b_throughput_vs_layers.png
- out/wave_atlas/m14b/m14b_candidate_gap_hist.png
- out/wave_atlas/m14b/m14b_summary.csv + m14b_summary.json
- wave_atlas.tex: раздел M14b + таблица эффективности (если включена)

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m14b_segmented_layers.py \
  --B 27720 \
  --layer-count 150 \
  --segment-len 100000 \
  --segments 6 \
  --seed 123 \
  --out-dir out/wave_atlas/m14b
```

**Риски/заметки:** это оценка throughput/survival по случайным сегментам t, а не полный CRT-генератор.

P4 - Связь с "инженерной" стоимостью

## M15 - Mersenne wave atlas
Status: Done (tag wave-atlas-v1.6)

**Цель:** показать волны на "чистой" системе, где порядок управляет делимостью.
**DoD:**
- heatmap p x q по делимости 2^p-1
- гистограмма ord_q(2)
- wave_atlas.tex: раздел M15

## M16 - Бюджетная модель / cost-of-testing bridge
Status: Done (tag wave-atlas-v1.6.1)

**Цель:** связать survival/throughput с реальной стоимостью теста кандидатов.
**DoD:**
- out/wave_atlas/m16/m16_total_cost_vs_layers.png
- out/wave_atlas/m16/m16_time_saved_vs_layers.png
- out/wave_atlas/m16/m16_break_even.csv + m16_break_even_table.tex
- wave_atlas.tex: раздел M16 с 2 рисунками и таблицей

## M17 - Двухступенчатая бюджетная модель (cheap + expensive test)
Status: Done (tag wave-atlas-v1.7)

**Цель:** привязать выгоду от слоёв к реалистичному пайплайну:
дешёвый фильтр (c1) -> редкие кандидаты на дорогой тест (c2) с pass-rate r1.

**Гипотеза:** при доминирующем дорогом тесте оптимум уходит в глубокие слои;
при очень малом r1 или дешёвом c2 оптимум может быть на средних слоях.

**DoD (артефакты):**
- out/wave_atlas/m17/m17_total_cost_vs_layers.png
- out/wave_atlas/m17/m17_time_saved_vs_layers.png
- out/wave_atlas/m17/m17_optimal_layers_heatmap.png
- out/wave_atlas/m17/m17_break_even.csv + m17_break_even_table.tex
- out/wave_atlas/m17/m17_scenarios.csv + m17_scenarios_table.tex
- wave_atlas.tex: раздел M17

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m17_two_stage_budget.py \
  --m14b-summary out/wave_atlas/m14b/m14b_summary.csv \
  --layers-points 1,6,24,60,100,150 \
  --N-raw 1e6 \
  --N-raw-big 1e9 \
  --workers 1,1024,10000 \
  --c1-list 1e-6,1e-4,1e-3,1e-2 \
  --c2-list 1,60,3600,86400 \
  --r1-list 1e-2,1e-4,1e-6 \
  --out-dir out/wave_atlas/m17
```

**Риски/заметки:** модель агрегированная; значения c1/c2/r1 задаются сценарно и должны интерпретироваться как "compute-seconds".

P5 - Вероятностная навигация по экспонентам Мерсенна

## M18 - Mersenne spectral density (вероятностная навигация по p)
Status: Done (tag wave-atlas-v1.8)

**Цель:** ранжировать экспоненты p по плотности "убийственных" резонансов
от малых простых, чтобы дорогие тесты запускались на "тихих" p.

**Гипотеза:** score, построенный по Q0, должен обогащать выживаемость
против более глубокого Q1 (enrichment@k, AUC).

**DoD (артефакты):**
- out/wave_atlas/m18/m18_score.csv
- out/wave_atlas/m18/m18_top_periods.csv + m18_top_periods.json
- out/wave_atlas/m18/m18_density_heatmap.png
- out/wave_atlas/m18/m18_enrichment_curve.png
- out/wave_atlas/m18/m18_score_vs_survival.png
- out/wave_atlas/m18/m18_summary.json
- wave_atlas.tex: раздел M18

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m18_mersenne_spectral_density.py \
  --p-min 2 \
  --p-max 20000 \
  --p-mode prime \
  --Q0 50000 \
  --Q1 200000 \
  --weight inv_q \
  --smooth-window 101 \
  --out-dir out/wave_atlas/m18
```

**Риски/заметки:** для оценки не смешивать Q0 (score) и Q1 (survival);
использовать разделение базиса и проверки.

P6 - Масштабирование и тюнинг весов

## M18b - Scaling/generalization (Q0/Q1/p_max)
Status: Done (tag wave-atlas-v1.9)

**Цель:** проверить устойчивость качества ранжирования при росте p_max и
на разных парах (Q0, Q1).

**DoD (артефакты):**
- out/wave_atlas/m18b/m18b_summary.csv + m18b_summary.json
- out/wave_atlas/m18b/m18b_auc_vs_pmax.png
- out/wave_atlas/m18b/m18b_enrichment_vs_pmax.png
- wave_atlas.tex: раздел M18b

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m18b_mersenne_scaling.py \
  --p-max-list 20000,50000,100000 \
  --p-mode prime \
  --pairs 20000:100000,50000:200000 \
  --weight inv_q \
  --out-dir out/wave_atlas/m18b
```

## M19 - Weight tuning (q-weights vs d-weights)
Status: Done (tag wave-atlas-v1.9)

**Цель:** сравнить схемы весов по q и по d(q), чтобы увеличить enrichment@10%.

**DoD (артефакты):**
- out/wave_atlas/m19/m19_weight_summary.csv + m19_weight_summary.json
- out/wave_atlas/m19/m19_auc_by_weight.png
- out/wave_atlas/m19/m19_enrichment_by_weight.png
- wave_atlas.tex: раздел M19

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m19_weight_tuning.py \
  --p-max 50000 \
  --p-mode prime \
  --Q0 50000 \
  --Q1 200000 \
  --weights ones,inv_q,inv_logq,logq,inv_d,inv_logd,logd,q,rand \
  --seed 123 \
  --out-dir out/wave_atlas/m19
```

**Риски/заметки:** sanity-check показал инвариантность метрик к весам при p-mode=prime,
поскольку для простого p условие d(q)|p сводится к d(q)=p; для тюнинга весов
нужна модификация цели или p-mode=all.

## M20b - Weight effects with all p (remove prime-mode degeneracy)
Status: Done (tag wave-atlas-v1.10)

**Цель:** убрать вырождение p-mode=prime и показать, что веса реально меняют
ранжирование на all/composite subsets.

**DoD (артефакты):**
- out/wave_atlas/m20b/m20b_summary.csv + m20b_summary.json
- out/wave_atlas/m20b/m20b_auc_by_weight_(all|prime|composite).png
- out/wave_atlas/m20b/m20b_enrichment10_by_weight_(all|prime|composite).png
- out/wave_atlas/m20b/m20b_score_vs_survival_bins_(all|prime|composite).png
- out/wave_atlas/m20b/m20b_notes.txt
- wave_atlas.tex: раздел M20b

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m20b_weight_effect_all_p.py \
  --p-max 50000 \
  --Q0 50000 \
  --Q1 200000 \
  --weights ones,inv_q,inv_logq,logq,q,inv_d,inv_logd,logd,rand_pos,rand_sign \
  --seed 123 \
  --out-dir out/wave_atlas/m20b
```

## M21 - Hazard model for prime p (death in Q1 \ Q0)
Status: Done (tag wave-atlas-v1.11)

**Цель:** ранжировать простые p из Z (killed_Q0=0) по риску умереть позже
в Q1\Q0, используя арифметическое давление q ≡ 1 (mod p).

**DoD (артефакты):**
- out/wave_atlas/m21/m21_dataset.csv
- out/wave_atlas/m21/m21_summary.csv + m21_summary.json
- out/wave_atlas/m21/m21_hazard_vs_death_bins.png
- out/wave_atlas/m21/m21_enrichment_curve.png
- out/wave_atlas/m21/m21_feature_importance.png
- out/wave_atlas/m21/m21_scatter_logp_vs_hazard.png
- wave_atlas.tex: раздел M21

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m21_hazard_model.py \
  --p-max 100000 \
  --Q0 50000 \
  --Q1 200000 \
  --out-dir out/wave_atlas/m21
```

## M22 - Two-stage Mersenne navigator (M18 + M21 queue)
Status: Done (tag wave-atlas-v1.12)

**Цель:** построить единую очередь простых p: ранние смерти (Q0) в хвост,
нулевой слой ранжировать по hazard (M21), и оценить экономию тестов.

**DoD (артефакты):**
- out/wave_atlas/m22/m22_queue.csv
- out/wave_atlas/m22/m22_enrichment_curve.png
- out/wave_atlas/m22/m22_tests_avoided.png
- out/wave_atlas/m22/m22_summary.json
- wave_atlas.tex: раздел M22

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m22_two_stage_navigator.py \
  --p-max 100000 \
  --Q0 50000 \
  --Q1 200000 \
  --test-costs 1,3600,86400 \
  --out-dir out/wave_atlas/m22
```

## M23 - Budgeted search simulation (M18 vs M22 vs random)
Status: Done (tag wave-atlas-v1.13)

**Цель:** сравнить стратегии очереди при фиксированном бюджете дорогих тестов,
измерить yield и экономию compute-seconds относительно random.

**DoD (артефакты):**
- out/wave_atlas/m23/m23_summary.csv + m23_summary.json
- out/wave_atlas/m23/m23_yield_vs_budget.png
- out/wave_atlas/m23/m23_bad_tests_avoided_vs_budget.png
- out/wave_atlas/m23/m23_compute_saved_vs_budget_3600.png
- out/wave_atlas/m23/m23_table.tex
- wave_atlas.tex: раздел M23

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m23_budgeted_search_sim.py \
  --p-max 100000 \
  --Q0 50000 \
  --Q1 200000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10 \
  --random-iters 300 \
  --seed 123 \
  --test-costs 1,3600,86400 \
  --out-dir out/wave_atlas/m23
```

## M24 - Scale stress-test (M22 at larger p_max/Q1)
Status: Done (tag wave-atlas-v1.14)

**Цель:** проверить устойчивость M22 при больших $p_{\max}$ и $Q_1$ и при
бюджетах выбора до 50%.

**DoD (артефакты):**
- out/wave_atlas/m24/*/m24_summary.csv + m24_summary.json
- out/wave_atlas/m24/*/m24_yield_vs_budget.png
- out/wave_atlas/m24/*/m24_bad_tests_avoided_vs_budget.png
- out/wave_atlas/m24/*/m24_compute_saved_vs_budget_1s.png
- out/wave_atlas/m24/*/m24_compute_saved_vs_budget_1h.png
- out/wave_atlas/m24/*/m24_compute_saved_vs_budget_1d.png
- out/wave_atlas/m24/*/m24_table.tex
- out/wave_atlas/m24/*/m24_manifest.json
- wave_atlas.tex: раздел M24 (A/B/C)

**Команды:**
```bash
# M24-A
PYTHONPATH=code python code/scripts/m24_scale_stress_test.py \
  --p-max 200000 --Q0 50000 --Q1 500000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 300 --test-costs 1,3600,86400 --seed 123 \
  --label p200k_Q0-50k_Q1-500k \
  --out-dir out/wave_atlas/m24

# M24-B
PYTHONPATH=code python code/scripts/m24_scale_stress_test.py \
  --p-max 200000 --Q0 100000 --Q1 1000000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 300 --test-costs 1,3600,86400 --seed 123 \
  --label p200k_Q0-100k_Q1-1M \
  --out-dir out/wave_atlas/m24

# M24-C
PYTHONPATH=code python code/scripts/m24_scale_stress_test.py \
  --p-max 100000 --Q0 50000 --Q1 200000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 300 --test-costs 1,3600,86400 --seed 123 \
  --label p100k_Q0-50k_Q1-200k \
  --out-dir out/wave_atlas/m24
```

## M25 - Hard-mode stress test (Q1=10M + strict filters)
Status: Done (tag wave-atlas-v1.15)

**Цель:** проверить устойчивость навигатора при Q1=10M и строгих
Mersenne-совместимых фильтрах в hazard-фичах.

**DoD (артефакты):**
- out/wave_atlas/m25/*/m25_summary.csv + m25_summary.json
- out/wave_atlas/m25/*/m25_queue.csv
- out/wave_atlas/m25/*/m25_hardness.json
- out/wave_atlas/m25/*/m25_yield_vs_budget.png
- out/wave_atlas/m25/*/m25_bad_tests_avoided_vs_budget.png
- out/wave_atlas/m25/*/m25_compute_saved_vs_budget_1d.png
- out/wave_atlas/m25/*/m25_hazard_count_hist.png
- out/wave_atlas/m25/*/m25_hazard_vs_survival_bins.png
- out/wave_atlas/m25/*/m25_table.tex
- out/wave_atlas/m25/*/m25_manifest.json
- wave_atlas.tex: раздел M25 (A/B/C)

**Команды:**
```bash
# M25-A (hard-mode)
PYTHONPATH=code python code/scripts/m25_hardmode_stress_test.py \
  --p-max 200000 --Q0 100000 --Q1 10000000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 200 --test-costs 1,3600,86400 --seed 123 \
  --hazard-modes binary,count,harmonic \
  --mersenne-strict 1 \
  --label p200k_Q0-100k_Q1-10M_strict \
  --out-dir out/wave_atlas/m25

# M25-B
PYTHONPATH=code python code/scripts/m25_hardmode_stress_test.py \
  --p-max 300000 --Q0 150000 --Q1 10000000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 200 --test-costs 1,3600,86400 --seed 123 \
  --hazard-modes binary,count,harmonic \
  --mersenne-strict 1 \
  --label p300k_Q0-150k_Q1-10M_strict \
  --out-dir out/wave_atlas/m25

# M25-C (baseline)
PYTHONPATH=code python code/scripts/m25_hardmode_stress_test.py \
  --p-max 200000 --Q0 100000 --Q1 1000000 \
  --budgets 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --random-iters 300 --test-costs 1,3600,86400 --seed 123 \
  --hazard-modes binary,count,harmonic \
  --mersenne-strict 1 \
  --label p200k_Q0-100k_Q1-1M_strict \
  --out-dir out/wave_atlas/m25
```

## M26 - Calibrated survival predictor + extrapolation
Status: Done (tag wave-atlas-v1.16)

**Цель:** калибровать вероятность survival по hazard-фичам и экстраполировать
ожидаемый yield/экономию на больших Q без полного Q1-лейблинга.

**DoD (артефакты):**
- out/wave_atlas/m26/*/m26_dataset.csv + m26_dataset_meta.json
- out/wave_atlas/m26/*/m26_model_summary.json
- out/wave_atlas/m26/*/m26_calibration_fitQ.png
- out/wave_atlas/m26/*/m26_calibration_by_Q.png
- out/wave_atlas/m26/*/m26_auc_by_Q.png
- out/wave_atlas/m26/*/m26_brier_by_Q.png
- out/wave_atlas/m26/*/m26_logloss_by_Q.png
- out/wave_atlas/m26/*/m26_predicted_yield_vs_budget.png
- out/wave_atlas/m26/*/m26_predicted_compute_saved_1d.png
- out/wave_atlas/m26/*/m26_table.tex
- out/wave_atlas/m26/*/m26_manifest.json
- wave_atlas.tex: раздел M26

**Команды:**
```bash
PYTHONPATH=code python code/scripts/m26_survival_dataset.py \
  --p-max 200000 --Q0 100000 \
  --Q-list 1000000,2000000,5000000,10000000 \
  --mersenne-strict 1 --seed 123 \
  --label p200k_Q0-100k_Qs-1-2-5-10M_strict \
  --out-dir out/wave_atlas/m26

PYTHONPATH=code python code/scripts/m26_survival_model.py \
  --dataset-csv out/wave_atlas/m26/p200k_Q0-100k_Qs-1-2-5-10M_strict/m26_dataset.csv \
  --fit-Q 10000000 \
  --eval-Q-list 1000000,2000000,5000000,10000000 \
  --model logit+isotonic --seed 123 \
  --out-dir out/wave_atlas/m26/p200k_Q0-100k_Qs-1-2-5-10M_strict
```
