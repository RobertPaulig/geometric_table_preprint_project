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
