Roadmap (Wave Atlas)

P0 — Repro/CI polish

## M10 — Repro/CI polish
**Цель:** воспроизводимость ключевых фигур/таблиц и сборки PDF.
**Гипотеза:** единая команда сборки снижает риск рассинхронизаций артефактов.
**DoD (артефакты):**
- единая команда `make_wave_atlas.sh` или `python -m code.scripts.build_wave_atlas_all`
- README: как собрать все артефакты и PDF
- контрольный PDF SHA256 зафиксирован в release notes
**Команды:**
```bash
cd docs && lualatex wave_atlas.tex && lualatex wave_atlas.tex
```
**Риски/заметки:** зависимость от локальной среды (MiKTeX/Perl), возможные drift в зависимостях.

P1 — Слои сита как “физика волн”

## M11 — Multi-wheel survey (m=8..14)
**Цель:** подтвердить, что главный слой соответствует первому невыбитому простому, и измерить затухание энергии по слоям для разных m.
**Гипотеза:** пики и спад энергии следуют p0, а не случайной периодике.
**DoD (артефакты):**
- out/wave_atlas/m11/*.png
- out/wave_atlas/m11/*.csv
- wave_atlas.tex: таблица M11 + рисунок M11
**Команды:**
```bash
# шаблон: для каждого m запуск detrend + sequential conditioning
PYTHONPATH=code python code/scripts/twin_t_axis_detrend.py --wheel-csv ... --B ... --t-max ... --cond-primes ... --max-layer 3 --out-dir out/wave_atlas/m11/mX
```
**Риски/заметки:** масштаб t_max, размер wheel-скана, compute budget.

## M12 — Residual-as-process (после 3 слоёв)
**Цель:** описать остаток после снятия детерминированных слоёв.
**Гипотеза:** остаток близок к шумовому/разреженному процессу без сильных периодик.
**DoD (артефакты):**
- out/wave_atlas/m12/residual_gaps.png
- out/wave_atlas/m12/residual_acf.png
- out/wave_atlas/m12/residual_summary.json
- wave_atlas.tex: 1 страница с выводом
**Команды:**
```bash
# анализ residue-процесса после conditioning
```
**Риски/заметки:** интерпретация без “доказательных” заявлений.

P2 — Мост wave ↔ GT

## M13 — GT wheel-lattice metrics расширение
**Цель:** найти GT-метрики, устойчиво реагирующие на слои sieve на wheel-решётке.
**Гипотеза:** часть граф-метрик (компоненты/кластеры/спектр) отражает слои лучше, чем gap/entropy.
**DoD (артефакты):**
- таблица “метрика → чувствительность к layer/mod”
- 2–3 фигуры с устойчивым эффектом
- wave_atlas.tex: подпункт M13
**Команды:**
```bash
# сканы по метрикам и eps
```
**Риски/заметки:** выбор метрик и параметров, вычислительная нагрузка.

P3 — Прикладная выгода: ускорение

## M14 — Candidate generator (wave-sieve accelerator)
Status: Done (tag wave-atlas-v1.5)
**Цель:** использовать слои как генератор кандидатов (t=r+Lq).
**Гипотеза:** throughput растёт, survival rate падает предсказуемо с числом слоёв.
**DoD (артефакты):**
- график throughput vs layers
- график survival rate vs layers
- CLI `generate_candidates.py --B ... --layers ... --count ...`
- wave_atlas.tex: 1 страница
**Команды:**
```bash
# генерация кандидатов по разрешённым residue классам
```
**Риски/заметки:** PRP-тестирование и лимиты вычислений.

P4 — Отдельная лаборатория волн

## M15 — Mersenne wave atlas (опционально)
Status: Done (tag wave-atlas-v1.6)
**Цель:** построить атлас волн для 2^p-1 через ord_q(2) и conditioning.
**Гипотеза:** периодики по ord_q(2) дают “идеальные” волны.
**DoD (артефакты):**
- heatmap p×q (делимость 2^p-1)
- таблица ord_q(2)
- отдельный раздел/appendix в PDF
**Команды:**
```bash
# генерация Mersenne heatmap
```
**Риски/заметки:** размер матриц, выбор диапазонов p и q.

## M16 — Бюджетная модель / стоимость проверки
Status: Done (tag wave-atlas-v1.6.1)
**Цель:** связать survival/throughput со стоимостью теста кандидата и показать,
когда дополнительные слои сита экономически выгодны.
**DoD:**
- out/wave_atlas/m16/m16_total_cost_vs_layers.png
- out/wave_atlas/m16/m16_time_saved_vs_layers.png
- out/wave_atlas/m16/m16_break_even.csv + m16_break_even_table.tex
- wave_atlas.tex: раздел M16 с 2 графиками и таблицей
