# physics_vetrov_katushev_m1

Выполнили Ветров И. В. (отвечал за код движка) и Катушев И. А. (отвечал за эксперименты, графики, отчет)
Численное моделирование полёта камня, брошенного под углом к горизонту: без
сопротивления, с линейным и с квадратичным сопротивлением воздуха.

[Отчёт](docs/report.md): уравнения, сверка с аналитикой, результаты.

## Структура

```
src/
  models.py        параметры броска, модели сопротивления, уравнения движения
  solver.py        численное решение (solve_ivp)
  analytical.py    аналитические решения для вакуума и линейной модели
  main.py          базовый бросок в трёх моделях
  sweeps.py        серии расчётов по углу, скорости и сопротивлению
  plotting.py      графики
  experiments.py   все эксперименты: графики в docs/plots/ и таблицы
tests/
docs/report.md
```

## Запуск

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt

python -m src.main
python -m src.experiments
python -m pytest
```
