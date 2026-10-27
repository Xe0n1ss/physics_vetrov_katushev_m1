import base64
from dataclasses import replace
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs
import webbrowser

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from src.analytical import small_angle_damped_period, small_angle_period
from src.experiments import ExperimentConfig, ExperimentResult, run_experiment


PHYSICAL_FIELDS = (
    ("rod_length", "Длина стержня", "м"),
    ("rod_mass", "Масса стержня", "кг"),
    ("sphere_radius", "Радиус шара", "м"),
    ("sphere_mass", "Масса шара", "кг"),
    ("damping", "Коэффициент трения", "Н·м·с/рад"),
)

INITIAL_VALUES = {
    "rod_length": "0.8",
    "rod_mass": "0.2",
    "sphere_radius": "0.08",
    "sphere_mass": "0.5",
    "damping": "0.015",
}


def _field_markup(name: str, label: str, unit: str, values: dict[str, str]) -> str:
    value = escape(values.get(name, ""), quote=True)
    return (
        f'<label class="field" for="{name}">'
        f'<span>{escape(label)}</span>'
        f'<input id="{name}" name="{name}" value="{value}" required '
        'type="text" inputmode="decimal" autocomplete="off">'
        f'<small>{escape(unit)}</small>'
        "</label>"
    )


def _figure_data(figure: Figure) -> str:
    FigureCanvasAgg(figure)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=125)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _plots(result: ExperimentResult) -> str:
    config = result.config
    base = config.parameters

    period_figure = Figure(figsize=(7.5, 3.5), layout="constrained")
    period_ax = period_figure.subplots()
    period_ax.plot(config.amplitudes, result.numerical_periods, "o-", label="Численно")
    period_ax.plot(config.amplitudes, result.theoretical_periods, "--", label="Точная формула")
    period_ax.axhline(small_angle_period(base), color="gray", linestyle=":", label="Малые углы")
    period_ax.set(xlabel="Начальная амплитуда, градусы", ylabel="Период, с")
    period_ax.grid(True)
    period_ax.legend()

    angle_figure = Figure(figsize=(7.5, 3.5), layout="constrained")
    angle_ax = angle_figure.subplots()
    angle_ax.plot(result.ideal.time, np.rad2deg(result.ideal.angle), label="Без трения")
    angle_ax.plot(result.friction.time, np.rad2deg(result.friction.angle), label="С трением")
    angle_ax.set(xlabel="Время, с", ylabel="Угол, градусы")
    angle_ax.grid(True)
    angle_ax.legend()

    energy_figure = Figure(figsize=(7.5, 3.5), layout="constrained")
    energy_ax = energy_figure.subplots()
    energy_ax.plot(result.ideal.time, result.ideal.energy, label="Без трения")
    energy_ax.plot(result.friction.time, result.friction.energy, label="С трением")
    energy_ax.set(xlabel="Время, с", ylabel="Энергия, Дж")
    energy_ax.grid(True)
    energy_ax.legend()

    phase_figure = Figure(figsize=(9, 4), layout="constrained")
    phase_axes = phase_figure.subplots(1, 2, sharex=True, sharey=True)
    angle_limit = float(np.max(np.abs(np.rad2deg(result.ideal.angle)))) * 1.1
    velocity_limit = float(np.max(np.abs(result.ideal.angular_velocity))) * 1.1
    for ax, trajectory, title, color in (
        (phase_axes[0], result.ideal, "Без трения", "#2459c7"),
        (phase_axes[1], result.friction, "С трением", "#d46a24"),
    ):
        angles = np.rad2deg(trajectory.angle)
        ax.plot(angles, trajectory.angular_velocity, color=color)
        ax.scatter(angles[0], trajectory.angular_velocity[0], color=color, marker="o", label="Старт")
        ax.scatter(angles[-1], trajectory.angular_velocity[-1], color=color, marker="x", label="Конец")
        ax.set(
            title=title,
            xlabel="Угол, градусы",
            xlim=(-angle_limit, angle_limit),
            ylim=(-velocity_limit, velocity_limit),
        )
        ax.grid(True)
        ax.legend()
    phase_axes[0].set_ylabel("Угловая скорость, рад/с")

    return "".join(
        f'<div class="chart"><h3>{title}</h3>'
        f'<img alt="{title}" src="data:image/png;base64,{_figure_data(figure)}"></div>'
        for title, figure in (
            ("Период и амплитуда", period_figure),
            ("Угол во времени", angle_figure),
            ("Энергия во времени", energy_figure),
            ("Фазовый портрет: угол и угловая скорость", phase_figure),
        )
    )


def _result_markup(result: ExperimentResult) -> str:
    config = result.config
    base = config.parameters
    theoretical_damped = small_angle_damped_period(replace(base, damping=config.damping))
    rows = "".join(
        "<tr>"
        f"<td>{amplitude:g}°</td>"
        f"<td>{numerical:.6f} с</td>"
        f"<td>{theoretical:.6f} с</td>"
        f"<td>{abs(numerical / theoretical - 1):.2e}</td>"
        "</tr>"
        for amplitude, numerical, theoretical in zip(
            config.amplitudes,
            result.numerical_periods,
            result.theoretical_periods,
            strict=True,
        )
    )
    energy_loss = result.friction.energy[0] - result.friction.energy[-1]
    return (
        '<section class="card results"><h2>Результаты</h2>'
        '<div class="metrics">'
        f'<div><span>Центр масс</span><strong>{base.center_of_mass_distance:.6f} м</strong></div>'
        f'<div><span>Момент инерции</span><strong>{base.moment_of_inertia:.6f} кг·м²</strong></div>'
        f'<div><span>Период при малом угле</span><strong>{small_angle_period(base):.6f} с</strong></div>'
        f'<div><span>Время расчёта</span><strong>{config.duration:.2f} с</strong></div>'
        '</div><h3>Без трения: период и сохранение энергии</h3>'
        '<div class="table-wrap"><table><thead><tr><th>Амплитуда</th>'
        '<th>Численно</th><th>Теория</th><th>Расхождение</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        f'<p>При {config.comparison_angle:g}° относительный дрейф энергии без трения: '
        f'<strong>{result.ideal.relative_energy_drift:.2e}</strong>.</p>'
        '<h3>С трением: затухание и потеря энергии</h3>'
        f'<p>Первый период с трением при {config.comparison_angle:g}°: '
        f'<strong>{result.friction.period:.6f} с</strong>.</p>'
        f'<p>При малом угле {config.small_angle:g}° период с трением: '
        f'<strong>{result.small_friction.period:.6f} с</strong> численно и '
        f'<strong>{theoretical_damped:.6f} с</strong> по линейной теории.</p>'
        f'<p>Потеря энергии с трением: <strong>{energy_loss:.6f} Дж</strong> '
        f'по траектории и <strong>{result.dissipated_energy:.6f} Дж</strong> '
        'по интегралу bω².</p>'
        '<p>На фазовом портрете каждая точка построена из рассчитанных угла и угловой '
        'скорости. Без трения кривая замкнута, с трением она сужается к равновесию.</p>'
        f'{_plots(result)}</section>'
    )


def render_page(
    submitted: dict[str, str] | None = None,
    error: str | None = None,
    result: ExperimentResult | None = None,
) -> str:
    values = {**INITIAL_VALUES, **(submitted or {})}
    physical = "".join(_field_markup(*field, values) for field in PHYSICAL_FIELDS)
    error_markup = f'<div class="error" role="alert">{escape(error)}</div>' if error else ""
    result_markup = _result_markup(result) if result else ""
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Физический маятник</title>
<style>
body {{ margin: 0; background: #f4f6fb; color: #172037; font: 16px system-ui, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 80px; }}
h1 {{ margin-bottom: 6px; }}
p.intro {{ color: #526078; margin: 0 0 24px; }}
.card {{ background: white; border: 1px solid #e0e5ef; border-radius: 16px; padding: 24px; margin: 20px 0; box-shadow: 0 8px 28px #1a26400d; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
.field {{ display: flex; flex-direction: column; gap: 6px; font-weight: 600; }}
.field input {{ font: inherit; padding: 10px 12px; border: 1px solid #bbc5d5; border-radius: 8px; }}
.field input:focus {{ outline: 2px solid #4b75db; border-color: transparent; }}
.field small {{ color: #64728a; font-weight: 400; }}
button {{ background: #2459c7; color: white; border: 0; border-radius: 9px; padding: 12px 20px; font: 600 16px system-ui; cursor: pointer; margin-top: 24px; }}
button:hover {{ background: #1745a5; }}
.error {{ background: #fff0f0; color: #9a1720; border: 1px solid #e5a5aa; border-radius: 9px; padding: 14px; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.metrics div {{ background: #eef3fc; border-radius: 9px; padding: 12px 16px; display: flex; flex-direction: column; gap: 4px; }}
.metrics span {{ color: #526078; font-size: 13px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e2e7f0; padding: 9px 12px; text-align: left; }}
.chart {{ margin-top: 24px; }}
.chart img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body><main>
<h1>Физический маятник</h1>
<p class="intro">Измените параметры тела и запустите полный набор экспериментов по заданию М5.</p>
{error_markup}
<section class="card"><form method="post" action="/run">
<h2>Параметры маятника</h2><div class="grid">{physical}</div>
<p>Программа автоматически исследует периоды при 5°, 15°, 30°, 60°, 90°, 120° и 150°, сохранение энергии без трения и затухание с трением. Для проверки теории малых углов используется 1°. Ускорение свободного падения — 9,81 м/с².</p>
<p>Для исследования малого трения допускается коэффициент не больше 10% от порога исчезновения колебаний. Допустимое значение зависит от размеров и масс введённого тела; если его превысить, форма покажет рассчитанный предел.</p>
<button type="submit">Рассчитать</button>
</form></section>
{result_markup}
</main></body></html>"""


class PendulumHandler(BaseHTTPRequestHandler):
    def _send_page(self, page: str, status: int = 200) -> None:
        body = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        self._send_page(render_page())

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < length <= 20000:
            self.send_error(413)
            return
        fields: dict[str, str] = {}
        try:
            payload = self.rfile.read(length).decode("utf-8")
            fields = {
                key: values[0]
                for key, values in parse_qs(payload, keep_blank_values=True).items()
            }
            config = ExperimentConfig.from_fields(fields)
            result = run_experiment(config)
            self._send_page(render_page(fields, result=result))
        except (UnicodeDecodeError, ValueError, RuntimeError) as error:
            self._send_page(render_page(fields, error=str(error)))

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    with ThreadingHTTPServer(("127.0.0.1", 0), PendulumHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Откройте {url}", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
