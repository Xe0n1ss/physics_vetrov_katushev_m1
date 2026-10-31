from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlencode

import pytest

from src.experiments import LABORATORY_LIMITS, ExperimentConfig, run_experiment
from src.gui import PendulumHandler, render_page


@pytest.fixture
def fields() -> dict[str, str]:
    return {
        "rod_length": "0.8",
        "rod_mass": "0.2",
        "sphere_radius": "0.08",
        "sphere_mass": "0.5",
        "damping": "0.015",
    }


def test_physical_parameters_are_editable(fields: dict[str, str]) -> None:
    first = ExperimentConfig.from_fields(fields)
    assert first.amplitudes == (5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    assert first.parameters.gravity == 9.81
    changed = {**fields, "rod_length": "1.2", "sphere_mass": "0.7"}
    second = ExperimentConfig.from_fields(changed)
    assert first.parameters.moment_of_inertia != second.parameters.moment_of_inertia
    assert first.duration != second.duration


def test_gravity_and_experiment_design_are_not_form_inputs(fields: dict[str, str]) -> None:
    changed = {**fields, "gravity": "100", "amplitudes": "1", "duration": "0"}
    config = ExperimentConfig.from_fields(changed)
    assert config.parameters.gravity == 9.81
    assert len(config.amplitudes) == 7
    assert config.duration > 0


def test_friction_within_small_damping_limit_is_allowed(fields: dict[str, str]) -> None:
    config = ExperimentConfig.from_fields({**fields, "damping": "0.2"})
    assert config.damping == 0.2


def test_friction_limit_depends_on_body(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.from_fields({**fields, "damping": "0.5"})
    heavier = {**fields, "rod_mass": "2", "sphere_mass": "5", "damping": "0.5"}
    assert ExperimentConfig.from_fields(heavier).damping == 0.5


@pytest.mark.parametrize("name", LABORATORY_LIMITS)
def test_laboratory_upper_limits(fields: dict[str, str], name: str) -> None:
    maximum = LABORATORY_LIMITS[name]
    accepted = ExperimentConfig.from_fields({**fields, name: str(maximum)})
    assert getattr(accepted.parameters, name) == maximum
    with pytest.raises(ValueError, match="лабораторной установки"):
        ExperimentConfig.from_fields({**fields, name: str(maximum * 1.01)})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rod_length", "0"),
        ("rod_mass", "-1"),
        ("sphere_radius", "nan"),
        ("sphere_mass", "inf"),
        ("rod_length", "1e308"),
        ("damping", "0"),
        ("damping", "10"),
        ("damping", "0.3"),
        ("damping", "3"),
    ],
)
def test_invalid_fields_are_rejected(
    fields: dict[str, str], name: str, value: str
) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.from_fields({**fields, name: value})


def test_result_and_browser_page(fields: dict[str, str]) -> None:
    config = ExperimentConfig.from_fields(fields)
    result = run_experiment(config)
    assert len(result.numerical_periods) == 7
    assert result.numerical_periods[0] < result.numerical_periods[1]
    assert result.ideal.relative_energy_drift < 1e-8
    page = render_page(fields, result=result)
    assert "Результаты" in page
    assert page.count("data:image/png;base64,") == 4
    assert "Фазовый портрет" in page
    assert "0.8" in page
    assert "не больше 3 м" in page


def test_local_web_interface_accepts_form(fields: dict[str, str]) -> None:
    with ThreadingHTTPServer(("127.0.0.1", 0), PendulumHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=15)
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert "Рассчитать" in response.read().decode("utf-8")

        connection.request(
            "POST",
            "/run",
            body=urlencode({**fields, "rod_mass": "-1"}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert "нужно положительное" in response.read().decode("utf-8")

        connection.request(
            "POST",
            "/run",
            body=urlencode({**fields, "rod_length": "4"}),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert "лабораторной установки" in response.read().decode("utf-8")

        connection.request(
            "POST",
            "/run",
            body=urlencode(fields),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        assert response.status == 200
        page = response.read().decode("utf-8")
        assert "Результаты" in page
        assert page.count("data:image/png;base64,") == 4
        connection.close()
        server.shutdown()
        thread.join()
