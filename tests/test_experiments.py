from src.experiments import EXPECTED_FIGURES, main


def test_main_creates_all_figures(tmp_path, monkeypatch) -> None:
    import src.experiments as experiments

    monkeypatch.setattr(experiments, "PLOTS_DIR", tmp_path)

    main()

    for name in EXPECTED_FIGURES:
        figure_path = tmp_path / name

        assert figure_path.exists(), f"не создан {name}"
        assert figure_path.stat().st_size > 1000, f"пустой {name}"
