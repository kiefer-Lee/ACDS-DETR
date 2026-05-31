from pathlib import Path


def test_comparison_configs_and_models_compile():
    root = Path(__file__).resolve().parents[1]
    for folder in ("configs", "models"):
        for path in (root / folder).glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
