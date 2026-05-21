from pathlib import Path


def test_configs_compile():
    root = Path(__file__).resolve().parents[1] / "configs"
    for path in root.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

