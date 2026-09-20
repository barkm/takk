from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_the_isv_package_does_not_depend_on_the_app():
    """The TAKK app (`takk`) is built on the ISV work (`isolated_sign_validation`), never the reverse:
    the model, the data pipeline and the evaluation have to stand on their own."""
    offenders = [path for path in (SRC / "isolated_sign_validation").rglob("*.py") if "takk" in path.read_text()]
    assert not offenders, [str(path) for path in offenders]
