import importlib
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_the_isv_package_does_not_depend_on_the_app():
    """The TAKK app (`takk`) is built on the ISV work (`isolated_sign_validation`), never the reverse:
    the model, the data pipeline and the evaluation have to stand on their own."""
    imports = re.compile(r"^\s*(?:from|import) takk\b", re.M)  # a mention in a docstring is fine
    offenders = [path for path in (SRC / "isolated_sign_validation").rglob("*.py") if imports.search(path.read_text())]
    assert not offenders, [str(path) for path in offenders]


def test_the_app_does_not_import_the_extraction_dependencies():
    """The practice app takes landmarks from a browser and extracts none, so its image carries
    neither MediaPipe nor OpenCV (nor matplotlib or yt-dlp, which only the data work needs). They are
    installed here, so the check is that importing the app never reaches them."""
    import builtins

    blocked = {"mediapipe", "cv2", "matplotlib", "yt_dlp"}
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"the app imported {name}")
        return real(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        importlib.reload(importlib.import_module("takk.main"))
    finally:
        builtins.__import__ = real
