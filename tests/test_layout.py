import importlib
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def importers(package: str, of: str) -> list[str]:
    """The modules of `package` that import `of`; a mention in a docstring is fine."""
    imports = re.compile(rf"^\s*(?:from|import) {of}\b", re.M)
    return [str(path) for path in (SRC / package).rglob("*.py") if imports.search(path.read_text())]


def test_the_isv_package_does_not_depend_on_the_app():
    """The TAKK app (`takk`) is built on the ISV work (`isolated_sign_verification`), never the reverse:
    the model, the training and the evaluation have to stand on their own."""
    assert not importers("isolated_sign_verification", "takk")


def test_the_app_uses_the_model_only_through_the_verifier():
    """An app knows as little as possible about the model: `verifier.py` is the one module of the ISV
    package it may import, so a change behind it (the preparation, the embedding, the scoring) never
    reaches the app."""
    imports = re.compile(r"^\s*(?:from|import) (isolated_sign_verification(?:\.\w+)*)", re.M)
    used = {module for path in (SRC / "takk").rglob("*.py") for module in imports.findall(path.read_text())}
    assert used <= {"isolated_sign_verification.verifier"}, used


def test_the_data_package_depends_on_neither_the_model_nor_the_app():
    """The sign data (`sign_data`: the landmark format, the extraction and the dataset adapters) is
    independent of any model and of what it is used for, so ISV and the app both build on it."""
    assert not importers("sign_data", "isolated_sign_verification")
    assert not importers("sign_data", "takk")


def test_the_app_does_not_import_the_extraction_dependencies():
    """The practice app takes landmarks from a browser and extracts none, so its image carries
    neither MediaPipe nor OpenCV (nor matplotlib, which only the data work needs). They are
    installed here, so the check is that importing the app never reaches them."""
    import builtins

    blocked = {"mediapipe", "cv2", "matplotlib"}
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
