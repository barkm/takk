import importlib.util
import sys
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_serving.py"


def load_script():
    """`scripts/` is not a package, so the script is loaded by path, as a test of a script must."""
    spec = importlib.util.spec_from_file_location("build_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_serving"] = module
    spec.loader.exec_module(module)
    return module


def test_publish_names_the_archive_by_its_digest_and_pins_it(tmp_path, monkeypatch):
    build_serving = load_script()
    bundle_dir = tmp_path / "a_run-a_glossary"
    bundle_dir.mkdir()
    (bundle_dir / "meta.json").write_text('{"run": "a_run"}')
    (bundle_dir / "signs.parquet").write_bytes(b"not really a parquet")
    pin = tmp_path / "deploy" / "bundle.txt"
    uploaded: list[list[str]] = []

    def fake_run(command, **kwargs):
        uploaded.append(command)
        Path(tmp_path / "uploaded.tar.gz").write_bytes(Path(command[3]).read_bytes())

    monkeypatch.setattr(build_serving.subprocess, "run", fake_run)
    name = build_serving.publish(bundle_dir, "gs://a-bucket/serving/", pin=pin)

    assert uploaded == [["gcloud", "storage", "cp", uploaded[0][3], f"gs://a-bucket/serving/{name}"]]
    written, digest = pin.read_text().split()
    assert written == name and name.endswith(".tar.gz") and digest.removeprefix("sha256:")[:12] in name
    # the archive holds the bundle's files by their own names, so it unpacks into a bundle directory
    with tarfile.open(tmp_path / "uploaded.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["meta.json", "signs.parquet"]
