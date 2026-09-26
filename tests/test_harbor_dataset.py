import tomllib

from touchstone.harbor.dataset import SUBDIRS, Dataset


def test_write_creates_layout_and_manifest(tmp_path):
    ds = Dataset(name="acme/support", description="support tasks", keywords=["b", "a"])
    root = ds.write(tmp_path / "touchstone")
    for sub in SUBDIRS:
        assert (root / sub).is_dir()
    assert (root / "report.md").read_text().startswith("# acme/support")

    manifest = tomllib.loads((root / "dataset.toml").read_text())
    assert manifest["dataset"]["name"] == "acme/support"
    assert manifest["dataset"]["keywords"] == ["a", "b"]  # sorted, stable
    assert manifest["tasks"] == []


def test_write_is_idempotent_and_keeps_report(tmp_path):
    root = tmp_path / "touchstone"
    Dataset(name="acme/support").write(root)
    (root / "report.md").write_text("# edited\n")
    Dataset(name="acme/support").write(root)  # re-run
    assert (root / "report.md").read_text() == "# edited\n"  # not clobbered


def test_read_roundtrips(tmp_path):
    Dataset(name="acme/support", version="2.0.0", keywords=["x"]).write(tmp_path)
    ds = Dataset.read(tmp_path)
    assert ds.name == "acme/support" and ds.version == "2.0.0" and ds.keywords == ["x"]


def test_survey_id_roundtrips_via_metadata_table(tmp_path):
    Dataset(name="acme/support", survey_id="abc123").write(tmp_path)
    manifest = tomllib.loads((tmp_path / "dataset.toml").read_text())
    assert manifest["metadata"]["touchstone"]["survey_id"] == "abc123"
    assert Dataset.read(tmp_path).survey_id == "abc123"


def test_no_metadata_table_written_without_a_survey_id(tmp_path):
    Dataset(name="acme/support").write(tmp_path)
    manifest = tomllib.loads((tmp_path / "dataset.toml").read_text())
    assert "metadata" not in manifest and Dataset.read(tmp_path).survey_id == ""


def test_task_dirs_lists_only_task_dirs_sorted(tmp_path):
    ds = Dataset(name="acme/support")
    ds.write(tmp_path)
    for name in ("b-task", "a-task"):
        d = tmp_path / "tasks" / name
        d.mkdir()
        (d / "task.toml").write_text("")
    (tmp_path / "tasks" / "not-a-task").mkdir()  # no task.toml -> ignored
    assert [d.name for d in ds.task_dirs()] == ["a-task", "b-task"]
