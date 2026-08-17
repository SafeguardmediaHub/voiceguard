# tests/test_eval_bundle.py
"""Fast tier for the v9h evaluation harness (REMEDIATION_PLAN H1).

This module must NEVER import detector, directly or transitively. tests/conftest.py
skips collection of detector-importing modules under $VOICEGUARD_CI_FAST because the
import loads ~380 MB of weights; if eval_bundle grew a module-level `import detector`,
every test here would silently stop running in CI. The weights-tier counterpart is
tests/test_eval_bundle_weights.py.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import eval_bundle as E                     # noqa: E402


def test_importing_eval_bundle_does_not_import_detector():
    """The constraint that keeps this whole file in the fast tier.

    Checked in a SUBPROCESS on purpose. Asserting `"detector" not in sys.modules`
    in-process only holds when this file runs alone -- in a full-suite run
    test_submodel_health.py imports detector first, so the assertion would be
    measuring test ordering rather than eval_bundle's imports.
    """
    import subprocess
    probe = ("import sys, eval_bundle; "
             "sys.exit(1 if 'detector' in sys.modules else 0)")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "scripts"))
    r = subprocess.run([sys.executable, "-c", probe],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (
        "importing eval_bundle pulled in detector (~380 MB of weights). "
        "tests/conftest.py would then skip this whole module under "
        f"$VOICEGUARD_CI_FAST.\nstderr: {r.stderr[-400:]}")


def test_corpus_paths_are_env_overridable(monkeypatch, tmp_path):
    """The corpora live at different paths on a GPU box (/kaggle/input/..., a
    mounted volume) than on the dev machine. Hardcoding a repo-relative path is
    the D9 mistake -- sweep_cascade._load pinned an absolute Windows ffmpeg path
    and silently failed everywhere else. Re-import to pick up the environment."""
    import importlib
    monkeypatch.setenv("VOICEGUARD_BIAS_DIR", str(tmp_path / "bias"))
    monkeypatch.setenv("VOICEGUARD_STUDIO_DIR", str(tmp_path / "studio"))
    monkeypatch.setenv("VOICEGUARD_STUDIO_FAKE_DIR", str(tmp_path / "sf"))
    monkeypatch.setenv("VOICEGUARD_MODELS_DIR", str(tmp_path / "models"))
    reloaded = importlib.reload(E)
    try:
        assert reloaded.BIAS_DIR == str(tmp_path / "bias")
        assert reloaded.STUDIO_DIR == str(tmp_path / "studio")
        assert reloaded.STUDIO_FAKE_DIR == str(tmp_path / "sf")
        assert all(str(tmp_path / "models") in p for p in reloaded.TRAIN_MANIFESTS)
        assert all(str(tmp_path / "models") in p for p in reloaded.VAL_MANIFESTS)
    finally:
        monkeypatch.undo()
        importlib.reload(E)      # restore defaults for the rest of the module


def test_corpus_paths_default_to_repo_relative(monkeypatch):
    import importlib
    for v in ("VOICEGUARD_BIAS_DIR", "VOICEGUARD_STUDIO_DIR",
              "VOICEGUARD_STUDIO_FAKE_DIR", "VOICEGUARD_MODELS_DIR"):
        monkeypatch.delenv(v, raising=False)
    reloaded = importlib.reload(E)
    try:
        assert reloaded.BIAS_DIR == os.path.join(reloaded.REPO_ROOT, "bias_audit_fakes")
        assert reloaded.STUDIO_DIR == os.path.join(reloaded.REPO_ROOT, "studio_clips")
    finally:
        monkeypatch.undo()
        importlib.reload(E)


def test_constants_match_the_deployed_operating_points():
    # detector.verdict_from_score splits at 0.85/0.55/0.30. The old notebook cell
    # used >=0.5, a threshold the deployed system uses nowhere.
    assert E.OP_LIKELY_FAKE == 0.55
    assert E.OP_REVIEW == 0.30
    assert E.HAUSA_FAKE_LIMIT == 50
    assert E.MAX_DECODE_FAILURE_RATE == 0.20


def test_provenance_error_is_an_eval_error():
    assert issubclass(E.ProvenanceError, E.EvalError)


# ── Manifest builders ────────────────────────────────────────────────────────

def _make_clip(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00")          # content is irrelevant; builders only walk names


def _bias_tree(root, counts):
    """counts: {language: (n_real, n_fake)}"""
    for lang, (n_real, n_fake) in counts.items():
        for i in range(n_real):
            _make_clip(os.path.join(root, lang, "real", f"{lang}_real_{i:03d}.wav"))
        for i in range(n_fake):
            _make_clip(os.path.join(root, lang, "fake", f"{lang}_fake_{i:03d}.wav"))


def test_bias_manifest_has_no_val_fallback_for_english_reals(tmp_path):
    """The original cell filled the empty english/real dir from val_v8_fresh.json --
    the TUNING set. That is leakage of exactly the kind the provenance work exists
    to catch. The builder must have no such fallback: an empty dir yields zero."""
    root = str(tmp_path / "bias")
    _bias_tree(root, {"english": (0, 50), "arabic": (50, 50)})
    entries = E.build_bias_audit_manifest(root)
    english = [e for e in entries if e["language"] == "english"]
    assert [e for e in english if e["label"] == 0] == []
    assert len([e for e in english if e["label"] == 1]) == 50


def test_bias_manifest_caps_hausa_at_first_50_fakes_alphabetically(tmp_path):
    root = str(tmp_path / "bias")
    _bias_tree(root, {"hausa": (50, 100)})
    entries = E.build_bias_audit_manifest(root)
    fakes = sorted(os.path.basename(e["path"])
                   for e in entries if e["language"] == "hausa" and e["label"] == 1)
    assert len(fakes) == 50
    assert fakes[0] == "hausa_fake_000.wav"
    assert fakes[-1] == "hausa_fake_049.wav"      # first 50 alphabetically, not random


def test_bias_manifest_reports_languages_with_no_fakes(tmp_path):
    """Igbo and Yoruba have zero fakes on disk, so neither can yield a catch rate or
    an EER. This is REMEDIATION_PLAN H7's 'parity PASS rests on 5 of 7 languages',
    enforced by the tooling instead of asserted in prose."""
    root = str(tmp_path / "bias")
    _bias_tree(root, {"igbo": (50, 0), "yoruba": (49, 0), "arabic": (50, 50)})
    comp = E.manifest_composition(E.build_bias_audit_manifest(root))
    assert comp["igbo"] == {"real": 50, "fake": 0}
    assert comp["yoruba"] == {"real": 49, "fake": 0}
    assert comp["arabic"] == {"real": 50, "fake": 50}


def test_bias_manifest_labels_and_sources(tmp_path):
    root = str(tmp_path / "bias")
    _bias_tree(root, {"french": (2, 3)})
    entries = E.build_bias_audit_manifest(root)
    assert {e["source"] for e in entries} == {"real_french", "fake_french"}
    assert sorted(e["label"] for e in entries) == [0, 0, 1, 1, 1]
    assert all(os.path.isabs(e["path"]) for e in entries)


def test_studio_manifest_is_all_real_and_keeps_subdir_as_language(tmp_path):
    root = str(tmp_path / "studio")
    _make_clip(os.path.join(root, "podcast", "a.mp3"))
    _make_clip(os.path.join(root, "audiobook", "b.wav"))
    _make_clip(os.path.join(root, "_downloads", "c.wav"))
    entries = E.build_studio_manifest(root)
    assert len(entries) == 3                        # _downloads included: keeps the
    assert all(e["label"] == 0 for e in entries)    # published 494 count comparable
    assert {e["language"] for e in entries} == {"podcast", "audiobook", "_downloads"}
    assert {e["source"] for e in entries} == {
        "studio_podcast", "studio_audiobook", "studio__downloads"}


def test_studio_fake_manifest_is_all_fake(tmp_path):
    root = str(tmp_path / "sf")
    _make_clip(os.path.join(root, "studio_edge_en_0006_en-GB-Libby.wav"))
    _make_clip(os.path.join(root, "studio_edge_en_0010_en-GB-Libby.wav"))
    entries = E.build_studio_fake_manifest(root)
    assert len(entries) == 2
    assert all(e["label"] == 1 and e["language"] == "english" for e in entries)


def test_build_manifest_rejects_an_unknown_set_name():
    with pytest.raises(E.EvalError, match="unknown set"):
        E.build_manifest("nope")


def test_build_manifest_raises_when_the_corpus_is_absent(tmp_path):
    with pytest.raises(E.EvalError, match="not found"):
        E.build_bias_audit_manifest(str(tmp_path / "does_not_exist"))


# ── Against the real local corpora (dev machine only; absent in CI/container) ──

@pytest.mark.skipif(not os.path.isdir(E.BIAS_DIR), reason="bias_audit_fakes/ absent")
def test_real_bias_corpus_yields_549_samples():
    """299 real + 250 fake. Guards the headline denominator: if this changes, every
    number produced from this set stops being comparable to the last run."""
    comp = E.manifest_composition(E.build_bias_audit_manifest())
    assert sum(c["real"] for c in comp.values()) == 299
    assert sum(c["fake"] for c in comp.values()) == 250


@pytest.mark.skipif(not os.path.isdir(E.STUDIO_DIR), reason="studio_clips/ absent")
def test_real_studio_corpus_yields_494_clips():
    assert len(E.build_studio_manifest()) == 494


# ── Metrics ──────────────────────────────────────────────────────────────────

def _row(label, score, status="ok", **kw):
    r = {"label": label, "score": score, "status": status, "n_chunks": 1,
         "stage1_chunks": 1, "stage2_chunks": 0, "latency_ms": 100.0}
    r.update(kw)
    return r


def test_eer_is_zero_on_a_perfectly_separable_set():
    scores = [0.01] * 20 + [0.99] * 20
    labels = [0] * 20 + [1] * 20
    assert E.compute_eer(scores, labels) == pytest.approx(0.0, abs=1e-6)


def test_eer_is_about_50_on_an_uninformative_set():
    scores = [0.5] * 40
    labels = [0] * 20 + [1] * 20
    assert E.compute_eer(scores, labels) == pytest.approx(50.0, abs=1.0)


def test_eer_is_none_not_nan_when_a_class_is_missing():
    """An EER over one class is not a small number, it is no number. Emitting nan
    would render as a value in JSON consumers; None renders as null / 'n/a'."""
    assert E.compute_eer([0.1, 0.2, 0.3], [0, 0, 0]) is None
    assert E.compute_eer([], []) is None


def test_fp_and_catch_are_measured_at_the_deployed_operating_points():
    # 4 reals: two would be flagged LIKELY_FAKE (>=0.55), a third only REVIEW (>=0.30)
    # 4 fakes: three caught at >=0.55, all four at >=0.30
    rows = [_row(0, 0.60), _row(0, 0.90), _row(0, 0.40), _row(0, 0.05),
            _row(1, 0.99), _row(1, 0.70), _row(1, 0.56), _row(1, 0.35)]
    m = E.compute_metrics(rows)
    assert m["fp_at_likely_fake"] == pytest.approx(50.0)    # 2 of 4 reals >= 0.55
    assert m["fp_at_review"] == pytest.approx(75.0)         # 3 of 4 reals >= 0.30
    assert m["catch_at_likely_fake"] == pytest.approx(75.0)  # 3 of 4 fakes >= 0.55
    assert m["catch_at_review"] == pytest.approx(100.0)      # 4 of 4 fakes >= 0.30


def test_metrics_exclude_error_rows_but_count_them():
    rows = [_row(0, 0.1), _row(1, 0.9),
            {"label": 1, "status": "error", "error": "ffmpeg error: ..."}]
    m = E.compute_metrics(rows)
    assert m["n_scored"] == 2
    assert m["n_error"] == 1
    assert m["n_real"] == 1 and m["n_fake"] == 1


def test_metrics_report_none_for_fp_when_there_are_no_reals():
    m = E.compute_metrics([_row(1, 0.9), _row(1, 0.8)])
    assert m["fp_at_likely_fake"] is None
    assert m["catch_at_likely_fake"] == pytest.approx(100.0)
    assert m["eer"] is None


def test_metrics_report_stage_routing_and_latency():
    rows = [_row(0, 0.1, n_chunks=4, stage1_chunks=3, stage2_chunks=1, latency_ms=100.0),
            _row(1, 0.9, n_chunks=6, stage1_chunks=6, stage2_chunks=0, latency_ms=300.0)]
    m = E.compute_metrics(rows)
    assert m["stage1_chunks"] == 9 and m["stage2_chunks"] == 1
    assert m["stage1_resolution_pct"] == pytest.approx(90.0)
    assert m["latency_ms_mean"] == pytest.approx(200.0)


def test_fmt_renders_none_as_na():
    assert E.fmt(None) == "n/a"
    assert E.fmt(2.4321) == "2.43"


# ── Leakage scan ─────────────────────────────────────────────────────────────

def _write_manifest(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return path


def test_leakage_finds_a_planted_exact_path_overlap(tmp_path):
    shared = str(tmp_path / "clip_a.wav")
    train = _write_manifest(str(tmp_path / "train.json"),
                            [{"path": shared, "label": 0, "source": "common_voice"}])
    entries = [{"path": shared, "label": 0, "language": "english", "source": "real_english"},
               {"path": str(tmp_path / "clip_b.wav"), "label": 1,
                "language": "english", "source": "fake_english"}]
    r = E.scan_leakage(entries, train_files=[train], val_files=[])
    assert r["exact_path_in_train"] == 1
    assert r["exact_path_in_val"] == 0


def test_leakage_finds_a_basename_overlap_under_a_different_directory(tmp_path):
    val = _write_manifest(str(tmp_path / "val.json"),
                          [{"path": "/elsewhere/clip_a.wav", "label": 0, "source": "cv"}])
    entries = [{"path": str(tmp_path / "here" / "clip_a.wav"), "label": 0,
                "language": "english", "source": "real_english"}]
    r = E.scan_leakage(entries, train_files=[], val_files=[val])
    assert r["exact_path_in_val"] == 0            # different directory
    assert r["basename_in_val"] == {"real": 1, "fake": 0}


def test_leakage_separates_shared_from_held_out_fake_engines(tmp_path):
    train = _write_manifest(str(tmp_path / "train.json"), [
        {"path": "/t/1.wav", "label": 1, "source": "edge_tts_en"},
        {"path": "/t/2.wav", "label": 1, "source": "xtts_v2"},
    ])
    entries = [
        {"path": "/e/a.wav", "label": 1, "language": "english", "source": "edge_tts_en"},
        {"path": "/e/b.wav", "label": 1, "language": "french",  "source": "elevenlabs_fr"},
    ]
    r = E.scan_leakage(entries, train_files=[train], val_files=[])
    assert r["fake_engines_shared"] == ["edge"]
    assert r["fake_engines_held_out"] == ["elevenlabs"]


def test_leakage_records_which_reference_manifests_were_missing(tmp_path):
    present = _write_manifest(str(tmp_path / "train.json"), [])
    missing = str(tmp_path / "nope.json")
    r = E.scan_leakage([], train_files=[present, missing], val_files=[])
    assert present in r["reference_manifests_found"]
    assert missing in r["reference_manifests_missing"]


def test_leakage_accepts_the_teacher_score_dict_shape(tmp_path):
    """teacher_scores_v9_*.json is {path: score}, not a list of dicts."""
    fp = str(tmp_path / "teacher.json")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"/t/clip_a.wav": 0.9}, f)
    entries = [{"path": "/t/clip_a.wav", "label": 1,
                "language": "english", "source": "fake_english"}]
    r = E.scan_leakage(entries, train_files=[fp], val_files=[])
    assert r["exact_path_in_train"] == 1


def test_leakage_skips_content_hashing_unless_deep(tmp_path):
    entries = [{"path": str(tmp_path / "x.wav"), "label": 0,
                "language": "english", "source": "real_english"}]
    assert E.scan_leakage(entries, train_files=[], val_files=[])["content_hash_overlap"] is None


# ── Provenance gate ──────────────────────────────────────────────────────────

class _FakeRegistry:
    """Stands in for bundle_registry.Registry, following the _StubRegistry pattern
    in tests/test_detector.py: __call__ returns self, so it can be substituted for
    detector._Registry (which the real code calls as a constructor)."""
    def __init__(self, problems, files=None):
        self._problems = problems
        self._files = files or {"aasist.pt": "bf6fb21a", "lcnn.pt": "f0f92004"}

    def __call__(self):
        return self

    def integrity_problems(self, version):
        return self._problems

    def get_bundle(self, version):
        return {"version": version, "files": self._files}


class _FakeDetector:
    def __init__(self, active_version="v9h", manifest=None, problems=None, files=None):
        self.ACTIVE_VERSION = active_version
        self._ACTIVE_MANIFEST = {"version": active_version} if manifest is None else manifest
        self._Registry = _FakeRegistry(problems if problems is not None else [], files)


def test_gate_accepts_a_verified_bundle_and_returns_its_artifact_hashes():
    d = _FakeDetector(files={"aasist.pt": "bf6fb21a", "wav2vec.pt": "37c916e8"})
    prov = E.assert_bundle_provenance("v9h", detector_module=d)
    assert prov["bundle"] == "v9h"
    assert prov["artifacts"] == {"aasist.pt": "bf6fb21a", "wav2vec.pt": "37c916e8"}


def test_gate_refuses_when_a_different_bundle_is_loaded():
    """detector resolves the bundle at import; if $VOICEGUARD_FORCE_BUNDLE was not
    set, ACTIVE_VERSION is whatever ACTIVE.json points at."""
    d = _FakeDetector(active_version="v9fixed")
    with pytest.raises(E.ProvenanceError, match="loaded 'v9fixed'"):
        E.assert_bundle_provenance("v9h", detector_module=d)


def test_gate_refuses_the_unverified_legacy_fallback():
    """detector._verify_bundle_before_load returns early with only a WARNING when
    manifest is None (the pre-registry models/ layout). Correct for serving, fatal
    here: it would print 'measured v9h' over numbers made by loose files in models/."""
    d = _FakeDetector(active_version="v9h", manifest=None)
    d._ACTIVE_MANIFEST = None
    with pytest.raises(E.ProvenanceError, match="unverified legacy"):
        E.assert_bundle_provenance("v9h", detector_module=d)


def test_gate_refuses_a_tampered_bundle():
    d = _FakeDetector(problems=["aasist.pt: sha256 deadbeef != registered cafe1234"])
    with pytest.raises(E.ProvenanceError, match="failed integrity"):
        E.assert_bundle_provenance("v9h", detector_module=d)


def test_gate_refuses_an_unregistered_bundle():
    # integrity_problems returns None (not []) for a version absent from the registry.
    d = _FakeDetector(problems=None)
    d._Registry = _FakeRegistry(None)
    with pytest.raises(E.ProvenanceError, match="not registered"):
        E.assert_bundle_provenance("v9h", detector_module=d)


# ── Scoring loop ─────────────────────────────────────────────────────────────

class _FakeScoringDetector:
    """Stands in for detector during scoring. Records how detect() was called."""
    def __init__(self, scores=None, fail_paths=()):
        self.scores = scores or {}
        self.fail_paths = set(fail_paths)
        self.calls = []

    def detect(self, path, audit=True):
        self.calls.append((path, audit))
        if path in self.fail_paths:
            raise RuntimeError(f"Audio loading failed ({os.path.basename(path)}): ffmpeg")
        return {
            "score": self.scores.get(path, 0.5),
            "verdict": "REVIEW",
            "chunks": 3,
            "cascade": {"stage1_chunks": 2, "stage2_chunks": 1,
                        "resolution_pct": 66.7, "band": [0.1, 0.8]},
            "elapsed": 0.25,
        }


def _entries(tmp_path, n, label=0):
    return [{"path": str(tmp_path / f"c{i}.wav"), "label": label,
             "language": "english", "source": "real_english"} for i in range(n)]


def test_scoring_never_writes_to_the_audit_log(tmp_path):
    """H6 makes every live detection append to the tamper-evident chain. 1,093
    evaluation detections must not enter it -- in a forensic export they would be
    indistinguishable from real customer detections."""
    d = _FakeScoringDetector()
    E.score_manifest(_entries(tmp_path, 3), str(tmp_path / "out"), detector_module=d)
    assert d.calls, "detect was never called"
    assert all(audit is False for _path, audit in d.calls)


def test_scoring_writes_rows_incrementally(tmp_path):
    out = str(tmp_path / "out")
    d = _FakeScoringDetector()
    rows = E.score_manifest(_entries(tmp_path, 4), out, detector_module=d)
    on_disk = [json.loads(l) for l in
               open(os.path.join(out, "rows.jsonl"), encoding="utf-8") if l.strip()]
    assert len(rows) == 4 and len(on_disk) == 4
    assert on_disk[0]["stage1_chunks"] == 2 and on_disk[0]["n_chunks"] == 3
    assert on_disk[0]["latency_ms"] == pytest.approx(250.0)


def test_scoring_counts_decode_failures_instead_of_skipping_them(tmp_path):
    """REMEDIATION_PLAN D9: sweep_cascade._load discarded ffmpeg's return code and
    silently skipped every clip, so a total decode failure read as a model collapse.
    A failure must shrink nothing silently."""
    ents = _entries(tmp_path, 5)
    d = _FakeScoringDetector(fail_paths=[ents[1]["path"]])
    rows = E.score_manifest(ents, str(tmp_path / "out"), detector_module=d)
    assert len(rows) == 5
    err = [r for r in rows if r["status"] == "error"]
    assert len(err) == 1
    assert "ffmpeg" in err[0]["error"]
    assert E.compute_metrics(rows)["n_error"] == 1


def test_scoring_aborts_when_most_of_a_set_will_not_decode(tmp_path):
    """Wholesale decode failure is an environment fault, not a result."""
    ents = _entries(tmp_path, 10)
    d = _FakeScoringDetector(fail_paths=[e["path"] for e in ents[:6]])
    with pytest.raises(E.EvalError, match="failed to decode"):
        E.score_manifest(ents, str(tmp_path / "out"), detector_module=d)


def test_resume_skips_paths_already_scored(tmp_path):
    out = str(tmp_path / "out")
    ents = _entries(tmp_path, 4)
    first = _FakeScoringDetector()
    E.score_manifest(ents[:2], out, detector_module=first)
    second = _FakeScoringDetector()
    rows = E.score_manifest(ents, out, detector_module=second, resume=True)
    assert [p for p, _a in second.calls] == [ents[2]["path"], ents[3]["path"]]
    assert len(rows) == 4        # two replayed from disk, two freshly scored


def test_resume_is_off_by_default_and_truncates(tmp_path):
    out = str(tmp_path / "out")
    ents = _entries(tmp_path, 2)
    E.score_manifest(ents, out, detector_module=_FakeScoringDetector())
    E.score_manifest(ents, out, detector_module=_FakeScoringDetector())
    on_disk = [l for l in open(os.path.join(out, "rows.jsonl"), encoding="utf-8") if l.strip()]
    assert len(on_disk) == 2, "a fresh run must not append to the previous run's rows"


# ── CLI / summary ────────────────────────────────────────────────────────────

def test_summary_carries_the_provenance_header_and_leakage_result(tmp_path, monkeypatch):
    root = str(tmp_path / "bias")
    _bias_tree(root, {"french": (2, 2)})
    monkeypatch.setattr(E, "BIAS_DIR", root)
    monkeypatch.setitem(E.MANIFEST_BUILDERS, "bias_audit",
                        lambda: E.build_bias_audit_manifest(root))

    d = _FakeScoringDetector()
    d.ACTIVE_VERSION = "v9h"
    d._ACTIVE_MANIFEST = {"version": "v9h"}
    d._Registry = _FakeRegistry([], {"aasist.pt": "bf6fb21a"})

    summary = E.run_eval("v9h", "bias_audit", out_root=str(tmp_path / "out"),
                         detector_module=d)
    assert summary["provenance"]["artifacts"] == {"aasist.pt": "bf6fb21a"}
    assert "exact_path_in_train" in summary["leakage"]
    assert summary["metrics"]["n_scored"] == 4
    assert summary["composition"]["french"] == {"real": 2, "fake": 2}

    written = json.load(open(os.path.join(str(tmp_path / "out"), "v9h", "bias_audit",
                                          "summary.json"), encoding="utf-8"))
    assert written["provenance"]["bundle"] == "v9h"


def _gated_detector(device=None):
    d = _FakeScoringDetector()
    d.ACTIVE_VERSION = "v9h"
    d._ACTIVE_MANIFEST = {"version": "v9h"}
    d._Registry = _FakeRegistry([], {"aasist.pt": "bf6fb21a"})
    if device is not None:
        d.DEVICE = device
    return d


@pytest.mark.parametrize("device,representative", [("cpu", True), ("cuda:0", False)])
def test_summary_records_the_device_and_flags_gpu_latency(tmp_path, monkeypatch,
                                                          device, representative):
    """Accuracy is device-insensitive; latency is not. Production is a CPU droplet,
    so a cuda run's ms figures must not be publishable as production latency."""
    root = str(tmp_path / "bias")
    _bias_tree(root, {"french": (1, 1)})
    monkeypatch.setitem(E.MANIFEST_BUILDERS, "bias_audit",
                        lambda: E.build_bias_audit_manifest(root))
    summary = E.run_eval("v9h", "bias_audit", out_root=str(tmp_path / "out"),
                         detector_module=_gated_detector(device))
    assert summary["device"] == device
    assert summary["latency_representative"] is representative


def test_per_language_metrics_withhold_eer_where_a_class_is_absent():
    rows = [_row(0, 0.1), _row(1, 0.9), _row(0, 0.2)]
    for r, lang in zip(rows, ["arabic", "arabic", "igbo"]):
        r["language"] = lang
    per = E.per_language_metrics(rows)
    assert per["arabic"]["eer"] is not None
    assert per["igbo"]["eer"] is None          # one class only -> no number, not nan


def test_run_eval_refuses_before_scoring_when_provenance_fails(tmp_path, monkeypatch):
    root = str(tmp_path / "bias")
    _bias_tree(root, {"french": (2, 2)})
    monkeypatch.setitem(E.MANIFEST_BUILDERS, "bias_audit",
                        lambda: E.build_bias_audit_manifest(root))
    d = _FakeScoringDetector()
    d.ACTIVE_VERSION = "v9fixed"               # wrong bundle
    d._ACTIVE_MANIFEST = {"version": "v9fixed"}
    d._Registry = _FakeRegistry([])
    with pytest.raises(E.ProvenanceError):
        E.run_eval("v9h", "bias_audit", out_root=str(tmp_path / "out"),
                   detector_module=d)
    assert d.calls == [], "no clip may be scored once the gate has refused"


def test_cli_has_no_flag_to_skip_the_provenance_gate():
    """A measurement whose weights cannot be identified has no value, so there is
    deliberately nothing to override. Guards against a future 'just this once' flag."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        E.main(["--help"])
    help_text = buf.getvalue()
    for flag in ("--skip-provenance", "--skip-gate", "--no-verify", "--force"):
        assert flag not in help_text
