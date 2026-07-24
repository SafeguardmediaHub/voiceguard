"""VoiceGuard model-flow hub: bundles, registry, active pointer.

A *bundle* is the atomic deployable unit — the 7 co-calibrated V9 artifacts
plus a bundle.json manifest. See docs/superpowers/specs/2026-07-07-model-flow-hub-design.md.
"""
import os, sys, json, shutil, argparse
from datetime import datetime, timezone

from governance import sha256_file, sha256_bytes, canonical_json
import remote_store

STORE_DIR      = os.environ.get("VOICEGUARD_MODEL_STORE", "model_store")
REGISTRY_FILE  = "registry.jsonl"     # under STORE_DIR
ACTIVE_FILE    = "ACTIVE.json"        # under STORE_DIR

BUNDLE_FILES = {
    "aasist.pt", "wav2vec.pt", "rawnet.pt",
    "xgb.json", "cal.json", "thresholds.json", "lcnn.pt",
}


class BundleError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def write_manifest(bundle_dir, meta):
    """Compute sha256 of each of the 7 files, merge into meta['files'],
    write bundle.json, return its path."""
    files = {}
    for name in sorted(BUNDLE_FILES):
        p = os.path.join(bundle_dir, name)
        if not os.path.exists(p):
            raise BundleError(f"cannot write manifest: missing {name} in {bundle_dir}")
        files[name] = {"sha256": sha256_file(p)}
    manifest = dict(meta)
    manifest["files"] = files
    manifest.setdefault("created_at", _now())
    path = os.path.join(bundle_dir, "bundle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(bundle_dir):
    with open(os.path.join(bundle_dir, "bundle.json"), encoding="utf-8") as f:
        return json.load(f)


def validate_bundle(bundle_dir):
    """Raise BundleError unless all 7 files + bundle.json are present and each
    file's sha256 matches the manifest."""
    if not os.path.exists(os.path.join(bundle_dir, "bundle.json")):
        raise BundleError(f"no bundle.json in {bundle_dir}")
    manifest = read_manifest(bundle_dir)
    recorded = manifest.get("files", {})
    for name in BUNDLE_FILES:
        p = os.path.join(bundle_dir, name)
        if not os.path.exists(p):
            raise BundleError(f"missing bundle file: {name}")
        if name not in recorded:
            raise BundleError(f"manifest has no hash for: {name}")
        actual = sha256_file(p)
        if actual != recorded[name]["sha256"]:
            raise BundleError(f"sha256 mismatch for {name} "
                              f"(disk {actual[:12]} != manifest {recorded[name]['sha256'][:12]})")


class Registry:
    def __init__(self, store_dir=STORE_DIR):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.registry_path = os.path.join(self.store_dir, REGISTRY_FILE)
        self.active_path = os.path.join(self.store_dir, ACTIVE_FILE)

    # ── registry.jsonl ──
    def _read_registry(self):
        if not os.path.exists(self.registry_path):
            return []
        out = []
        with open(self.registry_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def register_bundle(self, bundle_dir):
        validate_bundle(bundle_dir)
        manifest = read_manifest(bundle_dir)
        version = manifest["version"]
        if any(e["version"] == version for e in self._read_registry()):
            raise BundleError(f"version already registered: {version}")
        entry = {
            "version": version,
            "registered_at": _now(),
            "dir": os.path.abspath(bundle_dir),
            "files": {n: v["sha256"] for n, v in manifest["files"].items()},
            "manifest": manifest,
        }
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return version

    def list_bundles(self):
        return list(reversed(self._read_registry()))

    def get_bundle(self, version):
        for e in self._read_registry():
            if e["version"] == version:
                return e
        return None

    def verify_integrity(self, version):
        e = self.get_bundle(version)
        if e is None:
            return False
        for name, recorded_sha in e["files"].items():
            p = os.path.join(e["dir"], name)
            if not os.path.exists(p) or sha256_file(p) != recorded_sha:
                return False
        return True

    # ── ACTIVE.json (pointer) ──
    def _read_active_log(self):
        if not os.path.exists(self.active_path):
            return []
        with open(self.active_path, encoding="utf-8") as f:
            return json.load(f)

    def get_active(self):
        log = self._read_active_log()
        return log[-1]["version"] if log else None

    def _append_active(self, version, prev_version, actor, reason):
        log = self._read_active_log()
        prev_sha = log[-1]["entry_sha"] if log else "GENESIS"
        entry = {
            "seq": len(log),
            "version": version,
            "prev_version": prev_version,
            "activated_at": _now(),
            "actor": actor,
            "reason": reason,
            "prev_sha": prev_sha,
        }
        entry["entry_sha"] = sha256_bytes(
            (canonical_json({k: entry[k] for k in entry if k != "entry_sha"})
             + prev_sha).encode())
        log.append(entry)
        # Atomic write: write to a temp file then os.replace (atomic on the same
        # filesystem, POSIX + Windows), so a crash mid-write can never corrupt
        # the existing tamper-evident pointer log — you get either the old
        # complete file or the new complete file, never a partial one.
        tmp_path = self.active_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        os.replace(tmp_path, self.active_path)

    def promote(self, version, actor="cli", reason=""):
        if self.get_bundle(version) is None:
            raise BundleError(f"cannot promote unregistered version: {version}")
        if not self.verify_integrity(version):
            raise BundleError(f"integrity check failed for {version}; refusing to promote")
        prev = self.get_active()
        self._append_active(version, prev_version=prev, actor=actor, reason=reason)

    def rollback(self, actor="cli", reason=""):
        log = self._read_active_log()
        if not log or log[-1].get("prev_version") is None:
            raise BundleError("no previous active version to roll back to")
        target = log[-1]["prev_version"]
        self._append_active(target, prev_version=log[-1]["version"],
                            actor=actor, reason=reason or "rollback")
        return target

    def verify_active_chain(self):
        prev_sha = "GENESIS"
        for entry in self._read_active_log():
            expect = sha256_bytes(
                (canonical_json({k: entry[k] for k in entry if k != "entry_sha"})
                 + prev_sha).encode())
            if expect != entry.get("entry_sha") or entry.get("prev_sha") != prev_sha:
                return False
            prev_sha = entry["entry_sha"]
        return True

    # ── remote store (DigitalOcean Spaces) ──
    def push(self, version, client=None):
        """Upload bundle <version> (7 files + bundle.json) to Spaces and refresh the remote
        ACTIVE.json + registry.jsonl. Verifies local integrity first — never push a corrupt
        bundle."""
        entry = self.get_bundle(version)
        if entry is None:
            raise BundleError(f"cannot push unregistered version: {version}")
        if not self.verify_integrity(version):
            raise BundleError(f"integrity check failed for {version}; refusing to push")
        if client is None:
            client = remote_store.make_client()
        bucket, prefix = remote_store.bucket_prefix()
        for name in sorted(BUNDLE_FILES | {"bundle.json"}):
            remote_store.upload_file(client, bucket, f"{prefix}/bundles/{version}/{name}",
                                     os.path.join(entry["dir"], name))
        if os.path.exists(self.active_path):
            remote_store.upload_file(client, bucket, f"{prefix}/ACTIVE.json", self.active_path)
        if os.path.exists(self.registry_path):
            remote_store.upload_file(client, bucket, f"{prefix}/registry.jsonl", self.registry_path)
        return version

    def push_active(self, client=None):
        active = self.get_active()
        if active is None:
            raise BundleError("no active version to push")
        return self.push(active, client=client)

    def pull(self, version=None, client=None):
        """Download a bundle from Spaces into store_dir, verify every file's sha256 against its
        manifest, register it locally (correct local dir), and — when it is the remote-active
        version — write ACTIVE.json verbatim so the tamper-evident chain still verifies.
        Fails closed: a sha256 mismatch raises BundleError and never activates the bundle."""
        if client is None:
            client = remote_store.make_client()
        bucket, prefix = remote_store.bucket_prefix()

        active_bytes = remote_store.download_bytes(client, bucket, f"{prefix}/ACTIVE.json")
        remote_active = json.loads(active_bytes)[-1]["version"] if active_bytes else None
        if version is None:
            if remote_active is None:
                raise BundleError("remote store has no active pointer (ACTIVE.json); pass a version")
            version = remote_active

        bundle_dir = os.path.join(self.store_dir, version)
        os.makedirs(bundle_dir, exist_ok=True)
        for name in sorted(BUNDLE_FILES | {"bundle.json"}):
            remote_store.download_file(client, bucket, f"{prefix}/bundles/{version}/{name}",
                                       os.path.join(bundle_dir, name))
        validate_bundle(bundle_dir)                        # raises BundleError on sha256 mismatch

        if self.get_bundle(version) is None:
            self.register_bundle(bundle_dir)               # local entry with correct local dir
        if active_bytes is not None and version == remote_active:
            # verbatim bytes (keeps verify_active_chain valid), written atomically via
            # temp + os.replace — same crash-safe pattern as _append_active, so a crash
            # mid-write can't corrupt the tamper-evident pointer log.
            tmp_path = self.active_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(active_bytes)
            os.replace(tmp_path, self.active_path)
        return version


def _build_cli():
    p = argparse.ArgumentParser(description="VoiceGuard model-flow registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("active")
    r = sub.add_parser("register"); r.add_argument("bundle_dir")
    pr = sub.add_parser("promote"); pr.add_argument("version")
    pr.add_argument("--actor", default="cli"); pr.add_argument("--reason", default="")
    pr.add_argument("--restart", action="store_true",
                    help="run $VOICEGUARD_RESTART_CMD after promoting")
    pr.add_argument("--skip-health", action="store_true",
                    help="skip the per-sub-model health gate (records the skip in the reason)")
    pr.add_argument("--health-min-spread", type=float, default=0.05,
                    help="collapse floor passed to the health gate (default 0.05)")
    rb = sub.add_parser("rollback"); rb.add_argument("--actor", default="cli")
    rb.add_argument("--reason", default=""); rb.add_argument("--restart", action="store_true")
    vf = sub.add_parser("verify"); vf.add_argument("version")
    ps = sub.add_parser("push"); ps.add_argument("version", nargs="?"); ps.add_argument("--active", action="store_true")
    pl = sub.add_parser("pull"); pl.add_argument("version", nargs="?"); pl.add_argument("--active", action="store_true")
    sub.add_parser("migrate-v9")
    lg = sub.add_parser("log"); lg.add_argument("version")
    return p


def _run_health_gate(version, min_spread=0.05):
    """Run the per-sub-model health check against `version` as a CANDIDATE, before
    it is promoted. Loads the candidate via $VOICEGUARD_FORCE_BUNDLE (no live-pointer
    flip) in a subprocess (detector loads its models at import).

    Returns (ok: bool, detail: str). ok is False on a collapsed sub-model; a
    missing probe set is a distinct "cannot certify" outcome the caller must
    decide on. This exists because the AASIST V9 collapse passed every existing
    gate — all of which check ENSEMBLE output, not per-sub-model contribution.
    See docs/GRC_CONTROL_PACK.md G1/R4 and scripts/submodel_health.py.
    """
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scripts", "submodel_health.py")
    if not os.path.exists(script):
        return False, f"health script not found: {script}"
    env = dict(os.environ, VOICEGUARD_FORCE_BUNDLE=version)
    proc = subprocess.run(
        [sys.executable, script, "--json", "--min-spread", str(min_spread)],
        env=env, capture_output=True, text=True)
    # The health script emits one sentinel-prefixed JSON line on stdout (even on a
    # FAIL exit); detector's import prints a lot of other noise there, so we grep
    # for the prefix rather than parsing the whole stream. Absent prefix => the
    # candidate did not even produce a result (failed to load, or no probe data).
    prefix = "HEALTH_RESULT_JSON "
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(prefix):
            try:
                result = json.loads(line[len(prefix):])
            except Exception:
                result = None
            break
    if result is None:
        return None, (proc.stderr.strip() or proc.stdout.strip() or "health check produced no result")
    if result.get("passed"):
        weak = ", ".join(result.get("weak", [])) or "none"
        return True, (f"all sub-models varied (no collapse); weak-but-alive: {weak} "
                      f"[probe {result['n_real']}real/{result['n_fake']}fake]")
    return False, f"COLLAPSED sub-model(s): {', '.join(result.get('collapsed', []))}"


def _maybe_restart(do_restart):
    if not do_restart:
        return
    cmd = os.environ.get("VOICEGUARD_RESTART_CMD")
    if not cmd:
        print("  (--restart given but VOICEGUARD_RESTART_CMD is unset; restart the server manually)")
        return
    print(f"  restarting server: {cmd}")
    os.system(cmd)


def main(argv=None):
    args = _build_cli().parse_args(argv)
    reg = Registry()
    if args.cmd == "list":
        for b in reg.list_bundles():
            active = " (ACTIVE)" if b["version"] == reg.get_active() else ""
            print(f"  {b['version']:8s} registered {b['registered_at'][:19]}{active}")
    elif args.cmd == "active":
        print(reg.get_active() or "(none)")
    elif args.cmd == "register":
        print("registered:", reg.register_bundle(args.bundle_dir))
    elif args.cmd == "promote":
        reason = args.reason
        if args.skip_health:
            print("  [health gate] SKIPPED (--skip-health)")
            reason = (reason + " [health-gate:skipped]").strip()
        else:
            ok, detail = _run_health_gate(args.version, min_spread=args.health_min_spread)
            if ok is None:
                # Could not certify (usually: probe data absent). Fail closed —
                # the operator must supply probe data or pass --skip-health.
                print(f"  [health gate] CANNOT CERTIFY: {detail}", file=sys.stderr)
                print("  Provide the probe folders (studio_clips, bias_audit_fakes, ...) "
                      "or re-run with --skip-health to promote without the gate.",
                      file=sys.stderr)
                raise SystemExit(2)
            if not ok:
                print(f"  [health gate] FAILED: {detail}", file=sys.stderr)
                print(f"  Refusing to promote {args.version}. A collapsed sub-model feeds the "
                      "fusion a dead feature (the AASIST V9 failure). Fix the candidate or "
                      "override with --skip-health.", file=sys.stderr)
                raise SystemExit(1)
            print(f"  [health gate] PASSED: {detail}")
            reason = (reason + " [health-gate:passed]").strip()
        reg.promote(args.version, actor=args.actor, reason=reason)
        print(f"promoted {args.version} (active). Restart the server to apply.")
        _maybe_restart(args.restart)
    elif args.cmd == "rollback":
        tgt = reg.rollback(actor=args.actor, reason=args.reason)
        print(f"rolled back to {tgt}. Restart the server to apply.")
        _maybe_restart(args.restart)
    elif args.cmd == "verify":
        print("OK" if reg.verify_integrity(args.version) else "FAILED")
    elif args.cmd == "push":
        v = reg.push_active() if (args.active or not args.version) else reg.push(args.version)
        print("pushed:", v)
    elif args.cmd == "pull":
        v = reg.pull(None if (args.active or not args.version) else args.version)
        print("pulled:", v, "(active)" if reg.get_active() == v else "")
    elif args.cmd == "log":
        import tracking
        entry = reg.get_bundle(args.version)
        if entry is None:
            print("unknown version"); return
        run_id = tracking.log_bundle(entry["dir"])
        print("mlflow run:", run_id)
    elif args.cmd == "migrate-v9":
        migrate_v9()   # defined in Task 4


_V9_SOURCE = {
    "aasist.pt":       "models/aasist_v9_best.pt",
    "wav2vec.pt":      "models/wav2vec_v9_best.pt",
    "rawnet.pt":       "models/rawnet3.pt",
    "xgb.json":        "models/xgb_v9.json",
    "cal.json":        "models/cal_v9_params.json",
    "thresholds.json": "models/thresholds_v9.json",
    "lcnn.pt":         "models/lcnn_screener_v9.pt",
}


def migrate_v9(store_dir=STORE_DIR):
    import torch
    bundle_dir = os.path.join(store_dir, "v9")
    os.makedirs(bundle_dir, exist_ok=True)
    for dst, src in _V9_SOURCE.items():
        if not os.path.exists(src):
            raise BundleError(f"migration source missing: {src}")
        shutil.copy2(src, os.path.join(bundle_dir, dst))

    with open("models/thresholds_v9.json") as f:
        thresholds = json.load(f)
    lcnn_ck = torch.load("models/lcnn_screener_v9.pt", map_location="cpu", weights_only=False)
    cascade = {
        "low_thresh":  float(lcnn_ck["cascade_thresholds"]["low_thresh"]),
        "high_thresh": float(lcnn_ck["cascade_thresholds"]["high_thresh"]),
        "platt": {"coef": float(lcnn_ck["platt_calibration"]["coef"]),
                  "intercept": float(lcnn_ck["platt_calibration"]["intercept"])},
    }
    write_manifest(bundle_dir, {
        "version": "v9",
        "git_sha": os.popen("git rev-parse --short HEAD").read().strip() or None,
        "mlflow_run_id": None,
        # Provenance metrics (from Handoff_Summary_V9_Phase7.md; refit_ensemble held-out)
        "metrics": {
            "val_eer": 0.1325, "studio_fp": 0.12, "noizai_catch": 0.833,
            "per_language_fpr": {"yoruba": 0.02, "igbo": 0.04, "pidgin": 0.0,
                                 "hausa": 0.0, "arabic": 0.16},
        },
        "train_manifest_hash": None,
        "preprocessing": {"ensemble_peak_norm": True, "lcnn_peak_norm": False,
                          "notes": "LCNN mel per-sample standardized (scale-invariant); "
                                   "ensemble peak-normalizes."},
        "cascade": cascade,
        "verdict_thresholds": thresholds,
        # Startup smoke-check: this clip must score a fake verdict on load.
        "smoke_check": {"clip": "tests/golden_clips/fake_noizai_a4cd.mp3",
                        "expected_verdicts": ["LIKELY_FAKE", "AUTO_FAKE"]},
    })

    reg = Registry(store_dir=store_dir)
    reg.register_bundle(bundle_dir)
    reg.promote("v9", actor="migration", reason="initial V9 bundle")
    return "v9"


if __name__ == "__main__":
    main()
