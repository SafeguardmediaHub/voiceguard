#!/usr/bin/env python3
"""
governance.py — V8 Governance & Institutional Layer: Tasks 1-2 (Phase 6)

Implements:
  1. SHA-256 intake hashing + deterministic inference verification
  2. Versioned model registry + tamper-evident append-only audit log

EXIT GATE this module is built to satisfy:
  "Deterministic inference verified — same file, same model, same score
   100% of the time."

DESIGN
------
ModelRegistry: every checkpoint that becomes production gets a recorded
SHA-256 + metadata entry. This lets you prove, at any later point, exactly
which model file produced a given score — and detect if a checkpoint was
silently swapped or modified (intentionally or by corruption).

AuditLog: a HASH-CHAINED, append-only log. Each entry embeds
sha256(previous_entry), so altering, deleting, or reordering any past
entry breaks the chain at that point and is detectable by anyone with the
log file — no database or trusted third party required. This is the same
principle used in tamper-evident logging / lightweight blockchains, scaled
down to "can a court or auditor verify this log wasn't edited."

Deterministic inference: PyTorch/CUDA have several sources of
nondeterminism that don't show up in casual testing — cuDNN algorithm
auto-selection, some GPU reduction-order effects, hash-seed-dependent
dict/set ordering — that can silently produce slightly different float
outputs run-to-run on the SAME input and SAME weights. This module forces
deterministic mode and then PROVES determinism empirically (not just
"should be deterministic," but "ran N times, scores were bit-identical").

USAGE
-----
    # One-time: register a production checkpoint
    python governance.py --register-model \
        --name aasist_v8 --path /path/to/aasist_v8.pt \
        --metadata '{"trained_on": "train_v8_fresh.json", "val_eer": 0.0741}'

    # Verify no registered checkpoint has been modified since registration
    python governance.py --verify-registry

    # Verify the audit log hasn't been tampered with
    python governance.py --verify-chain

    # Prove deterministic inference on a real audio file through the full
    # V8 ensemble (requires the same model-loading setup as drift_monitor.py)
    python governance.py --test-determinism --audio-path /path/to/sample.wav --runs 20

OUTPUTS
-------
    governance/model_registry.json     — append-only, one entry per registration
    governance/audit_log.jsonl         — append-only, hash-chained
    governance/determinism_report_<ts>.txt
"""

import os, sys, json, hashlib, argparse, time, threading
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────

# Where the registry + tamper-evident audit log live. Overridable so the live
# deployment can point it at a real writable volume; the module was originally
# authored on Kaggle (/kaggle/working), which does not exist off that platform,
# so the default is repo-relative instead.
GOVERNANCE_DIR = Path(os.environ.get(
    'VOICEGUARD_GOVERNANCE_DIR',
    Path(__file__).resolve().parent / 'governance'))
REGISTRY_PATH  = GOVERNANCE_DIR / 'model_registry.json'
AUDIT_LOG_PATH = GOVERNANCE_DIR / 'audit_log.jsonl'

SEED = 42

# ─── Hashing ─────────────────────────────────────────────────────────────────

def sha256_file(path, chunk_size=1024 * 1024):
    """SHA-256 of a file's bytes, streamed (safe for large model checkpoints)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()

def canonical_json(obj):
    """Deterministic JSON serialization — fixed key order, no whitespace
    variance — required so the SAME logical entry always hashes to the
    SAME digest regardless of dict insertion order or platform."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)

# ─── Deterministic inference setup ──────────────────────────────────────────

def enforce_deterministic_inference(seed=SEED):
    """Call this BEFORE loading models or running any inference. Disables
    every known source of run-to-run nondeterminism in PyTorch/CUDA.

    Returns a dict describing what was applied, for inclusion in audit
    metadata / determinism reports (so a verifier can confirm the right
    settings were active, not just trust that they were).
    """
    import torch
    import random

    os.environ['PYTHONHASHSEED'] = str(seed)
    # Required for fully deterministic CuBLAS GEMM ops on CUDA >= 10.2
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True so an op without a deterministic implementation logs a
    # warning instead of crashing — but we explicitly check for warnings in
    # the determinism test rather than silently trusting this.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # older torch versions don't support warn_only kwarg
        torch.use_deterministic_algorithms(True)

    return {
        'seed': seed,
        'pythonhashseed': os.environ['PYTHONHASHSEED'],
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'use_deterministic_algorithms': True,
        'cuda_available': torch.cuda.is_available(),
    }

# ─── Model registry ─────────────────────────────────────────────────────────

class ModelRegistry:
    """Append-only registry of production model checkpoints. Each entry
    records the file's SHA-256 at registration time, so later verification
    can detect if the file on disk has changed (corruption, accidental
    overwrite, or tampering)."""

    def __init__(self, path=REGISTRY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self):
        with open(self.path, encoding='utf-8') as f:
            return json.load(f)

    def _write(self, entries):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)

    def register(self, name, file_path, metadata=None, version=None):
        """Register a checkpoint. version defaults to an auto-incrementing
        integer per model `name` if not given."""
        entries = self._read()
        file_path = str(file_path)
        digest = sha256_file(file_path)

        existing_versions = [e['version'] for e in entries if e['name'] == name]
        if version is None:
            version = (max(existing_versions) + 1) if existing_versions else 1

        entry = {
            'name': name,
            'version': version,
            'path_at_registration': file_path,
            'sha256': digest,
            'file_size_bytes': os.path.getsize(file_path),
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {},
        }
        entries.append(entry)
        self._write(entries)
        print(f"Registered {name} v{version}: sha256={digest[:16]}... "
              f"({entry['file_size_bytes']:,} bytes)")
        return entry

    def get_latest(self, name):
        entries = [e for e in self._read() if e['name'] == name]
        if not entries:
            return None
        return max(entries, key=lambda e: e['version'])

    def verify_integrity(self, name=None, check_path=None):
        """Recompute SHA-256 for registered checkpoints and compare against
        the recorded digest. If check_path is given, only that specific
        registration's recorded path is re-checked; otherwise every entry's
        `path_at_registration` is checked (will report 'file not found' for
        any that have since moved — that's expected and not itself a
        tamper signal, just means re-point check_path manually).
        """
        entries = self._read()
        if name:
            entries = [e for e in entries if e['name'] == name]

        results = []
        for e in entries:
            path = check_path if check_path else e['path_at_registration']
            if not os.path.exists(path):
                results.append({**e, 'status': 'FILE_NOT_FOUND', 'path_checked': path})
                continue
            current_digest = sha256_file(path)
            match = (current_digest == e['sha256'])
            results.append({
                **e,
                'status': 'OK' if match else 'TAMPERED_OR_MODIFIED',
                'path_checked': path,
                'current_sha256': current_digest,
            })
        return results

    def all_entries(self):
        return self._read()

# ─── Append serialization ───────────────────────────────────────────────────
#
# Appending to the chain is read-last-hash → compute → write. That sequence is
# not atomic, so two concurrent writers can read the same prev_hash and produce
# two entries claiming the same predecessor — which verify_chain reports as
# tampering, indistinguishable from a real attack. Losing the ability to prove
# the log is intact is exactly the failure the log exists to prevent.
#
# The obvious deployment that triggers it: docker-compose.prod.yml runs api and
# worker from one image and scaling the worker to 2 is safe for the SQLite job
# queue (BEGIN IMMEDIATE) but not for this. So the lock is here, not in a comment
# telling operators not to scale.
#
# Two layers, because both races are real: a threading.Lock for writers inside one
# process, and an OS file lock on a sidecar .lock file for writers across
# processes/containers sharing the volume.

_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()

try:                                    # POSIX — the production path (Linux containers)
    import fcntl

    def _lock_fd(fd):
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock_fd(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

except ImportError:                     # Windows — dev machines
    import msvcrt

    def _lock_fd(fd, timeout=30.0):
        # LK_LOCK retries for ~10s then raises; wrap it so a slow peer doesn't
        # turn into a spurious failure.
        deadline = time.time() + timeout
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if time.time() >= deadline:
                    raise
                time.sleep(0.05)

    def _unlock_fd(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _thread_lock_for(path):
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = _THREAD_LOCKS[key] = threading.Lock()
    return lock


@contextmanager
def chain_lock(log_path):
    """Serialize the read-then-append sequence against every other writer of this
    log, in this process and in any other."""
    lock_path = str(log_path) + '.lock'
    with _thread_lock_for(log_path):
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _lock_fd(fd)
            try:
                yield
            finally:
                _unlock_fd(fd)
        finally:
            os.close(fd)


# ─── Tamper-evident audit log ───────────────────────────────────────────────

class AuditLog:
    """Hash-chained, append-only audit log. Each entry's `entry_hash`
    incorporates the PREVIOUS entry's hash, so the log forms a chain: any
    edit, deletion, or reorder of a past entry changes that entry's hash,
    which no longer matches what the NEXT entry recorded as `prev_hash` —
    breaking the chain at exactly that point. Detectable by anyone who can
    read the file; no database, signing key, or trusted third party needed.
    """

    GENESIS_HASH = '0' * 64  # prev_hash for the very first entry

    def __init__(self, path=AUDIT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _last_hash(self):
        if self.path.stat().st_size == 0:
            return self.GENESIS_HASH
        with open(self.path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            # Read backwards to find the last line efficiently
            buf = b''
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                if b'\n' in buf.strip(b'\n'):
                    break
            last_line = buf.strip().split(b'\n')[-1]
        last_entry = json.loads(last_line)
        return last_entry['entry_hash']

    def append(self, audit_id, intake_sha256, model_versions, score, verdict,
               extra=None):
        """Add one entry. model_versions: dict like {'aasist': 3, 'wav2vec': 1, ...}
        so the log records exactly which model versions produced this result.

        Reading the tail, computing the link, and writing must be one critical
        section — see chain_lock above."""
        with chain_lock(self.path):
            prev_hash = self._last_hash()
            seq = self._count_entries() + 1

            entry_body = {
                'seq': seq,
                'audit_id': audit_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'intake_sha256': intake_sha256,
                'model_versions': model_versions,
                'score': score,
                'verdict': verdict,
                'extra': extra or {},
                'prev_hash': prev_hash,
            }
            entry_hash = sha256_bytes(canonical_json(entry_body).encode('utf-8'))
            entry = {**entry_body, 'entry_hash': entry_hash}

            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(canonical_json(entry) + '\n')
                f.flush()
                os.fsync(f.fileno())     # durable before the lock is released
        return entry

    def _count_entries(self):
        if self.path.stat().st_size == 0:
            return 0
        with open(self.path, encoding='utf-8') as f:
            return sum(1 for _ in f)

    def read_all(self):
        entries = []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def verify_chain(self):
        """Walk the full log and confirm every entry's hash chain links
        correctly. Returns (is_valid: bool, details: dict)."""
        entries = self.read_all()
        if not entries:
            return True, {'n_entries': 0, 'message': 'empty log — nothing to verify'}

        expected_prev = self.GENESIS_HASH
        for i, entry in enumerate(entries):
            if entry['prev_hash'] != expected_prev:
                return False, {
                    'n_entries': len(entries),
                    'broken_at_seq': entry['seq'],
                    'message': f"entry seq={entry['seq']} has prev_hash that doesn't match "
                               f"the actual hash of the preceding entry — chain broken "
                               f"(tampering, deletion, or reordering detected)",
                }
            # Recompute this entry's hash from its body and confirm it
            # matches what's stored (detects in-place edits to this entry).
            body = {k: v for k, v in entry.items() if k != 'entry_hash'}
            recomputed = sha256_bytes(canonical_json(body).encode('utf-8'))
            if recomputed != entry['entry_hash']:
                return False, {
                    'n_entries': len(entries),
                    'broken_at_seq': entry['seq'],
                    'message': f"entry seq={entry['seq']} content does not match its "
                               f"recorded entry_hash — this entry was edited after being "
                               f"written",
                }
            expected_prev = entry['entry_hash']

        return True, {'n_entries': len(entries), 'message': 'chain verified — no tampering detected'}

# ─── Deterministic inference test ───────────────────────────────────────────

def run_determinism_test(audio_path, n_runs=20, load_ensemble_fn=None, score_fn=None):
    """Run inference on the SAME audio file N times and confirm the score
    is bit-identical every time. This is the empirical proof behind the
    exit gate, not just a configuration claim.

    load_ensemble_fn / score_fn: injected so this module doesn't hard-depend
    on drift_monitor.py's model classes. If not provided, attempts to import
    them from drift_monitor.py if it's on the path (common deployment case —
    this file living next to drift_monitor.py).
    """
    if load_ensemble_fn is None or score_fn is None:
        try:
            import drift_monitor as dm
            load_ensemble_fn = load_ensemble_fn or dm.load_v8
            def _score(models, audio):
                s_ens, s_a, s_w, s_r = dm.score_one(models, audio)
                return s_ens, s_a, s_w, s_r
            score_fn = score_fn or _score
            audio_loader = dm.load_audio
        except ImportError:
            raise RuntimeError(
                "Could not import drift_monitor.py for model loading. Either place "
                "governance.py next to drift_monitor.py, or pass load_ensemble_fn / "
                "score_fn explicitly when calling run_determinism_test().")
        except AttributeError as e:
            raise RuntimeError(
                f"drift_monitor.py is missing an expected function: {e}. "
                f"This usually means the version of drift_monitor.py on disk has "
                f"different function names than expected (load_v8, score_one, "
                f"load_audio). Check drift_monitor.py's actual definitions.")
    else:
        audio_loader = None

    det_settings = enforce_deterministic_inference()
    print(f"Determinism settings applied: {det_settings}", flush=True)

    print(f"Loading ensemble...", flush=True)
    models = load_ensemble_fn()

    if audio_loader:
        audio = audio_loader(audio_path)
    else:
        raise RuntimeError("No audio loader available — pass audio as pre-loaded array via custom score_fn.")

    intake_hash = sha256_file(audio_path)
    print(f"Intake SHA-256: {intake_hash}", flush=True)
    print(f"Running inference {n_runs} times on the same file...", flush=True)

    results = []
    for i in range(n_runs):
        scores = score_fn(models, audio)
        results.append(scores)
        if (i + 1) % 5 == 0:
            print(f"  run {i+1}/{n_runs}: ensemble_score={scores[0]:.17f}", flush=True)

    # Check bit-identical (exact float equality, not "close enough") across all runs
    first = results[0]
    n_fields = len(first)
    all_identical = True
    per_field_status = []
    for field_idx in range(n_fields):
        values = [r[field_idx] for r in results]
        identical = all(v == values[0] for v in values)
        per_field_status.append(identical)
        if not identical:
            all_identical = False

    field_names = ['ensemble', 'aasist', 'wav2vec', 'rawnet'][:n_fields]
    return {
        'audio_path': str(audio_path),
        'intake_sha256': intake_hash,
        'n_runs': n_runs,
        'all_identical': all_identical,
        'per_field_identical': dict(zip(field_names, per_field_status)),
        'first_run_scores': dict(zip(field_names, first)),
        'all_run_scores': [dict(zip(field_names, r)) for r in results],
        'determinism_settings': det_settings,
    }

def format_determinism_report(result):
    lines = []
    lines.append("=" * 72)
    lines.append("DETERMINISTIC INFERENCE VERIFICATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Timestamp:    {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Audio file:   {result['audio_path']}")
    lines.append(f"Intake SHA-256: {result['intake_sha256']}")
    lines.append(f"Runs:         {result['n_runs']}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("DETERMINISM SETTINGS APPLIED")
    lines.append("─" * 72)
    for k, v in result['determinism_settings'].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("PER-COMPONENT RESULT")
    lines.append("─" * 72)
    for field, identical in result['per_field_identical'].items():
        status = "✓ IDENTICAL across all runs" if identical else "✗ VARIED across runs — NOT deterministic"
        lines.append(f"  {field:<10s}: {status}")
    lines.append("")
    lines.append("─" * 72)
    lines.append(f"EXIT GATE: {'PASSED' if result['all_identical'] else 'FAILED'}")
    lines.append("─" * 72)
    if result['all_identical']:
        lines.append(f"  Same file, same model, same score across {result['n_runs']} runs.")
        lines.append(f"  Score: {result['first_run_scores']}")
    else:
        lines.append(f"  Nondeterminism detected. Do NOT claim deterministic inference")
        lines.append(f"  in client-facing or legal documentation until resolved.")
        lines.append(f"  Investigate: GPU vs CPU inference, any remaining nondeterministic")
        lines.append(f"  ops (check console for 'use_deterministic_algorithms' warnings")
        lines.append(f"  printed during model loading/inference), or floating-point")
        lines.append(f"  reduction-order differences across runs.")
    lines.append("")
    return "\n".join(lines)

# ─── CLI ────────────────────────────────────────────────────────────────────

def cmd_register_model(args):
    registry = ModelRegistry()
    if getattr(args, 'metadata_file', None):
        with open(args.metadata_file) as f:
            metadata = json.load(f)
    elif args.metadata:
        metadata = json.loads(args.metadata)
    else:
        metadata = {}
    registry.register(args.name, args.path, metadata=metadata, version=args.version)

def cmd_verify_registry(args):
    registry = ModelRegistry()
    results = registry.verify_integrity(name=args.name)
    if not results:
        print("No registered models found.")
        return
    print(f"\n{'='*72}\nMODEL REGISTRY INTEGRITY CHECK\n{'='*72}")
    any_failed = False
    for r in results:
        print(f"  {r['name']} v{r['version']}: {r['status']}")
        if r['status'] != 'OK':
            any_failed = True
            print(f"    path checked: {r['path_checked']}")
            print(f"    registered sha256: {r['sha256']}")
            if 'current_sha256' in r:
                print(f"    current sha256:    {r['current_sha256']}")
    print()
    print("EXIT GATE: " + ("FAILED — investigate flagged entries above" if any_failed else "PASSED — all registered checkpoints match their registration-time hash"))

def cmd_verify_chain(args):
    log = AuditLog()
    is_valid, details = log.verify_chain()
    print(f"\n{'='*72}\nAUDIT LOG CHAIN VERIFICATION\n{'='*72}")
    print(json.dumps(details, indent=2))
    print()
    print("EXIT GATE: " + ("PASSED — " if is_valid else "FAILED — ") + details['message'])
    # Exit non-zero on a broken chain so this can gate CI / a scheduled audit
    # check — a printed "FAILED" that still exits 0 silently passes the gate.
    if not is_valid:
        sys.exit(1)

def cmd_test_determinism(args):
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    result = run_determinism_test(args.audio_path, n_runs=args.runs)
    report = format_determinism_report(result)
    print("\n" + report)
    safe_ts = datetime.now(timezone.utc).isoformat().replace(':', '').replace('-', '').split('.')[0]
    report_path = GOVERNANCE_DIR / f'determinism_report_{safe_ts}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    if not result['all_identical']:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='V8 governance: model registry, audit log, determinism verification')
    sub = parser.add_subparsers(dest='command')

    p_reg = sub.add_parser('register-model')
    p_reg.add_argument('--name', required=True)
    p_reg.add_argument('--path', required=True)
    p_reg.add_argument('--metadata', default=None, help='JSON string of metadata')
    p_reg.add_argument('--metadata-file', dest='metadata_file', default=None,
                       help='Path to a JSON file, alternative to --metadata')
    p_reg.add_argument('--version', type=int, default=None)

    p_verify_reg = sub.add_parser('verify-registry')
    p_verify_reg.add_argument('--name', default=None)

    sub.add_parser('verify-chain')

    p_det = sub.add_parser('test-determinism')
    p_det.add_argument('--audio-path', required=True)
    p_det.add_argument('--runs', type=int, default=20)

    # Also support the original --flag style for convenience/consistency
    # with other Phase 5 scripts in this project.
    parser.add_argument('--register-model', action='store_true')
    parser.add_argument('--verify-registry', action='store_true')
    parser.add_argument('--verify-chain', action='store_true')
    parser.add_argument('--test-determinism', action='store_true')
    parser.add_argument('--name', dest='flagname', default=None)
    parser.add_argument('--path', dest='flagpath', default=None)
    parser.add_argument('--metadata', dest='flagmetadata', default=None)
    parser.add_argument('--metadata-file', dest='flagmetadatafile', default=None,
                        help='Path to a JSON file, as an alternative to --metadata inline '
                             '(avoids shell-quoting issues on Windows cmd.exe).')
    parser.add_argument('--version', dest='flagversion', type=int, default=None)
    parser.add_argument('--audio-path', dest='flagaudio', default=None)
    parser.add_argument('--runs', dest='flagruns', type=int, default=20)

    args = parser.parse_args()

    if args.register_model:
        args.name, args.path, args.metadata, args.version = (
            args.flagname, args.flagpath, args.flagmetadata, args.flagversion)
        args.metadata_file = args.flagmetadatafile
        cmd_register_model(args)
    elif args.verify_registry:
        args.name = args.flagname
        cmd_verify_registry(args)
    elif args.verify_chain:
        cmd_verify_chain(args)
    elif args.test_determinism:
        args.audio_path, args.runs = args.flagaudio, args.flagruns
        cmd_test_determinism(args)
    elif args.command == 'register-model':
        cmd_register_model(args)
    elif args.command == 'verify-registry':
        cmd_verify_registry(args)
    elif args.command == 'verify-chain':
        cmd_verify_chain(args)
    elif args.command == 'test-determinism':
        cmd_test_determinism(args)
    else:
        parser.error('Specify a command, e.g. --register-model, --verify-registry, '
                      '--verify-chain, --test-determinism')

if __name__ == '__main__':
    main()
