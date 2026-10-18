"""User-authorized shared storage mode: actual I/O health, no free-space floor."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
MODE = 'shared_no_checkpoint_no_fixed_reserve'
AMENDMENT = ROOT / 'assets/user_no_checkpoint_storage_amendment_20261017.json'


def validate_policy(policy):
    assert policy['storage_admission_mode'] == MODE
    assert policy['checkpoint_writes'] is False
    assert policy['user_amendment_sha256'] == hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment['storage_admission_mode'] == MODE
    assert amendment['periodic_checkpoint_writes'] is amendment['final_checkpoint_writes'] is False
    assert amendment['checkpoint_readback_requirement_waived'] is True
    assert amendment['historical_checkpoint_deletion_authorized'] is False


def check_write_health(directory=ROOT / 'tmp'):
    """Write, fsync, and remove one small private file; propagate real I/O errors."""
    fd, name = tempfile.mkstemp(prefix='storage-health-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(b'\0' * 4096)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        Path(name).unlink(missing_ok=True)


def validate_manifest(manifest, policy_path, start_key='minimum_free_gib'):
    policy = json.loads(Path(policy_path).read_text())
    validate_policy(policy)
    assert manifest['storage_admission_mode'] == MODE
    assert manifest[start_key] == manifest['dynamic_stop_free_gib'] == 0
    return policy
