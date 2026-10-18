#!/usr/bin/env bash
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
: "${SOURCE_SNAPSHOT_DIR:?a frozen campaign snapshot is required}"
: "${SOURCE_SNAPSHOT_SHA256:?}"
: "${SOURCE_FILES_MANIFEST_SHA256:?}"
[[ "$SOURCE_SNAPSHOT_DIR" == "$campaign/snapshots/"* ]] || {
  echo 'Source snapshot must stay under campaign snapshots' >&2
  return 1
}
export SOURCE_CHECKOUT_DIR="$SOURCE_SNAPSHOT_DIR/runtime/source"
export PYTHONPATH="$SOURCE_CHECKOUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
"$campaign/env/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

snapshot = Path(os.environ['SOURCE_SNAPSHOT_DIR']).resolve()
campaign = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
assert snapshot.is_relative_to(campaign / 'snapshots')
root = snapshot / 'runtime/source'
manifest = snapshot / 'runtime_source_sha256.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == os.environ['SOURCE_FILES_MANIFEST_SHA256']
checksums = dict(line.split()[::-1] for line in (snapshot / 'SHA256SUMS').read_text().splitlines())
assert checksums['source_full.tar.gz'] == os.environ['SOURCE_SNAPSHOT_SHA256']
expected_versions = json.loads((snapshot / 'provenance/runtime_installed_versions.json').read_text())
actual_versions = sorted([p.metadata['Name'], p.version] for p in importlib.metadata.distributions(
    path=[str(campaign / 'env/lib/python3.12/site-packages')]))
assert actual_versions == expected_versions, 'Installed runtime differs from frozen provenance'
for name, expected in json.loads(manifest.read_text()).items():
    path = root / name
    assert path.resolve().is_relative_to(root)
    with path.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == expected, 'Frozen source changed: ' + name
print('Verified frozen source:', root, file=sys.stderr)
PY
