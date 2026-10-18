#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh

campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
repo="$campaign/source"
snapshot_name=${1:?usage: freeze_source.sh SNAPSHOT_NAME}
[[ "$snapshot_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'invalid snapshot name' >&2; exit 2; }
snapshot="$campaign/snapshots/$snapshot_name"
[[ ! -e "$snapshot" ]] || { echo "snapshot already exists: $snapshot" >&2; exit 2; }
base=93c8a747252d27e4549fba4e4b23abc19d6114bb
[[ "$(git -C "$repo" rev-parse HEAD)" == "$base" ]] || { echo 'source base commit changed' >&2; exit 2; }

mkdir -p "$snapshot"
git -C "$repo" diff --binary HEAD > "$snapshot/changes.patch"
git -C "$repo" ls-files --others --exclude-standard -z |
  tar --null -C "$repo" -T - -cf "$snapshot/untracked.tar"
git -C "$repo" status --short > "$snapshot/git_status.txt"
git -C "$repo" rev-parse HEAD > "$snapshot/base_commit.txt"
tar --exclude='*/__pycache__' --exclude='*/.pytest_cache' -C "$campaign" -czf "$snapshot/source_full.tar.gz.tmp" source
mv "$snapshot/source_full.tar.gz.tmp" "$snapshot/source_full.tar.gz"
(
  cd "$snapshot"
  sha256sum base_commit.txt changes.patch untracked.tar source_full.tar.gz > SHA256SUMS
)
sha256sum "$snapshot/source_full.tar.gz"
mkdir "$snapshot/runtime"
tar -C "$snapshot/runtime" -xzf "$snapshot/source_full.tar.gz"
SNAPSHOT_DIR="$snapshot" /usr/bin/python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

snapshot = Path(os.environ['SNAPSHOT_DIR'])
root = snapshot / 'runtime/source'
records = {}
for directory, folders, names in os.walk(root):
    folders[:] = sorted(name for name in folders if name not in ('.git', '__pycache__', '.pytest_cache'))
    for name in sorted(names):
        path = Path(directory) / name
        if path.suffix in ('.pyc', '.pyo'):
            continue
        assert path.resolve().is_relative_to(root.resolve()), 'Snapshot contains an external file link'
        with path.open('rb') as file:
            records[str(path.relative_to(root))] = hashlib.file_digest(file, 'sha256').hexdigest()
(snapshot / 'runtime_source_sha256.json').write_text(json.dumps(records, indent=2) + '\n')
with (snapshot / 'source_full.tar.gz').open('rb') as file:
    sha = hashlib.file_digest(file, 'sha256').hexdigest()
manifest_sha = hashlib.sha256((snapshot / 'runtime_source_sha256.json').read_bytes()).hexdigest()
provenance = snapshot / 'provenance'
provenance.mkdir()
campaign = snapshot.parent.parent
for relative in ('assets/pylock.runtime-cu129.toml', 'assets/runtime_installed_versions.json',
                 'assets/native_gl_manifest.json', 'assets/hf_asset_sizes.json',
                 'assets/reward-fast-manifest.json', 'assets/ltx-fast-manifest.json',
                 'logs/runtime_imports_cu129.json', 'logs/runtime_dependency_exception_review.json',
                 'assets/runtime_gpu_validation.json', 'assets/runtime-selected.env',
                 'logs/cpu_runtime_results.xml', 'logs/cpu_checkpoint_clock_results.xml'):
    shutil.copy2(campaign / relative, provenance / Path(relative).name)
for relative in ('assets/pytest_asyncio_test_only.json', 'logs/four_variant_rollout_config_cpu.json',
                 'scripts/run_cpu_regressions.sh', 'scripts/validate_rollout_config_cpu.py'):
    shutil.copy2(campaign / relative, provenance / Path(relative).name)
runtime_sha = hashlib.sha256((provenance / 'pylock.runtime-cu129.toml').read_bytes()).hexdigest()
native_sha = hashlib.sha256((provenance / 'native_gl_manifest.json').read_bytes()).hexdigest()
(snapshot / 'source_snapshot.env').write_text(
    f'export SOURCE_SNAPSHOT_DIR={snapshot}\n'
    f'export SOURCE_SNAPSHOT_SHA256={sha}\n'
    f'export SOURCE_FILES_MANIFEST_SHA256={manifest_sha}\n'
    f'export RUNTIME_LOCK_SHA256={runtime_sha}\n'
    f'export NATIVE_GL_MANIFEST_SHA256={native_sha}\n')
print('Frozen runtime checkout:', root, 'files:', len(records))
PY
