# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFT child-only cache roots without importing a GPU runtime."""

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / 'scripts/omni_art_joint32/nft_cache_env.sh'
EXTENSION = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/extensions/nft_full32')


class NFTCacheEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nft-cache-test-', dir=EXTENSION / 'tmp')
        self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name)
        self.environment = dict(os.environ, OMNI_VARIANT='NFT_FULL32_U1',
                                OMNI_RUNTIME_ROOT=str(self.runtime),
                                TMPDIR=str(self.runtime / 'tmp'), HOME='/root',
                                FLASHINFER_WORKSPACE_BASE='/root/.cache/flashinfer',
                                HF_HOME=str(EXTENSION / 'unchanged-hf'),
                                NCCL_ALGO='Ring', CUDA_VISIBLE_DEVICES='',
                                PYTHONDONTWRITEBYTECODE='1')

    def execute(self, **updates):
        environment = dict(self.environment, **updates)
        script = '''
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMNI_VARIANT="$2" OMNI_RUNTIME_ROOT="$3" TMPDIR="$4"
export HOME='/root' FLASHINFER_WORKSPACE_BASE='/root/.cache/flashinfer'
export HF_HOME="$5" NCCL_ALGO='Ring'
source "$1"
python -I -S -c 'import json,os; print(json.dumps({key:os.environ[key] for key in ("HOME","FLASHINFER_WORKSPACE_BASE","TMPDIR","HF_HOME","NCCL_ALGO","CUDA_VISIBLE_DEVICES")}))'
'''
        return subprocess.run(
            ['bash', '-c', script, 'nft-cache-test', str(HELPER), environment['OMNI_VARIANT'],
             environment['OMNI_RUNTIME_ROOT'], environment['TMPDIR'], environment['HF_HOME']],
            env=environment, capture_output=True, text=True, timeout=25,
        )

    def test_exports_owned_home_workspace_and_preserves_tmp_assets_transport(self):
        parent_home = os.environ.get('HOME')
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(values['HOME'], str(self.runtime / 'cachehome'))
        self.assertEqual(values['FLASHINFER_WORKSPACE_BASE'], str(self.runtime / 'cache/flashinfer'))
        for key in ('TMPDIR', 'HF_HOME', 'NCCL_ALGO', 'CUDA_VISIBLE_DEVICES'):
            self.assertEqual(values[key], self.environment[key])
        for path in ('cachehome', 'cache/flashinfer', 'tmp'):
            self.assertTrue((self.runtime / path).is_dir())
        self.assertEqual(os.environ.get('HOME'), parent_home)

    def test_u2_uses_same_owned_child_cache_roots(self):
        result = self.execute(OMNI_VARIANT='NFT_FULL32_U2')
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(values['HOME'], str(self.runtime / 'cachehome'))
        self.assertEqual(values['FLASHINFER_WORKSPACE_BASE'], str(self.runtime / 'cache/flashinfer'))
        self.assertEqual(values['TMPDIR'], str(self.runtime / 'tmp'))
        self.assertEqual(values['NCCL_ALGO'], 'Ring')
        self.assertEqual(values['CUDA_VISIBLE_DEVICES'], '')

    def test_missing_runtime_fails_before_creating_cache_directories(self):
        for runtime in ('', str(self.runtime / 'missing'), 'relative'):
            with self.subTest(runtime=runtime):
                result = self.execute(OMNI_RUNTIME_ROOT=runtime)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.runtime / 'cachehome').exists())

    def test_unsafe_runtime_or_temporary_root_fails_closed(self):
        for updates in ({'OMNI_RUNTIME_ROOT': '/root'},
                        {'TMPDIR': '/tmp'}, {'TMPDIR': ''}, {'TMPDIR': 'relative'},
                        {'TMPDIR': str(self.runtime / '../escape')},
                        {'TMPDIR': str(self.runtime / 'tmp/../still-owned')},
                        {'OMNI_VARIANT': 'ART_FULL32'}):
            with self.subTest(updates=updates):
                result = self.execute(**updates)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.runtime / 'cachehome').exists())

    def test_cache_and_tmp_symlinks_rejected_without_external_writes(self):
        outside = self.runtime / 'outside'
        outside.mkdir()
        for relative in ('cachehome', 'cache'):
            with self.subTest(relative=relative):
                link = self.runtime / relative
                link.symlink_to(outside, target_is_directory=True)
                result = self.execute()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(outside.iterdir()), [])
                link.unlink()

    def test_existing_ray_short_symlink_is_preserved_when_target_is_owned(self):
        runtime = self.runtime / 'node-runtime'
        temporary = runtime / 'tmp/ray'
        temporary.mkdir(parents=True)
        alias = self.runtime / 'ray-short'
        alias.symlink_to(temporary, target_is_directory=True)
        result = self.execute(OMNI_RUNTIME_ROOT=str(runtime), TMPDIR=str(alias))
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(values['TMPDIR'], str(alias))
        self.assertEqual(values['HOME'], str(runtime / 'cachehome'))
        self.assertEqual(alias.resolve(), temporary)

    def test_ray_short_symlink_escaping_runtime_is_rejected(self):
        runtime = self.runtime / 'node-runtime'
        runtime.mkdir()
        outside = self.runtime / 'not-node-owned'
        outside.mkdir()
        alias = self.runtime / 'ray-short'
        alias.symlink_to(outside, target_is_directory=True)
        result = self.execute(OMNI_RUNTIME_ROOT=str(runtime), TMPDIR=str(alias))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((runtime / 'cachehome').exists())

    def test_cache_mkdir_failure_is_not_reported_as_success(self):
        (self.runtime / 'cache').write_text('not a directory')
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_stdlib_only_validation_and_nft_only_preimport_wiring(self):
        helper = HELPER.read_text()
        embedded = helper.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        imports = set()
        for node in ast.walk(ast.parse(embedded)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {'os', 'pathlib', 'shlex', 'tempfile'})
        self.assertIn(str(EXTENSION), helper)
        self.assertIn('/m2v_intern2/zhenglin/omni_art_ablation_20261016/extensions/nft_full32', helper)
        self.assertIn('timeout --foreground 15s python -I -S -', helper)
        payload = (REPO / 'scripts/omni_art_joint32/node_payload.sh').read_text()
        call = '    source "$control/nft_cache_env.sh"'
        self.assertEqual(payload.count(call), 1)
        self.assertIn('if [[ "${OMNI_VARIANT:-}" == NFT_FULL32_U* ]]; then\n' + call, payload)
        self.assertLess(payload.index('source "$control/runtime_env.sh"'), payload.index(call))
        self.assertLess(payload.index('export TMPDIR="$RAY_TMPDIR"'), payload.index(call))
        self.assertLess(payload.index(call), payload.index('python "$control/transport.py"'))
        self.assertLess(payload.index(call), payload.index('ray start'))


if __name__ == '__main__':
    unittest.main()
