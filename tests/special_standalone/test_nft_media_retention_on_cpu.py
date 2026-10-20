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

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('nft_media_retention_test', ROOT / 'scripts/omni_art_joint32/media_retention.py')
RETENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETENTION)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name) / 'nft_full32'
        self.output = self.runtime / 'attempts/test-nft'
        (self.output / 'logs').mkdir(parents=True)
        self.manifest = {'runtime_root': str(self.runtime), 'attempt': self.output.name,
                         'output': str(self.output), 'variant': 'nft_full32_u1'}
        (self.output / 'manifest.json').write_text(json.dumps(self.manifest))
        self.write_versions(51)

    def write_versions(self, version, updates_per_rollout=1):
        rollout = version
        version *= updates_per_rollout
        for rank in range(32):
            row = {'event': 'nft_ema_update', 'rank': rank, 'successful_optimizer_updates': version,
                   'policy_version': version, 'old_version': version, 'ema_updates': rollout,
                   'decay': min(0.001 * rollout, 0.5)}
            (self.output / 'logs' / f'full32_rank_{rank}.jsonl').write_text(json.dumps(row) + '\n')

    def media(self, relative, old=True):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'owned test media')
        os.utime(path, (100 if old else 1000, 100 if old else 1000))
        return path

    def prune(self):
        return RETENTION.prune_completed_media(self.output, now=1000, allowed_runtime_roots=(self.runtime,))

    def test_only_old_optional_media_is_deleted_and_initial_eval_raw_weights_survive(self):
        initial = self.media('validation/0/0.mp4')
        old = self.media('validation/10/0.mp4')
        fallback = self.media('rollout_media/20/0.pt')
        raw = self.media('validation/10/0.jsonl')
        weights = self.media('checkpoints/10/0.pt')
        unique = self.media('validation/10/transformer.pt')
        recent = self.media('validation/40/0.mp4')
        active = self.media('validation/60/0.mp4')
        fresh = self.media('validation/10/1.mp4', old=False)
        result = self.prune()
        self.assertFalse(old.exists())
        self.assertFalse(fallback.exists())
        self.assertTrue(all(path.exists() for path in (initial, raw, weights, unique, recent, active, fresh)))
        self.assertEqual(result['minimum_committed_ema'], 51)
        self.assertEqual(len(result['deleted_files']), 2)

    def test_slow_or_missing_rank_prevents_unsafe_cleanup(self):
        target = self.media('validation/10/0.mp4')
        row = json.loads((self.output / 'logs/full32_rank_31.jsonl').read_text())
        for key in ('successful_optimizer_updates', 'policy_version', 'old_version', 'ema_updates'):
            row[key] = 5
        row['decay'] = 0.005
        (self.output / 'logs/full32_rank_31.jsonl').write_text(json.dumps(row) + '\n')
        self.assertEqual(self.prune()['minimum_committed_ema'], 5)
        self.assertTrue(target.exists())
        (self.output / 'logs/full32_rank_0.jsonl').unlink()
        self.assertIsNone(self.prune()['minimum_committed_ema'])
        self.assertTrue(target.exists())

    def test_symlink_escape_and_incomplete_commit_are_not_followed(self):
        outside = Path(self.temporary.name) / 'outside.mp4'
        outside.write_bytes(b'not task media')
        link = self.output / 'validation/10/0.mp4'
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        self.prune()
        self.assertTrue(link.is_symlink() and outside.exists())
        (self.output / 'logs/full32_rank_31.jsonl').write_text('{"event":"nft_ema_update"')
        self.assertIsNone(self.prune()['minimum_committed_ema'])

    def test_u2_retention_uses_completed_rollouts_and_requires_both_updates(self):
        self.manifest['variant'] = 'nft_full32_u2'
        (self.output / 'manifest.json').write_text(json.dumps(self.manifest))
        self.write_versions(51, updates_per_rollout=2)
        old = self.media('validation/30/0.mp4')
        retained = self.media('validation/40/0.mp4')
        initial = self.media('validation/0/0.mp4')
        result = self.prune()
        self.assertEqual(result['minimum_committed_ema'], 51)
        self.assertFalse(old.exists())
        self.assertTrue(retained.exists() and initial.exists())
        protected = self.media('validation/10/0.mp4')
        path = self.output / 'logs/full32_rank_31.jsonl'
        original = json.loads(path.read_text())
        for updates in ({'successful_optimizer_updates': 101, 'policy_version': 101, 'old_version': 101},
                        {'ema_updates': 102}, {'decay': 0.102}):
            with self.subTest(invalid_commit=updates):
                path.write_text(json.dumps(dict(original, **updates)) + '\n')
                self.assertIsNone(self.prune()['minimum_committed_ema'])
                self.assertTrue(protected.exists())
        original.update(successful_optimizer_updates=30, policy_version=30, old_version=30,
                        ema_updates=15, decay=0.015)
        path.write_text(json.dumps(original) + '\n')
        self.assertEqual(self.prune()['minimum_committed_ema'], 15)
        self.assertTrue(protected.exists())

    def test_wrong_attempt_or_non_nft_manifest_fails_closed(self):
        self.manifest['variant'] = 'art'
        (self.output / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self.prune()


if __name__ == '__main__':
    unittest.main()
