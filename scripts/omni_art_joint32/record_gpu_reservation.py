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

"""Persist allocation intervals separately from successful scientific progress."""
import json
import os
from pathlib import Path
import socket
import sys
import time

root = Path(os.environ['OUTPUT_DIR'])
campaign = Path(os.environ['OMNI_RUNTIME_ROOT'])
assert root.resolve().is_relative_to(campaign)
host = socket.gethostname().split('.')[0]
path = root / 'logs' / f'gpu_reservation_{host}.json'
operation = sys.argv[1]
if operation == 'mean-start':
    rows = [json.loads(p.read_text()) for p in (root / 'logs').glob('gpu_reservation_*.json')]
    expected = set(os.environ['OMNI_EXPECTED_IPS'].split(','))
    assert len(rows) == 4 and {row['ip'] for row in rows} == expected
    assert all(row['attempt'] == os.environ['OMNI_ATTEMPT'] and row['gpus'] == 8 for row in rows)
    assert all('stop_unix' not in row for row in rows)
    print(sum(row['start_unix'] for row in rows) / 4)
elif operation in {'start', 'stop'}:
    if operation == 'start':
        assert not path.exists(), 'reservation already recorded'
        record = {'host': host, 'ip': os.environ['OMNI_NODE_IP'], 'gpus': 8,
                  'attempt': os.environ['OMNI_ATTEMPT'], 'nonce': os.environ['OMNI_NONCE'],
                  'operation': os.environ.get('OMNI_OPERATION', 'training'), 'start_unix': time.time()}
    else:
        record = json.loads(path.read_text())
        assert record['attempt'] == os.environ['OMNI_ATTEMPT'] and record['nonce'] == os.environ['OMNI_NONCE']
        record['stop_unix'] = time.time()
        record['reserved_gpu_seconds'] = 8 * max(0, record['stop_unix'] - record['start_unix'])
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
else:
    raise ValueError(f'unsupported reservation operation {operation}')
