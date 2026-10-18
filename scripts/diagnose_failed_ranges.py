"""Read small ranges and record safe error fields only, never signed URLs."""
from datetime import datetime, UTC
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

from huggingface_hub import get_hf_file_metadata
import requests

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
state = json.loads((root / 'logs/fast_ltx_state.json').read_text())
tasks = {t['name']: t for t in json.loads((root / 'assets/ltx-fast-manifest.json').read_text())}
records = []
for failure in state['failed'][:2]:
    task = tasks[failure['name']]
    for fresh in [False, True]:
        record = {'name': task['name'], 'cache_bust': fresh, 'unix': time.time()}
        try:
            url = task['url']
            if fresh:
                url += ('&' if '?' in url else '?') + 'cache_bust=' + str(int(time.time()))
            metadata = get_hf_file_metadata(url, timeout=10)
            assert metadata.size == task['size'] and metadata.etag == task['sha256']
            parameters = parse_qs(urlparse(metadata.location).query)
            if 'Expires' in parameters:
                record['signature_seconds_remaining'] = int(parameters['Expires'][0]) - time.time()
            elif 'X-Amz-Date' in parameters and 'X-Amz-Expires' in parameters:
                date = datetime.strptime(parameters['X-Amz-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=UTC)
                record['signature_seconds_remaining'] = date.timestamp() + int(parameters['X-Amz-Expires'][0]) - time.time()
            with requests.get(metadata.location, headers={'Range': 'bytes=0-1023', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(5, 10)) as response:
                record.update(status=response.status_code, content_range=response.headers.get('Content-Range'))
                if response.status_code == 206:
                    record['read_bytes'] = len(next(response.iter_content(1024)))
        except Exception as error:
            record['error_type'] = type(error).__name__
            response = getattr(error, 'response', None)
            if response is not None:
                record['http_status'] = response.status_code
        records.append(record)
        print(json.dumps(record), flush=True)
(root / 'logs/failed_range_diagnostic.json').write_text(json.dumps(records, indent=2) + '\n')
