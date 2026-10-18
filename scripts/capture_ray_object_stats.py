"""Read bounded native Raylet memory/reference stats for one exact live formal pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from ray._private.grpc_utils import init_grpc_channel
from ray._private.internal_api import get_state_from_address
from ray.core.generated import node_manager_pb2, node_manager_pb2_grpc

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def capture(manifest_path):
    manifest_path = manifest_path.resolve()
    assert manifest_path.is_relative_to(ROOT / 'attempts')
    manifest = json.loads(manifest_path.read_text())
    output = manifest_path.parent
    owner = json.loads((output / 'coordinator_state.json').read_text())
    fields = Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(') ', 1)[1].split()
    assert fields[19] == str(owner['pid_start_ticks']) and fields[0] != 'Z'
    authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
    expected = {node['ip'] for node in manifest['nodes']}
    assert expected <= authorized
    head = manifest['nodes'][0]
    # Obtain the exact port from this attempt's whitelisted environment field.
    port = next(line.split('=', 1)[1] for line in Path(head['env_file']).read_text().splitlines()
                if line.startswith('export OMNI_RAY_PORT='))
    assert port.isdigit()
    address = f'{head["ip"]}:{port}'
    state = get_state_from_address(address)
    nodes = [node for node in state.node_table() if node['Alive']]
    assert len(nodes) == 2 and {node['NodeManagerAddress'] for node in nodes} == expected

    def query(node):
        channel = init_grpc_channel(f'{node["NodeManagerAddress"]}:{node["NodeManagerPort"]}')
        try:
            reply = node_manager_pb2_grpc.NodeManagerServiceStub(channel).GetNodeStats(
                node_manager_pb2.GetNodeStatsRequest(include_memory_info=True), timeout=12)
        finally:
            channel.close()
        objects, workers = [], []
        for worker in reply.core_workers_stats:
            workers.append({'pid': worker.pid, 'pending_tasks': worker.num_pending_tasks,
                            'executed_tasks': worker.num_executed_tasks,
                            'running_tasks': worker.num_running_tasks, 'task_queue_length': worker.task_queue_length,
                            'failed_arg_pinning_requests': worker.num_of_failed_arg_pinning_requests})
            for ref in worker.object_refs:
                if ref.object_size <= 0:
                    continue
                objects.append({'object_id': ref.object_id.hex(), 'object_size': ref.object_size,
                                'reference_worker_pid': worker.pid, 'call_site': ref.call_site,
                                'local_ref_count': ref.local_ref_count,
                                'submitted_task_ref_count': ref.submitted_task_ref_count,
                                'pinned_in_memory': ref.pinned_in_memory})
        sizes = {}
        for obj in objects:
            sizes[obj['object_id']] = max(sizes.get(obj['object_id'], 0), obj['object_size'])
        counters = {field.name: value for field, value in reply.store_stats.ListFields()
                    if field.message_type is None and field.type not in (field.TYPE_STRING, field.TYPE_BYTES)}
        return {'node_ip': node['NodeManagerAddress'], 'store_stats': counters,
                'object_reference_sizes_available': bool(sizes),
                'known_unique_referenced_objects': len(sizes), 'maximum_known_referenced_object_bytes': max(sizes.values(), default=None),
                'largest_100_reference_records': sorted(objects, key=lambda obj: -obj['object_size'])[:100],
                'workers': workers}

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(query, nodes))
    result = {'attempt': manifest['attempt'], 'observed_unix': time.time(), 'address': address,
              'launch_sha256': manifest['launch_manifest_sha256'], 'nodes': records,
              'scope': 'Read-only native Raylet GetNodeStats; no ray.init, tasks, GPU work or configuration changes. Referenced object sizes are a point-in-time sample, not a complete RPC size distribution or disk footprint.'}
    target = output / 'logs/ray_object_reference_stats.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    sample = {'attempt': manifest['attempt'], 'observed_unix': result['observed_unix'],
              'scope': 'Native Ray RPC counters, NOT scientific optimizer updates.',
              'nodes': [{key: value for key, value in node.items()
                         if key != 'largest_100_reference_records'} for node in records]}
    with (output / 'logs/ray_worker_execution_samples.jsonl').open('a') as stream:
        stream.write(json.dumps(sample) + '\n')
    for node in records:
        print(json.dumps({'attempt': manifest['attempt'], **{key: value for key, value in node.items()
                                                            if key not in ('largest_100_reference_records', 'workers')},
                          'largest_reference_examples': node['largest_100_reference_records'][:3]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    capture(parser.parse_args().manifest)
