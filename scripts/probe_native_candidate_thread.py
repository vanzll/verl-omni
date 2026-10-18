"""Locate optional-native imports in the actual Diffusers auto-loader path."""
import faulthandler
import json
import os
import socket
import sys
import threading
import time


def phase(name, **extra):
    print(json.dumps({'host': socket.gethostname(), 'pid': os.getpid(), 'unix': time.time(),
                      'stage': name, **extra}), flush=True)


faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
phase('import_accelerate')
from accelerate import init_empty_weights
phase('import_diffusers_candidate_helper')
from diffusers.pipelines.pipeline_loading_utils import ALL_IMPORTABLE_CLASSES, get_class_obj_and_candidates
phase('start_background_thread')
results, errors = [], []


def test():
    try:
        phase('background_meta_enter')
        with init_empty_weights():
            phase('background_actual_candidate_enumeration')
            cls, candidates = get_class_obj_and_candidates(
                library_name='diffusers', class_name='LTX2VideoTransformer3DModel',
                importable_classes=ALL_IMPORTABLE_CLASSES, pipelines=None, is_pipeline_module=False)
        assert cls.__name__ == 'LTX2VideoTransformer3DModel'
        import onnxruntime
        results.append({'class': cls.__module__ + '.' + cls.__name__,
                        'onnx_version': onnxruntime.__version__, 'candidate_names': list(candidates)})
        phase('background_actual_candidate_enumeration_passed')
    except BaseException as error:
        errors.append({'type': type(error).__name__, 'message': str(error)})


worker = threading.Thread(target=test)
worker.start()
worker.join(120)
assert not worker.is_alive() and not errors, (errors, worker.is_alive())
faulthandler.cancel_dump_traceback_later()
phase('native_AutoModel_candidate_resolution_thread_meta_passed', results=results,
      scope='actual background-thread/meta loader enumeration; no GPU/model construction or sustained-health guarantee')
