import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import diffusers
import pytest
import torch

source_path = Path(__file__).resolve().parents[4] / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
source_tree = ast.parse(source_path.read_text())
loader_function = next(node for node in source_tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_load_diffusers_pretrained")
loader_namespace = {"json": json, "Path": Path}
exec(compile(ast.Module(body=[loader_function], type_ignores=[]), str(source_path), "exec"), loader_namespace)
_load_diffusers_pretrained = loader_namespace["_load_diffusers_pretrained"]


def write_model(root, class_name="LTX2VideoTransformer3DModel", auto_map=None):
    component = root / "transformer"
    component.mkdir()
    config = {"_class_name": class_name}
    if auto_map is not None:
        config["auto_map"] = auto_map
    (component / "config.json").write_text(json.dumps(config))
    (root / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", class_name]}))


def test_local_ltx_exact_class_kwargs_and_identity_match_auto_model(tmp_path, monkeypatch):
    write_model(tmp_path)
    target_class = diffusers.LTX2VideoTransformer3DModel
    loader = Mock(side_effect=lambda *arguments, **keywords: SimpleNamespace())
    monkeypatch.setattr(target_class, "from_pretrained", loader)
    baseline = diffusers.AutoModel.from_pretrained(str(tmp_path), torch_dtype=torch.bfloat16,
                                                 trust_remote_code=False, subfolder="transformer")
    baseline_call = loader.call_args
    optimized = _load_diffusers_pretrained(str(tmp_path), torch_dtype=torch.bfloat16,
                                          trust_remote_code=False, subfolder="transformer")
    assert loader.call_args == baseline_call
    assert optimized._diffusers_load_id == baseline._diffusers_load_id


def test_local_ltx_does_not_scan_optional_class_candidates(tmp_path, monkeypatch):
    write_model(tmp_path)
    target_class = diffusers.LTX2VideoTransformer3DModel
    monkeypatch.setattr(target_class, "from_pretrained", Mock(return_value=SimpleNamespace()))
    import diffusers.pipelines.pipeline_loading_utils as loading_utils
    monkeypatch.setattr(loading_utils, "get_class_obj_and_candidates", Mock(side_effect=AssertionError("scan")))
    _load_diffusers_pretrained(str(tmp_path), torch_dtype=torch.bfloat16,
                              trust_remote_code=False, subfolder="transformer")


@pytest.mark.parametrize("case", ["other_class", "remote_code", "auto_map", "index_mismatch"])
def test_nonlocal_or_custom_models_keep_auto_model_path(tmp_path, monkeypatch, case):
    write_model(tmp_path, class_name="OtherModel" if case == "other_class" else "LTX2VideoTransformer3DModel",
                auto_map={"AutoModel": "custom.Model"} if case == "auto_map" else None)
    if case == "index_mismatch":
        (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["another", "OtherModel"]}))
    loader = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(diffusers.AutoModel, "from_pretrained", loader)
    expected = dict(torch_dtype=torch.bfloat16, trust_remote_code=case == "remote_code", subfolder="transformer")
    assert _load_diffusers_pretrained(str(tmp_path), **expected) is loader.return_value
    loader.assert_called_once_with(str(tmp_path), **expected)
