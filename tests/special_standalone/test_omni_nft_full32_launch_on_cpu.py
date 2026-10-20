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

"""CPU launch contracts; no allocation, lease, Ray or GPU operations."""

import copy
import ast
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/omni_art_joint32"
CAMPAIGN = Path("/m2v_intern/zhenglin/omni_art_ablation_20261016")
WRAPPER = REPO / "examples/omninft_trainer/ltx2/run_ltx2_t2av_nft_full32_u1.sh"
PREFIX = "LTX23-OMNINFT-NOKL-FULL32-U1-PROMPTSTD-P32K8-S42-"


def load(name):
    spec = importlib.util.spec_from_file_location("nft_launch_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load("control")
validator = load("validate_config")
transport = load("transport")


def select(config, dotted):
    for component in dotted.split("."):
        config = config[component]
    return config


def method(path, class_name, name):
    tree = ast.parse((REPO / path).read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    return function


def compile_method(function, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(tree, "actual-worker-contract", "exec"), namespace)
    return namespace[function.name]


class NFTLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.shim = cls.directory / "bash_env"
        cls.shim.write_text(
            'source() { case "$1" in */my_bashrc.sh|*/runtime_env.sh|*/use_source_snapshot.sh) '
            'return 0;; *) builtin source "$@";; esac; }\n'
            'python() { if [[ "$1" == */transport.py && "${2:-}" == hydra ]]; then '
            'printf "%s\\n" "$NFT_TEST_TRANSPORT_ARGS"; else command python "$@"; fi; }\n'
        )
        cls.configs = {}
        for micro in (1, 2, 4, 8):
            output = cls.directory / f"resolved_{micro}.yaml"
            environment = cls.environment(micro, output)
            result = subprocess.run(["bash", str(WRAPPER)], env=environment,
                                    text=True, capture_output=True, timeout=40)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            cls.configs[micro] = OmegaConf.to_container(OmegaConf.load(output), resolve=True)

    @classmethod
    def environment(cls, micro=4, output=None):
        return dict(os.environ, BASH_ENV=str(cls.shim), OMNI_CONFIG_ONLY="1",
                    OMNI_CONFIG_OUTPUT=str(output or cls.directory / "unused.yaml"),
                    OMNI_CAMPAIGN_ROOT=str(CAMPAIGN), OMNI_CONTROL_DIR=str(SCRIPTS),
                    OMNI_RUNTIME_ROOT=str(cls.directory), SOURCE_CHECKOUT_DIR=str(REPO),
                    SOURCE_SNAPSHOT_SHA256="a" * 64, SOURCE_FILES_MANIFEST_SHA256="b" * 64,
                    OUTPUT_DIR=str(cls.directory / "output"), RUN_NAME=PREFIX + "cpu",
                    WANDB_RUN_ID="fresh-cpu-only", WANDB_RESUME="never", OMNI_VARIANT="NFT_FULL32_U1",
                    OMNI_FULL_FINETUNE="1", RAY_ADDRESS="127.0.0.1:65530", PYTHONPATH=str(REPO),
                    OMNI_UPDATE_MICROBATCH=str(micro), PYTHONDONTWRITEBYTECODE="1",
                    NFT_TEST_TRANSPORT_ARGS="\n".join(
                        "++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(value)
                        for key, value in transport.COMMON_ENVIRONMENT.items()))

    def test_actual_hydra_configuration_and_all_microbatch_options(self):
        for micro, config in self.configs.items():
            with self.subTest(micro=micro):
                validator.validate(config, micro, variant="nft_full32_u1")
                validator.validate(config, micro)
                contract = control.batch_contract(micro, variant="nft_full32_u1")
                self.assertEqual(contract["gradient_accumulation"], 8 // micro)
                self.assertEqual(contract["DP"] * micro * contract["gradient_accumulation"], 256)
                self.assertEqual(contract["update_samples"] * contract["updates_per_rollout"], 256)
                self.assertEqual(contract["global_minibatch_prompts"], 32)
                self.assertEqual(contract["expanded_global_minibatch"], 256)
                self.assertEqual(contract["training_timesteps"], 20)
                self.assertEqual(contract["policy_states"], ["current", "ema_old"])
                self.assertFalse(contract["base_reference"])
                self.assertEqual(contract["fsdp_size"], 32)
                self.assertIsNone(contract["target_updates"])
                self.assertIsNone(contract["target_rollouts"])

    def test_u2_recipe_composes_two_disjoint_minibatches_with_same_ema_schedule(self):
        for micro in (1, 2, 4):
            output = self.directory / f"resolved_u2_{micro}.yaml"
            environment = self.environment(micro, output)
            environment.update(OMNI_VARIANT="NFT_FULL32_U2", RUN_NAME=PREFIX.replace("-U1-", "-U2-") + "cpu")
            wrapper = REPO / control.launch_recipe("nft_full32_u2")
            subprocess.run(["bash", str(wrapper)], env=environment, capture_output=True, text=True,
                           timeout=40, check=True)
            config = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
            validator.validate(config, micro, variant="nft_full32_u2")
            contract = control.batch_contract(micro, variant="nft_full32_u2")
            self.assertEqual(contract["updates_per_rollout"], 2)
            self.assertEqual(contract["update_samples"], 128)
            self.assertEqual(contract["gradient_accumulation"], 4 // micro)
            self.assertEqual(contract["update_samples"] * contract["updates_per_rollout"], 256)
            self.assertEqual(select(config, "algorithm.old_policy_decay_schedule"), "linear_to_0_5")
            self.assertEqual(select(config, "algorithm.old_policy_update_interval"), 1)
            self.assertEqual(select(config, "actor_rollout_ref.actor.ppo_epochs"), 1)
            self.assertEqual(select(config, "ray_kwargs.ray_init.runtime_env.env_vars.OMNI_VARIANT"),
                             "NFT_FULL32_U2")
        with self.assertRaises(ValueError):
            control.batch_contract(8, variant="nft_full32_u2")

    def test_routed_full20_art_recipe_resolves_all_scientific_switches(self):
        output = self.directory / "resolved_art_routed20.yaml"
        environment = self.environment(4, output)
        environment.update(OMNI_VARIANT="ART_FULL32_ROUTED",
                           RUN_NAME="LTX23-ART-ROUTED20-FULL32-P32K8-U2-S42-cpu")
        wrapper = REPO / control.launch_recipe("art_full32_routed")
        result = subprocess.run(["bash", str(wrapper)], env=environment, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
        validator.validate(config, 4, variant="art_full32_routed")
        self.assertFalse(select(config, "algorithm.art_exclude_final_step"))
        self.assertTrue(select(config, "algorithm.art_prompt_group_std"))
        self.assertTrue(select(config, "algorithm.art_first_update_mask_exempt"))
        self.assertEqual(select(config, "actor_rollout_ref.actor.diffusion_loss.art_video_weight"), 1.0)
        self.assertEqual(select(config, "actor_rollout_ref.actor.diffusion_loss.art_audio_weight"), 1.0)
        self.assertEqual(select(config, "ray_kwargs.ray_init.runtime_env.env_vars.OMNI_VARIANT"),
                         "ART_FULL32_ROUTED")

    def test_validate_rejects_art_and_scientific_drift(self):
        changes = {
            "algorithm.global_std": True,
            "algorithm.art_exclude_final_step": True,
            "algorithm.art_joint_latent": True,
            "algorithm.old_policy_decay_schedule": "copy",
            "algorithm.old_policy_decay": 0.5,
            "algorithm.timestep_fraction": 0.95,
            "actor_rollout_ref.model.algorithm": "diffusion_art",
            "actor_rollout_ref.model.full_finetune": False,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
            "actor_rollout_ref.actor.diffusion_loss.prompt_group_std": False,
            "actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef": 1e-4,
            "actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef": 1e-4,
            "actor_rollout_ref.actor.diffusion_loss.ref_kl_coef": 1e-4,
            "actor_rollout_ref.actor.diffusion_loss.mix_beta": 0.5,
            "actor_rollout_ref.actor.use_kl_loss": True,
            "actor_rollout_ref.actor.fsdp_config.fsdp_size": 16,
            "actor_rollout_ref.rollout.rollout_adapter": "default",
            "actor_rollout_ref.rollout.pipeline.num_inference_steps": 19,
            "reward.share_rpc_batches": False,
            "reward.aggregation": "weighted_sum",
            "trainer.expected_optimizer_updates_per_rollout": 2,
            "trainer.run_until_stopped": False,
            "trainer.total_training_steps": 5,
            "trainer.save_freq": 10,
            "trainer.experiment_name": "LTX23-ART-A12-FULL32-P32K8-U2-S42-cpu",
        }
        for dotted, value in changes.items():
            with self.subTest(key=dotted):
                config = copy.deepcopy(self.configs[4])
                parent, key = dotted.rsplit(".", 1)
                select(config, parent)[key] = value
                with self.assertRaises(ValueError):
                    validator.validate(config, 4, variant="nft_full32_u1")

    def test_f07_scientific_anchor_and_art_infrastructure(self):
        anchors = CAMPAIGN / "extensions/nft_full32/baseline_wandb"
        baseline = json.loads((anchors / "f07cd5db/config.json").read_text())
        art = json.loads((anchors / "266a1574205eda99/config.json").read_text())
        config = self.configs[4]
        for dotted in (
            "algorithm.adv_mode", "algorithm.old_policy_decay_schedule", "algorithm.old_policy_update_interval",
            "algorithm.timestep_fraction", "actor_rollout_ref.model.algorithm", "actor_rollout_ref.model.path",
            "actor_rollout_ref.model.policy_state_adapters", "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.betas", "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.optim.clip_grad", "actor_rollout_ref.actor.diffusion_loss.mix_beta",
            "actor_rollout_ref.actor.diffusion_loss.adv_clip_max", "actor_rollout_ref.rollout.rollout_adapter",
            "actor_rollout_ref.model.pipeline", "actor_rollout_ref.rollout.pipeline",
            "actor_rollout_ref.rollout.val_kwargs.pipeline", "data", "reward.reward_functions",
        ):
            with self.subTest(anchor="nft", key=dotted):
                actual, expected = select(config, dotted), select(baseline, dotted)
                if dotted == "data":
                    expected = dict(expected, apply_chat_template_kwargs={})
                self.assertEqual(actual, expected)
        for dotted in (
            "reward.models", "reward.num_workers", "reward.profile_dispatch", "reward.share_rpc_batches",
            "reward.shared_rpc_cache_batches", "reward.rpc_excluded_tensor_keys",
            "actor_rollout_ref.actor.fsdp_config", "actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation",
            "actor_rollout_ref.rollout.engine_kwargs", "actor_rollout_ref.rollout.agent",
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
        ):
            with self.subTest(anchor="art", key=dotted):
                self.assertEqual(select(config, dotted), select(art, dotted))

    def test_explicit_scientific_difference_whitelist(self):
        anchors = CAMPAIGN / "extensions/nft_full32/baseline_wandb"
        baseline = json.loads((anchors / "f07cd5db/config.json").read_text())
        infra = json.loads((anchors / "266a1574205eda99/config.json").read_text())
        report = validator.compare_nft_anchors(self.configs[4], baseline, infra, 4)
        self.assertEqual(report["unexpected"], {})
        self.assertEqual(report["changes"]["actor_rollout_ref.rollout.algo.noise_level"]["category"],
                         "explicit_ode_zero_noise_requirement")
        for dotted, value in {
            "actor_rollout_ref.actor.optim.lr": 1e-5,
            "actor_rollout_ref.actor.optim.betas": [0.9, 0.95],
            "actor_rollout_ref.actor.optim.lr_scheduler_type": "cosine",
            "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.1,
            "algorithm.old_policy_update_interval": 2,
            "algorithm.old_policy_decay_schedule": "copy",
            "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps": 20,
            "data.shuffle": False,
            "data.seed": 43,
            "reward.reward_functions.hpsv3.score_cap": 10.0,
            "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_use_tiling": False,
        }.items():
            with self.subTest(key=dotted):
                config = copy.deepcopy(self.configs[4])
                parent, key = dotted.rsplit(".", 1)
                select(config, parent)[key] = value
                with self.assertRaisesRegex(ValueError, "unexpected NFT"):
                    validator.compare_nft_anchors(config, baseline, infra, 4)

    def test_legacy_art_defaults_and_full32_contracts(self):
        for full in (False, True):
            for micro in (1, 2, 4):
                contract = control.batch_contract(micro, full=full)
                self.assertEqual(contract["updates_per_rollout"], 2)
                self.assertEqual(contract["update_samples"], 128)
                self.assertEqual(contract["gradient_accumulation"], 4 // micro)
                self.assertEqual(contract["fsdp_size"], 32 if full else 1)
                expected = validator.expected(micro, full=full)
                self.assertTrue(expected["algorithm.art_joint_latent"])
                self.assertTrue(expected["algorithm.art_exclude_final_step"])
                self.assertEqual(expected["actor_rollout_ref.actor.ppo_mini_batch_size"], 16)
                self.assertEqual(expected["trainer.expected_optimizer_updates_per_rollout"], 2)
        for invalid in (0, 3, 16):
            with self.assertRaises(ValueError):
                control.batch_contract(invalid, variant="nft_full32_u1")
        with self.assertRaises(ValueError):
            control.batch_contract(8)
        with self.assertRaises(ValueError):
            control.launch_recipe("typo")

    def test_actual_actor_post_init_trainer_expansion_worker_split_and_timestep_denominator(self):
        path = "verl_omni/workers/config/diffusion/actor.py"
        classes = []
        for name, bases in (("DiffusionActorConfig", []),
                            ("FSDPDiffusionActorConfig", [ast.Name(id="DiffusionActorConfig", ctx=ast.Load())])):
            classes.append(ast.ClassDef(name=name, bases=bases, keywords=[], decorator_list=[],
                                       body=[method(path, name, "__post_init__")]))
        namespace = {"MISSING": "???"}
        tree = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
        exec(compile(tree, "actual-actor-post-init", "exec"), namespace)
        actor = namespace["FSDPDiffusionActorConfig"]()
        actor.__dict__.update(copy.deepcopy(self.configs[4]["actor_rollout_ref"]["actor"]))
        actor.fsdp_config = SimpleNamespace(**actor.fsdp_config)
        actor.__post_init__()
        self.assertEqual(actor.rollout_n, 8)
        self.assertEqual(actor.ppo_mini_batch_size, 32)
        self.assertEqual(actor.engine.strategy, "fsdp2")

        class Rows:
            def __init__(self, rows, metadata=None):
                self.rows = rows
                self.metadata = dict(metadata or {})
                self.meta_info = {}
                self.shape = (len(rows),)

            def __contains__(self, key):
                return False

            def __getitem__(self, key):
                return SimpleNamespace(shape=(len(self.rows), 20))

            def to(self, device):
                return self

        assignments = []
        iterator_calls = []

        def assign(data, **values):
            data.metadata.update(values)
            assignments.append(values)

        def iterator(data, mini_batch_size, epochs, seed, dataloader_kwargs):
            iterator_calls.append((mini_batch_size, epochs, seed, dataloader_kwargs))
            for epoch in range(epochs):
                for offset in range(0, len(data.rows), mini_batch_size):
                    yield Rows(data.rows[offset:offset + mini_batch_size], data.metadata)

        tensors = SimpleNamespace(
            assign_non_tensor=assign,
            pop=lambda data, key, default=None: data.metadata.pop(key, default),
            get=lambda data, key: data[key],
            get_non_tensor_data=lambda data, key, default=None: data.metadata.get(key, default),
            make_iterator=iterator,
        )
        worker_train = compile_method(method("verl_omni/workers/engine_workers.py", "TrainingWorker", "train_mini_batch"), {
            "tu": tensors, "maybe_fix_3d_position_ids": lambda data: None, "NonTensorData": lambda value: value,
            "Timer": lambda **kwargs: nullcontext(),
        })
        calls = []
        backward_steps = []
        current_micro = 4

        def micro_batches(data, **kwargs):
            return ([Rows(data.rows[offset:offset + current_micro], data.metadata)
                     for offset in range(0, len(data.rows), current_micro)], None)

        engine_train = compile_method(method("verl_omni/workers/engine/fsdp/diffusers_impl.py",
                                             "DiffusersFSDPEngine", "_run_forward_backward_batch"), {
            "tu": tensors, "prepare_micro_batches": micro_batches, "nullcontext": nullcontext,
            "get_device_id": lambda: "cpu",
        })

        def update(global_data):
            for rank in range(32):
                local = Rows(global_data.rows[rank * 8:(rank + 1) * 8], global_data.metadata)

                def forward(data, step, **kwargs):
                    denominator = data.metadata["gradient_accumulation_steps"]
                    loss = SimpleNamespace(backward=lambda: backward_steps.append((tuple(data.rows), step, denominator)))
                    return loss, {"metrics": {}, "loss": 0.0}

                engine = SimpleNamespace(
                    get_data_parallel_size=lambda: 32, get_data_parallel_rank=lambda: rank,
                    train_mode=lambda **kwargs: nullcontext(), is_mp_src_rank_with_outputs=lambda: False,
                    full_finetune=True, ulysses_sequence_parallel_size=1,
                    _validate_full_cached_window=lambda data, timesteps_key: None,
                    get_data_parallel_group=lambda: None, forward_step=forward,
                    postprocess_batch_func=lambda **kwargs: kwargs,
                )

                def train_batch(data):
                    calls.append((tuple(data.rows), data.metadata["update_lr_scheduler"]))
                    engine_train(engine, data, None, False, timesteps_key="all_timesteps")
                    return {}

                worker_train(SimpleNamespace(engine=engine, train_batch=train_batch), local)
            return {"metrics": {}}

        trainer_update = compile_method(method("verl_omni/trainer/diffusion/ray_diffusion_trainer.py",
                                               "DirectPreferenceRayTrainer", "_update_actor"), {
            "OmegaConf": OmegaConf, "tu": tensors,
            "_to_diffusion_worker_tensordict": lambda data: data,
            "embeds_padding_2_no_padding": lambda data: data,
            "get_diffusion_loss_fn": lambda mode: SimpleNamespace(requires_whole_groups=False),
            "rename_dict": lambda data, prefix: {prefix + key: value for key, value in data.items()},
            "DataProto": SimpleNamespace(from_single_dict=lambda **kwargs: kwargs),
        })
        for updates, current_micro in [(u, m) for u in (1, 2) for m in (1, 2, 4, 8) if m <= 8 // u]:
            calls.clear()
            backward_steps.clear()
            iterator_calls.clear()
            config = OmegaConf.create(self.configs[current_micro])
            config.actor_rollout_ref.actor.ppo_mini_batch_size = 32 // updates
            trainer = SimpleNamespace(config=config, actor_rollout_wg=SimpleNamespace(update_actor=update),
                                      _assert_campaign_worker_updates=lambda output: None)
            trainer_update(trainer, Rows(list(range(256))))
            self.assertEqual(len(calls), 32 * updates)
            self.assertEqual([sample for rows, scheduler in calls for sample in rows], list(range(256)))
            self.assertEqual([scheduler for rows, scheduler in calls], ([True] if updates == 1 else [False, True]) * 32)
            self.assertTrue(all(size == 8 // updates and epochs == 1 and kwargs == {"shuffle": False}
                                for size, epochs, seed, kwargs in iterator_calls))
            self.assertEqual([seed for size, epochs, seed, kwargs in iterator_calls], list(range(42, 74)))
            self.assertEqual(len(backward_steps), 32 * (8 // current_micro) * 20)
            self.assertEqual({denominator for rows, step, denominator in backward_steps}, {20 * (8 // updates // current_micro)})
            self.assertEqual({step for rows, step, denominator in backward_steps}, set(range(20)))
        self.assertTrue(any(values.get("global_batch_size") == 256 and values.get("mini_batch_size") == 256
                            for values in assignments))

    def test_wrapper_rejects_wrong_variant_resume_identity_and_extra_arguments(self):
        for changes, arguments in (
            ({"OMNI_VARIANT": "ART_FULL32"}, []), ({"OMNI_FULL_FINETUNE": "0"}, []),
            ({"OMNI_UPDATE_MICROBATCH": "3"}, []), ({"WANDB_RESUME": "must"}, []),
            ({"RUN_NAME": "LTX23-ART-A12-FULL32-P32K8-U2-S42-cpu"}, []), ({}, ["trainer.total_training_steps=5"]),
        ):
            with self.subTest(changes=changes, arguments=arguments):
                environment = self.environment()
                environment.update(changes)
                result = subprocess.run(["bash", str(WRAPPER), *arguments], env=environment,
                                        text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_training_argv_matches_config_only_and_preserves_failure(self):
        binary = self.directory / "bin"
        binary.mkdir(exist_ok=True)
        capture = self.directory / "argv.json"
        python = binary / "python3"
        python.write_text(
            f"#!{CAMPAIGN}/env/bin/python\nimport json,os,sys\n"
            "open(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\nsys.exit(17)\n"
        )
        python.chmod(0o755)
        environment = self.environment()
        environment.update(OMNI_CONFIG_ONLY="0", CAPTURE=str(capture), PATH=str(binary) + ":" + os.environ["PATH"],
                           OMNI_CAMPAIGN_ROOT=str(self.directory / "campaign"))
        result = subprocess.run(["bash", str(WRAPPER)], env=environment, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 17, result.stderr)
        argv = json.loads(capture.read_text())
        self.assertEqual(argv[:6], ["-m", "verl_omni.trainer.main_diffusion", "--config-path",
                                   str(REPO / "examples/omninft_trainer/ltx2"), "--config-name", "ltx2_omninft"])
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=argv[3], version_base=None):
            config = OmegaConf.to_container(compose(config_name=argv[5], overrides=argv[6:]), resolve=True)
        validator.validate(config, 4, variant="nft_full32_u1")

    def test_nft_source_manifest_requires_nft_files_and_rejects_mutation(self):
        required = [str(control.launch_recipe("nft_full32_u1")), "scripts/omni_art_joint32/control.py",
                    "examples/omninft_trainer/ltx2/ltx2_omninft.yaml",
                    "verl_omni/trainer/diffusion/diffusion_algos.py", "verl_omni/workers/engine/full_policy.py",
                    "verl_omni/workers/engine/full_nft_policy.py", "verl_omni/pipelines/ltx2_omni_nft/config.py"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in required:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            files = {name: control.digest(root / name) for name in required}
            manifest = root / "files.json"

            def verify():
                manifest.write_text(json.dumps(files))
                with patch.dict(os.environ, SOURCE_CHECKOUT_DIR=str(root), SOURCE_FILES_MANIFEST=str(manifest),
                                SOURCE_FILES_MANIFEST_SHA256=control.digest(manifest), OMNI_VARIANT="NFT_FULL32_U1"):
                    control.verify_source()

            verify()
            omitted = files.pop(required[0])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                verify()
            files[required[0]] = omitted
            (root / required[0]).write_text("changed after freeze")
            with self.assertRaisesRegex(RuntimeError, "source digest mismatch"):
                verify()

    def test_prepare_selects_nft_paths_environment_and_pinned_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            source = campaign / "extensions/nft_full32/source"
            scripts = source / "scripts/omni_art_joint32"
            scripts.mkdir(parents=True)
            recipe = source / control.launch_recipe("nft_full32_u1")
            config_path = source / "examples/omninft_trainer/ltx2/ltx2_omninft.yaml"
            files = [scripts / "control.py", recipe, config_path,
                     *[campaign / "scripts" / name for name in
                       ("gpu_lease_lib.sh", "runtime_env.sh", "check_assets.py", "inspect_gpu_node.py")],
                     campaign / "bin/gpu_free", campaign / "bin/gpu_hold",
                     campaign / "extensions/nft_full32/baseline_wandb/f07cd5db/config.json",
                     campaign / "extensions/nft_full32/baseline_wandb/266a1574205eda99/config.json",
                     campaign / "assets/vggsound/parquet/train.parquet", campaign / "assets/vggsound/parquet/test8.parquet"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            incident = campaign / "review.json"
            incident.write_text(json.dumps({"status": "reviewed", "notes": "controlled stop reviewed",
                                            "evidence_paths": [str(config_path)]}))
            manifest = campaign / "source-files.json"
            manifest.write_text("{}")
            args = SimpleNamespace(variant="nft_full32_u1", full_finetune=False, update_microbatch=4,
                                   runtime_root=None, fallback_runtime_root=None, incident_review=incident,
                                   source_root=source, source_manifest=manifest,
                                   source_manifest_sha256=control.digest(manifest), source_archive_sha256="a" * 64,
                                   wandb_entity="test-only")
            audits = [dict(hostname=f"host{index}", processes=[dict(pid=index + 1, start_ticks="1", cmd="holder")])
                      for index in range(4)]
            calls = []

            def run(command, **kwargs):
                calls.append((command, kwargs))
                if command[:2] == ["bash", "-c"]:
                    output = Path(command[-1])
                    OmegaConf.save(OmegaConf.create(self.configs[4]), output)
                return subprocess.CompletedProcess(command, 0, stdout="cpu fixture", stderr="")

            with patch.object(control, "CAMPAIGN", campaign), patch.object(control, "CONTROL", scripts), \
                    patch.object(control, "TRAIN_SHA", control.digest(files[-2])), \
                    patch.object(control, "VAL_SHA", control.digest(files[-1])), \
                    patch.object(control, "allocation"), patch.object(control, "audit", side_effect=audits), \
                    patch.object(control.socket, "gethostname", return_value="host0"), \
                    patch.object(control, "choose_runtime", side_effect=lambda primary, fallback: primary), \
                    patch.object(control, "choose_service_ports", return_value=(17001, 30001)), \
                    patch.object(control.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                control.prepare(args)
            outputs = list((campaign / "extensions/nft_full32/attempts").iterdir())
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            self.assertTrue(output.name.startswith(PREFIX))
            prepared = json.loads((output / "manifest.json").read_text())
            self.assertEqual(prepared["variant"], "nft_full32_u1")
            self.assertEqual(prepared["launch_recipe"], str(control.launch_recipe("nft_full32_u1")))
            self.assertEqual(prepared["contract"]["gradient_accumulation"], 2)
            self.assertEqual(prepared["runtime_root"], str(campaign / "extensions/nft_full32"))
            pinned = json.loads(Path(prepared["launch_manifest"]).read_text())
            self.assertIn(str(recipe), pinned)
            self.assertIn(str(config_path), pinned)
            self.assertFalse(any("diffusionart_trainer" in name for name in pinned))
            for node in prepared["nodes"]:
                rows = Path(node["env_file"]).read_text().splitlines()
                environment = dict(shlex.split(row.removeprefix("export "))[0].split("=", 1) for row in rows)
                self.assertEqual(environment["OMNI_VARIANT"], "NFT_FULL32_U1")
                self.assertEqual(environment["OMNI_FULL_FINETUNE"], "1")
                self.assertEqual(environment["OMNI_ART_CACHE_ACCEPTANCE"], "0")
                self.assertEqual(environment["WANDB_RESUME"], "never")
            self.assertEqual(calls[0][1]["env"]["OMNI_VARIANT"], "NFT_FULL32_U1")
            self.assertIn("nft_full32_u1", calls[-1][0])
            self.assertEqual(calls[-1][0][-2:], ["--report", str(output / "provenance/nft-config-diff.json")])
            control.pinned(prepared)
            recipe.write_text("mutated")
            with self.assertRaisesRegex(RuntimeError, "pinned runtime changed"):
                control.pinned(prepared)


if __name__ == "__main__":
    unittest.main()
