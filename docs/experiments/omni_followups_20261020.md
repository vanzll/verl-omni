# Omni Full32 follow-up experiments

Date: 2026-10-20. These are exploratory candidates, not completed training results.

The runtime baseline is the actual frozen source of NFT run `0ac35d8bcff5e26b`:
source manifest SHA256 `288d4d9ad9c1f3193480e01df9e62581a2c4330f412f7c63ce86d91c2005f4d6`,
archive SHA256 `87b5f7e0dec475eab1c889d1ff12d1898e0e736466dbd6a3d660eee958ca119c`.
It includes the native readiness-clock repair and the measured working RO0 transport.
Publishing its runtime files here does not retroactively assign a Git commit to that run.

| Setting | NFT U2 | ART routed20 |
| --- | --- | --- |
| `launch.sh prepare --variant` | `nft_full32_u2` | `art_full32_routed` |
| Environment variant | `NFT_FULL32_U2` | `ART_FULL32_ROUTED` |
| Scientific anchor | NFT `0ac35d8bcff5e26b` | ART `266a1574205eda99` |
| Rollout / optimizer batch | P32 K8 = 256 / 128 | P32 K8 = 256 / 128 |
| Updates / epochs per rollout | 2 / 1 | 2 / 1 |
| Default DP / microbatch / sample accumulation | 32 / 4 / 1 | 32 / 4 / 1 |
| Training timestep positions / backward calls per update | 20 / 20 at micro4 | 20 / 20 at micro4 |
| Frozen old | Separate sharded EMA module | Cached positive conditional velocity |
| Old refresh | Once after both updates | One logical snapshot after both updates |
| Policy/native version after rollout R | 2R | 2R |

NFT changes only the update split and the lifecycle checks needed to express it.
`ppo_mini_batch_size=16` expands by K8 to128 trajectories; `ppo_epochs=1`, no shuffle.
Contiguous DP dispatch gives one prompt's K8 per rank, split4+4: both windows cover P32K4.
No trajectory is replayed. At micro2/1 the sample accumulation is2/4; total backward
scaling divides by sample accumulation times20. EMA decay stays `min(.001*R,.5)`;
old is frozen across both updates and cannot be exported until the one EMA refresh.
Beta1, advantage clip5, prompt-group std, modality loss mean, zero reference KL and
reference-model omission remain unchanged. The original U1 recipe remains available.

ART keeps per-component prompt-K8 sample std (correction1, epsilon1e-4), then routes:
video = z(VideoAlign) +1.5 z(HPSv3) + z(DeSync);
audio = .5 z(AudioBox) + z(CLAP) + z(DeSync).
Each modality computes its own covariance, final unit-RMS direction, drift mask and
mean MSE. The objective is `(video_loss + audio_loss)/2`, including the original
inactive-row denominator. There are no cross-modality reward covariance terms.
Keep A1 first-window mask exemption, cached conditional targets and explicit
prompt-balanced4+4 splitting. `art_exclude_final_step=false` includes all20 original
positions; cache positions and versions are checked. All joint-latent flags and
`art_joint_reward_routing` are false. This combines routing/loss geometry and the
last-position change, so it cannot separately attribute an outcome to either factor.

Both retain the same Full transformer, FSDP32/DP32/SP1, rollout TP2, pretrained
revision, seed42, sampler, reward models/weights, geometry, LR/optimizer, eval cadence,
no checkpoints and fresh W&B identity. ART and NFT still have different objectives,
old-policy targets and restoration/masking semantics; routing does not make them
identical. Existing telemetry is reused, including actual optimizer/rollout counters,
EMA decay, modality diagnostics, gradients, losses, phase times and GPU cost.

Communication: preserve native NCCL2.29.7, the exact existing RDMA plugin, Ring/Simple,
P2P enabled, explicit RO0 and each host's eight local active RoCE HCA choices. The
current32-rank Ray collective gate and actual actor/native evidence remain required;
a fast synthetic gate alone does not establish healthy model execution. RO0 is a
working observed configuration, not an established unique root-cause explanation.
Reuse the existing generation-fenced lease launcher; no nested owner, unconditional
holder trap, Socket tensor fallback, library replacement or old-source overlay.

Run each arm only on an available authorized four-node allocation. Do not take over
an active lease or stop an existing run. The two32-GPU arms need disjoint allocations
for concurrent execution. Acceptance is at least five finite global optimizer updates
and continued progress in a later rollout, with all32 actors/native receivers verified.
Leave healthy training and its supervisor running after acceptance.

The third requested experiment remains planned: original Hunyuan Video ART/NFT,
matched sparse versus full timestep training. Freeze the sampling budget, loss
averaging and EMA/update clocks before designing that matrix; no Hunyuan job is
implemented or launched by this change. The existing campaign ledger is `track.md`
in the local `verl-omni-art-ablation-20261016` worktree and the actual remote campaign.

Validation: existing pinned Linux campaign environment, CUDA hidden:171 passed,
150 subtests passed,2 skipped. The two skips require explicit fixed-native-source
cache evidence paths. Production worker minibatch/timestep loops and native TP2
serialized32-receiver transaction oracles are exercised; these are CPU checks,
not fresh distributed GPU acceptance. Local shell syntax, fatal Python lint,
whitespace and ownership/lease regressions also passed. The tested runtime differs
from publication only by11 mandatory license headers, verified AST-identical.
