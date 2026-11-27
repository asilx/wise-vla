# What data does WISE-VLA need?

For both stages, collect **action-labelled robot episodes**. At each control tick, save what the policy could observe, the instruction in effect, and the expert command issued from that state. WISE learns a chunk of future commands from each such observation. The Qwen recipe starts with a pretrained vision-language backbone. **Pre-training** here means teaching the WISE graph and action policy across tasks. **Post-training** means adapting that policy to a particular deployment with supervised demonstrations and expert corrections.

| Decision | Pre-training dataset | Post-training dataset |
|---|---|---|
| What to cover | Task families, object identities, layouts, instructions, camera conditions and starting poses. Include both goals when a relation changes which object to approach. | The deployment robot, tool, cameras, control rate and task distribution. Include typical runs and specific states where the deployed policy struggles. |
| Where the actions come from | Timestamped teleoperation, kinesthetic demonstrations converted to controller commands, or a simulator expert. Match the control schema across sources. | Target-task demonstrations, contiguous expert takeovers on learner rollouts, and retained nominal demonstrations. Keep the originating rollout and intervention time. |
| What graph to supply | Entities and directed relations available to the policy at each tick, with a consistent vocabulary. Annotate the active target when known. | Recreate the deployment graph source, including its latency, missing nodes and stale observations. Have an expert label the intended target separately when available. |
| What to hold out | Whole recording sessions, layouts, object sets or sites representative of the intended generalization claim. | Reserve target-domain sessions and paired nominal/disturbed trials before collecting corrections. Also retain a pre-training evaluation set to measure forgetting. |

## One observation becomes one training sample

Use a schema-version-1 manifest. Paths in it are relative to the dataset root. Every episode directory contains one row per command in each of the following files.

| File | Contents at index `t` |
|---|---|
| `states.npy` | Row of shape `[S]` in a numeric array of shape `[T, S]`. Record measured state immediately before dispatching command `t`. |
| `actions.npy` | Row of shape `[A]` in an array of shape `[T, A]`. Store the expert command in physical controller units. Use float32 for both arrays. |
| `observations.jsonl` | A nonempty instruction, at least one named RGB image, and an optional inline graph or relative path to a snapshot. Store image and graph files under this episode. |

```json
{"schema_version":1,"episodes":[{"id":"session07_pick03","path":"session07_pick03","split":"train"},{"id":"session11_pick02","path":"session11_pick02","split":"val"}]}
```

For example, this line in `session07_pick03/observations.jsonl` refers to files captured for tick 42. For a six-joint position-control recording, `states[42]` can contain the six measured joint positions and `actions[42]` the six commanded setpoints. Record the joint order, angle units and gripper convention in the dataset description.

```json
{"instruction":"Retrieve the tool from the bin containing it.","images":{"front":"front/000042.png","wrist":"wrist/000042.png"},"graph":"graphs/000042.json"}
```

The loader constructs an `[H, A]` target from `actions[t:t+H]` and masks the padded episode tail. An optional `action_dim_mask` on each observation marks valid channels **for that tick**. Every sample needs at least one valid action scalar. Store finite measured states. Store raw commands once and let the trainer normalize them. A video of a human pick can inform task design or graph annotation. To use it for the action loss, first recover synchronized robot states and robot-executable expert commands in the chosen control schema. [Exact graph fields, masks and feature order](data.md)

## Collecting broad pre-training data

1. **Fix the control contract.** Publish ordered state/action channels, units, frame, absolute or delta convention, tool and gripper semantics, camera names and calibration, and control frequency. Convert recordings to one agreed contract before pooling them. Six joint setpoints and six Cartesian deltas would teach incompatible actions even though their widths match.
2. **Align observations to decisions.** Use timestamps to select the latest camera frames and graph evidence available at command time. Record camera delay and controller latency. Replay a few complete episodes and check that the observed grasp, gripper command and change in measured state occur in the expected order.
3. **Vary factors independently.** For tool retrieval, swap which bin contains the tool while keeping the initial view and instruction fixed. Include the corresponding command trajectory for each goal. Also vary distractors, object positions and instruction phrasing. A dataset where the correct bin is always on the left offers little evidence that reading `inside` helps.
4. **Track graph and label provenance.** Keep stable entity IDs within each episode. Put known facts such as `part inside bin_2` in the input graph only when the online system can supply them, for example from an inventory event. Optional `target_id` names an existing node for the auxiliary target loss. Keep it out of online observations and keep label-derived `target` edges out of the inputs. Frames with no graph can still train the action path. For sparsely annotated predicates, set `model.graph.relation_loss_weight: 0` because reconstruction interprets unlisted predicates as absent.
5. **Audit coverage by episode.** Count layouts, independent sessions, task/goal pairs, valid commands per channel, and frames with graph and target labels. Record who or what produced each demonstration. Fit statistics only on training episodes. For a local LeRobot v3.0 recording, the importer can align RGB, state and action and join optional per-frame graph sidecars. [SO-100 import and annotation example](data.md#lerobot-v30-import)

For a concrete first recording, the repository has an import recipe for `lerobot/svla_so100_pickplace` with top and wrist RGB plus six state and action channels. This exercises the robot imitation path. To study knowledge-graph grounding on these recordings, add frame-aligned graph annotations and include tasks whose correct action depends on those relations. For later fine-tuning on an SO-100, start from a WISE policy pre-trained with the same six-channel robot control contract. The 2D Graph Reach checkpoint below has a different action space. [Pinned recording and commands](data.md#reproduce-the-so-100-example)

## Collecting target-domain post-training data

Start with a rollout log from the source checkpoint. Group failures by event, such as passing the correct bin, grasping empty space, or losing the target after occlusion. Keep a separate count of how often each event occurs. Use the log to choose where an expert should intervene.

At takeover, start a **new episode** with the actual learner-visited state, synchronized images and graph. Execute the expert's first correction, observe its successor, and continue recording the expert until completion or handback. End the episode at the boundary so an action chunk cannot span controllers. A failed learner command or a success flag alone supplies no expert action target. Keep failure outcome, source checkpoint, layout/session ID and takeover step as collection metadata.

Mix these recoveries with successful target-domain demonstrations and nominal examples from the original distribution. Track both episodes and valid timesteps per source. The trainer samples timesteps, so a long recovery contributes more examples than a short one. Choose the mix using validation rollouts. Keep paired goals, augmentations, learner runs and their derived recoveries in the same split. Assign a physical session or scene family **once across both stages** so a pre-training recording cannot reappear as a held-out post-training trial.

For Graph Reach, the following sequence creates both stages from the repository root. Use new dataset and stage output directories for a fresh experiment.

```bash
export OMP_NUM_THREADS=1
wise-vla-graph-reach generate --output runs/graph-reach/data
wise-vla-train --config configs/graph_reach.yaml
BASE=runs/graph-reach/target-conditioned/latest.pt
wise-vla-graph-reach collect-corrections --checkpoint "$BASE" \
  --output runs/graph-reach/corrections
wise-vla-train --config configs/graph_reach_posttrain.yaml --init-from "$BASE"
wise-vla-graph-reach evaluate --checkpoint runs/graph-reach/posttrained/latest.pt \
  --pairs 64 --seed 20261127
```

The example uses 32 training layouts with two goals per layout. Each learner rollout can add one expert recovery from a state off the nominal path. Eight separate validation layouts contain nominal demonstrations. These counts describe the synthetic example. For a physical dataset, increase coverage according to failures on held-out sessions and report the number of independent trials.

For a robot run, point `training.manifest` at your deployment demonstrations and give the stage a fresh `training.output_dir`. Keep the source model configuration and state/action semantics. You can adjust graph loss weights and masking probability for the new label coverage. `--init-from` loads the deployed EMA weights when available and inherits the source coordinate normalization. It starts a fresh optimizer, schedule and sampler. To continue an interrupted stage, use `--resume` with **that stage's** checkpoint. [Training options](training.md#starting-a-new-training-stage)

## Deciding whether the data helped

Evaluate the source policy, a copy fine-tuned on nominal demonstrations, and a copy fine-tuned on nominal demonstrations plus recoveries. Give the two fine-tunes the same initial weights, normalization, update budget, learning rate and sampling settings. Run all three on identical held-out layouts, goal assignments and noise seeds. Report successes over total episodes, wrong-target events, timeouts, and recovery successes over intervention opportunities. Check graph reliance by removing the task relation on the same layouts. Check retention on held-out source tasks. Offline action errors and target-node accuracy help locate a problem. Closed-loop trials measure whether the robot finishes. [Measured Graph Reach comparison](validation.md#corrective-post-training-2026-11-27)
