# Architecture and learning objectives

## Inputs and representations

An observation consists of RGB views, an instruction, a proprioceptive state vector, and a world-graph snapshot. One training sample additionally contains the next `H` actions and a boolean validity mask. A checkpoint is tied to one explicit state/action schema; padding alone does not make different robot embodiments interchangeable.

The production backbone is `Qwen/Qwen3-VL-2B-Instruct`. Image/text preprocessing uses its own chat template and processor. Cameras are sorted by name and labeled in the prompt. Multimodal tokens are never silently truncated. Overlong instructions are rejected. The image-resolution limit is configurable.

We keep the final multimodal hidden states `V ∈ R^(B×L×Dv)` and project them to policy width `D`. Node labels are tokenized independently and represented by the masked mean of the backbone's pretrained input embeddings. They are lexical features, not separately generated descriptions or independent VLM reasoning results. A trainable projection maps them to `D`.

## Graph encoder

Graph data use `N` padded nodes, `R` relation types and a confidence tensor `E ∈ [0,1]^(B×N×N×R)`. `E[b,i,j,r]` is a directed source-to-target relation. Different relations may coexist on one pair. No edge is encoded as all zeros. Self edges are rejected.

Node initialization sums three paths:

1. An MLP over numeric attributes (geometry, confidence, observation age, bounding box).
2. A trainable node-type embedding.
3. The projected semantic label embedding.

A learned global token guarantees nonempty attention context even for an empty graph. Graph nodes have no learned index embeddings, so relabeling node indices and permuting all associated tensors preserves graph equivariance. Stable object identity is maintained by `WorldGraph`; IDs themselves are not integer embeddings.

Each relation block adds normalized outgoing/incoming predicate messages, applies node self-attention, cross-attends to VLM tokens, then applies a SwiGLU feed-forward layer. For head `h`, the structural attention bias between nodes is:

`bias[h,i,j] = Σr E[i,j,r] wf[r,h] + Σr E[j,i,r] wr[r,h]`.

Separate forward/reverse parameters preserve edge direction. Predicate messages also enter node values, ensuring relation content is represented beyond an attention preference. Padded nodes are masked as keys and zeroed after each block. All real nodes may attend to one another; this is dense relation-biased attention, not a sparse neighborhood-only GNN. Memory scales quadratically in the number of graph nodes, and the default node limit is 64.

Bounding boxes are numeric grounding cues, and graph queries cross-attend to the complete multimodal context. This version does not apply a hard ROI mask to Qwen's image patches. Tracking, camera calibration, object localization and semantic relation extraction remain outside the network.

## Action expert

The trainable action expert receives normalized robot state `s`, noisy action chunk `xt`, time `t`, VLM tokens `V`, and graph tokens `G`.

Each block has:

- Bidirectional action self-attention with rotary positions and QK RMS normalization.
- A separate gated cross-attention path to `V`.
- A separate gated cross-attention path to `G`.
- RMSNorm and SwiGLU.
- Time/state-conditioned scale, shift and residual gates.

Time/state embeddings also enter the initial action representation directly, preserving conditioning from the first optimization step. Graph/VLM gates start nonzero, so graph gradients are available immediately. The action sequence is bidirectional because an entire future chunk is generated jointly.

With `model.action.condition_on_target: true`, the target head supplies an additional conditioning vector. Let `pᵢ` be the softmax of its predicted scores over the real nodes and `fᵢ` the numeric attributes of node `i` from the graph snapshot. A learned linear projection maps their weighted average to the action model's width.

```text
target_condition = W · Σᵢ pᵢ fᵢ
condition = time_embedding(t) + state_embedding(s) + target_condition
```

The encoder computes the target vector once per observation. Every timestep in the action chunk receives it, both at the input and through the blocks' adaptive modulation. An empty graph contributes exactly zero. If several nodes have appreciable probability, their attributes contribute in proportion to their scores. Flow-loss gradients reach the target selector through these probabilities even on examples without a target label. The optional `target_id` supervises the classification loss only.

The flag defaults to `false`, which preserves the original network and its checkpoint parameter names. Full graph ablation in the offline evaluator disables both graph-token attention input and the target vector. The closed-loop reach benchmark's `no_edges` condition instead recomputes the prediction with all nodes present and their relations removed.

## Flow matching

Let `a` be a demonstrated normalized action chunk, and `ε ~ N(0,I)`. The implementation consistently uses:

`xt = (1-t) ε + t a`, with `t=0` meaning noise and `t=1` meaning actions.

The target velocity is `a-ε`. Time samples are logit-normal by default; uniform sampling is configurable. The objective is the mean squared error over valid action dimensions and timesteps:

`Lflow = Σ M ⊙ (vθ(xt,t,s,V,G) - (a-ε))² / Σ M`.

Invalid padded actions and noise are zeroed before interpolation and attention. Invalid timesteps cannot serve as self-attention keys. Each sample must contain at least one valid action. This prevents padded episode tails from leaking arbitrary values into valid predictions.

Inference starts from Gaussian noise and numerically integrates toward `t=1`. Euler uses one field evaluation per step; Heun uses two. Both are implemented with the same orientation and tested using an analytically known constant field. The VLM and graph are encoded once per sampled chunk, not at every flow step. Cross-attention K/V projections are currently recomputed by the action blocks; no persistent cross-step KV cache or asynchronous RTC implementation is claimed.

## Graph objectives and annotation semantics

The total objective is:

`L = Lflow + λrel Lrel + λtarget Ltarget`.

During training, random unordered node pairs are selected. Both directed edge vectors for those pairs are zeroed before graph encoding. A directed pair head reconstructs all predicates for hidden pairs using binary cross entropy, with supplied confidence values as soft targets. **Unannotated relations are treated as negative for this auxiliary loss.** For incomplete/open-world annotations, set `relation_loss_weight: 0` until an appropriate known-label mask is supplied by a future dataset extension. Setting the weight to zero also disables relation corruption.

`target_id` is optional supervision. It is converted to an index for cross entropy over real nodes; it is not fed to the encoder. No target loss is taken for unlabeled or empty graphs. A relation literally named `target`, if supplied as an input edge, is still input information; do not create such an edge from held-out supervision.

Pair reconstruction and target selection support learning structured representations. They do not establish symbolic consistency, causal reasoning, explanation faithfulness, or improved robot success on their own.

## Runtime graph and execution

`WorldGraph` stores timestamped nodes and relations. A snapshot removes expired observations, includes normalized node age, and omits relations with missing endpoints. Node removal also removes incident edges. The policy wrapper invalidates queued actions on any graph change or instruction change; it can also be explicitly forced to replan.

A graph update must share the camera/state time and robot coordinate frame. A graph can be sourced from Isaac ground truth, perception/tracking, annotated demonstrations, or a separately implemented KnowRob/RDF export. These adapters should emit the same documented JSON schema.

Predicted actions are denormalized before being returned. The package does not send hardware commands. A deployment must define control frequency, reference frames, action conventions, collision checking, motion limits, stop behavior and sensor-failure handling independently.
