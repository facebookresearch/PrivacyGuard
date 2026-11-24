# Jacobian lens (J-lens) preview — experimental

For questions, reach out to Matthew Grange

An optional, experimental PrivacyGuard analysis that shows, for one prompt, what each
layer × token position of a decoder language model reads out as ranked vocabulary
tokens under a fitted Jacobian lens, side by side with an ordinary logit lens and the
model's actual next-token output. A separate, controlled experiment replaces one
concept representation (spider) with another (ant) and checks whether the answer
moves (8 legs toward 6), with no-op, matched-norm and restoration controls.

This is a research visualization. It is **not** a privacy metric, it does not change
any other PrivacyGuard attack or analysis, and it makes no claim about model intent,
consciousness, a "global workspace", or privacy efficacy. Layers are candidate read-out
sites, not validated workspace bands. A readout on one prompt is not a replication of
the reference paper's results.

## Method

The Jacobian lens follows the reference implementation released with Anthropic's
global-workspace research,
https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e
(Apache License 2.0). For a source layer `l` the lens is the average Jacobian of the
final block's output (before the final norm) with respect to the layer-`l` residual:

    J_l = mean over prompts [ mean over valid source positions s [ sum over targets t >= s of d h_final[t] / d h_l[s] ] ]

It is estimated on a separate generic calibration corpus with model weights frozen and
no optimizer (activation gradients only). A residual `h` at layer `l` is read out by
applying the model's own native output head to `J_l @ h`; the logit lens applies the
same head to `h` directly. Valid positions skip the first 16 tokens and the last one,
as in the reference.

The causal experiment uses closed-form concept directions from separate, number-free
template sentences (`contrast_mean_clamp`), not the reference's linear-probe swap
(`data/experiments/probe-swap.json`), and is labeled by the method it actually uses.
Its success rule, band, strengths and controls are fixed in
`glimmer_preview_pipeline.py` before any model run; every arm is reported.

## Model and pinned inputs

| Item | Value |
|---|---|
| Model | `meta-models/Muse-Glimmer-30B`, main BF16 checkpoint (not the `-assistant` drafter), https://huggingface.co/meta-models/Muse-Glimmer-30B |
| Revision | `1297fd7dfeac4598948a5d15886968435c81c2b7` (model card license: Apache-2.0) |
| Shard SHA-256 | `model-00001-of-00002.safetensors` `8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694`; `model-00002-of-00002.safetensors` `b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381` |
| Tokenizer SHA-256 | `tokenizer.json` `c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d` |
| Model code | `transformers==5.19.0` (`muse_glimmer`); native head `20 * tanh(lm_head(final_norm(h)) * 0.19611613513818404 / 20)`, untied `lm_head` |
| Method reference | `anthropics/jacobian-lens` @ `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Calibration corpus | `Salesforce/wikitext` @ `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-103-raw-v1` train shard 0 (SHA-256 `74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212`), first N records with at least 600 characters, 128 tokens each; not distributed with this package |
| Prompts | `blog_no_period`: `The number of legs on the animal that spins webs is`; `blog_literal`: the same with the final period; `upstream_probe_swap`: `Fact: The number of legs on the animal that spins webs is ` (each with an explicit BOS token); post-primary follow-ups add `The number of legs on the insect that lives in colonies is` and its `Fact: ... is ` form |

Loading verifies every shard's SHA-256 against these values, refuses a mismatch, and
drops the unused vision tower. Finite differences in bf16 are dominated by rounding (a
logit's bf16 spacing is 0.125 near 16), so derivative fidelity is established by the
float32 run, not by the bf16 one.
`hidden_states` from `output_hidden_states=True` are not used: in transformers 5.19 the
last entry is replaced by the final-normed state, so residuals are taken with forward
hooks on the decoder blocks.

## Reproduce

Hardware: one 80 GB GPU for the bf16 runs and about 115 GB of RAM for the float32 CPU
derivative check. Peak GPU memory, step time and fit cost are recorded in the numerical
receipt (`compute_profile`, at sequence length 128) and in the fit manifest. The
inference-only quantized builds (GGUF, ExecuTorch) and the `-assistant` drafter are not
targets for this analysis.

```
git clone https://github.com/facebookresearch/PrivacyGuard.git && cd PrivacyGuard
# transformers 5.19.0 is the version the results below used. Other transformers or torch
# versions, or other GPUs, can change scores slightly.
pip install -e ".[jacobian_lens]" "transformers==5.19.0"
hf download meta-models/Muse-Glimmer-30B --revision 1297fd7dfeac4598948a5d15886968435c81c2b7 --local-dir CKPT
hf download Salesforce/wikitext --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --include "wikitext-103-raw-v1/train-00000-of-00002.parquet" --local-dir WIKITEXT

# 1a. GPU receipt: shard hashes, head and resume parity, memory profile.
python -m privacy_guard.analysis.jacobian_lens.glimmer_numerics --checkpoint_dir=CKPT --output=OUT/numerics_bf16.json
# 1b. Derivative check in float32 on CPU: VJP vs central finite differences.
python -m privacy_guard.analysis.jacobian_lens.glimmer_numerics --checkpoint_dir=CKPT --device=cpu --dtype=float32 \
  --profile_batches= --output=OUT/numerics_fp32_cpu.json
# 1c. Compare bf16 GPU VJPs with the float32 reference.
python -m privacy_guard.analysis.jacobian_lens.glimmer_numerics --checkpoint_dir=CKPT \
  --reference_receipt=OUT/numerics_fp32_cpu.json --output=OUT/numerics_bf16.json

# 2. Lens fit, split into two shards (one per GPU, or one after the other). Each command is
#    resumable: rerun it to continue. Then merge the shard lenses.
for SHARD in 0 1; do
  python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline fit \
    --checkpoint-dir CKPT --wikitext-dir WIKITEXT --out-dir OUT --private-dir PRIVATE --n-prompts 32 \
    --shard $SHARD --num-shards 2
done
python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline merge \
  --checkpoint-dir CKPT --out-dir OUT --private-dir PRIVATE \
  --shard-lenses PRIVATE/lens_shard0of2.pt PRIVATE/lens_shard1of2.pt \
  --shard-manifests OUT/fit_manifest_shard0of2.json OUT/fit_manifest_shard1of2.json

# 3. Readout receipts for every prompt, the predeclared causal experiment with its controls,
#    and the post-primary follow-ups.
python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline run \
  --checkpoint-dir CKPT --wikitext-dir WIKITEXT --out-dir OUT --private-dir PRIVATE \
  --lens PRIVATE/lens_merged.pt --fit-manifest OUT/fit_manifest_merged.json --causal --followups --top-k 10

# 3b. Held-out lens quality: per-layer top-1 agreement of the J-lens and the logit lens with
#     the model's own next token, on WikiText records the fit never saw.
python -m privacy_guard.analysis.jacobian_lens.glimmer_preview_pipeline evaluate \
  --checkpoint-dir CKPT --wikitext-dir WIKITEXT --out-dir OUT --private-dir PRIVATE \
  --lens PRIVATE/lens_merged.pt --fit-manifest OUT/fit_manifest_merged.json --heldout-start 200

# 4. Self-contained offline viewer. Cells use the float32 head and the actual output uses
#    native bf16 logits, which differ by up to one bf16 step (0.125) near the top scores.
python -m privacy_guard.analysis.jacobian_lens.preview_render_main --receipts OUT/jlens_preview_receipts.json \
  --out-dir OUT/preview --head-score-tolerance 0.125
```

`PRIVATE` holds tensors (lens matrices, fit checkpoints, concept directions) and is
never part of a receipt. Every receipt is built through `JacobianLensAttack`
(`privacy_guard/attacks/jacobian_lens/`), PrivacyGuard's `BaseAttack` for this preview: given a
model adapter, a prompt and an optional fitted lens, `run_attack()` returns the
`JacobianLensPreviewInput` that `JacobianLensPreviewNode` analyzes. The tutorial notebook
`tutorial_notebooks/jacobian_lens_preview_tutorial.ipynb` loads the receipts, summarizes
them with `JacobianLensPreviewNode` and embeds the viewer.

## Results on Muse Glimmer-30B (first run, 2026-11-24)

One A100 80 GB, the commands above, a lens fitted on N=32 WikiText prompts. This is an
engineering lens, not a replication of the reference paper's lenses. The tutorial ships a
trimmed copy of the receipts (`tutorial_notebooks/data/jacobian_lens/jlens_preview_receipts.json.gz`,
top-5, three decimals). Each trimmed receipt records the hash of its full source receipt.

- Numerics: the native head is reproduced bit for bit, and so is resuming the forward
  from any block. In float32, VJPs match central finite differences to <= 1e-3 relative
  error, and gradients are exactly 0 at positions later than the target. bf16 VJPs match
  float32 with cosine >= 0.998.
- Lens: an independent reimplementation of the estimator, checked against the fit's first
  prompt, reproduces rows 0-7 bit for bit on the same GPU and four rows to cosine >= 0.996
  in float32 on CPU, at every layer.
- Held-out lens quality, on 8 WikiText records the fit never saw (888 positions): top-1
  agreement with the model's own next token, averaged over layers, is 0.160 for the J-lens
  and 0.120 for the logit lens. The J-lens leads at 40 of 51 layers, mostly mid-stack
  (layer 12: 0.125 vs 0.007; layer 24: 0.087 vs 0.007). The logit lens is equal or better at
  the last layers (layer 50: 0.791 vs 0.856). This is a modest mid-stack gain from a small
  fit and a small held-out set, not evidence of a better lens in general.
- Reproduction: an independent build of the same code, rerun on the same GPU, reproduces
  every receipt exactly: all cells, ranks, scores and intervention results.
- Readout: the table gives the best rank of a single-token spider form at the final prompt
  position.

  | Prompt | J-lens | Logit lens |
  |---|---|---|
  | `Fact: ... spins webs is ` | ranks 1-6 at layers 21-31 and 35-49, ranks 11-79 at layers 32-34 | ranks 2-5 at layers 15-17, ranks 15-2496 at layers 18-34, ranks 1-3 at layers 35-45 |
  | `... spins webs is.` | rank 1 at layers 41-49 | ranks 1-6 at layers 39-49 |
  | `... spins webs is` | best rank 5 (layer 44) | rank 1 at layer 41, rank 3 at layer 36 |

- Causal experiment, predeclared primary: a contrast-mean clamp from spider to ant over
  layers 13-38 at strength 1. It does not change the answer on any prompt, so the result
  is negative.
- At strength 2, a predeclared secondary (exploratory) arm, the edit moves the `Fact:`
  prompt's answer from 8 to 6 (p 0.53). This is a sparse, large local edit: only 4-5
  (layer, position) pairs pass the gate, each moved by about 0.21 times its residual norm.
  The matched-norm controls reuse exactly those positions and norms.
- Post-primary follow-ups, declared after the primary result, at strength 2 move the answer
  to 4 (spider to dog, p 0.82) and to 2 (spider to bird, p 0.87). Matched-norm unrelated
  and random edits keep 8 (p 0.84-0.93), and so does the all-positions restoration (bitwise
  equal to baseline). Restoring only the final prompt position also keeps 8 (p 0.87), so
  the edit at that position is needed.
- The reverse edit (ant to spider on a six-legged prompt) does not move the answer.
- The prompt with the final period has no answer at baseline (`A. 6 B. 8`), so the causal
  experiment is not applicable to it.

## Outputs and data boundary

A `privacyguard.jlens_preview/v1` receipt holds token IDs, decoded token text, scores and
full-vocabulary ranks for the top-k and the tracked tokens of each cell, the model's own
greedy output, intervention summaries and pinned identities. The validator rejects
activations, residual vectors, Jacobian matrices, full logit arrays and weights. The HTML
viewer inlines all data and code and sets a Content-Security-Policy that blocks network
requests: no remote scripts, styles, fonts or model calls.

## Attribution and licenses

- Method and reference estimator: Anthropic, `anthropics/jacobian-lens` (Apache License
  2.0), pinned at the commit above; `lens_fitting.py` adapts its estimator and says so.
- Model: Meta, `meta-models/Muse-Glimmer-30B` (Apache License 2.0 per its model card,
  with its usage policy); weights are downloaded by the user and never redistributed here.
- Model implementation: Hugging Face Transformers 5.19.0 (Apache License 2.0).
- Calibration text: WikiText-103 (Creative Commons Attribution-ShareAlike); only hashes
  and lengths of the selected records appear in manifests.
