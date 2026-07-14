# GPT2 RoPE Scaling
##RoPE,Positional Interpolation,NTK-aware,NTK-by-parts,YaRN

A minimal, from-scratch implementation of GPT-2 with rotary positional embeddings (RoPE),
plus four inference-time context-extension variants: PI, NTK-aware, NTK-by-parts, and YaRN.

Style benchmark: Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## What's inside

- 6-layer GPT-2 (~10M params) with RoPE-native attention (no `wpe`, no ALiBi)
- 5 self-contained positional-encoding modules with a shared `forward(start, end) → (cos, sin)` interface
- Full training pipeline on Tiny Shakespeare (~30 min on M-series Mac)
- KV-cache autoregressive generation with 4 sampling strategies
- OOD cliff experiment: same trained weights, 5 different PE methods at 2× training context

## The PE evolution chain

Each variant is one small mathematical delta from the previous one. All are inference-time
transformations — no retraining required to swap.

| Method | What it changes | Formula (relative to RoPE) |
|---|---|---|
| **RoPE**         | baseline                            | θᵢ = base^(−2i/d) |
| **PI**           | positions                           | position → position / s |
| **NTK-aware**    | base (per-dim implicit)             | base → base · s^(d/(d−2)) |
| **NTK-by-parts** | per-dimension γ ramp                | θᵢ' = (1−γᵢ)·(θᵢ/s) + γᵢ·θᵢ |
| **YaRN**         | NTK-by-parts + attention temperature | +√(1/t) into cos/sin, where 1/t = 0.1·ln(s) + 1 |

The differences are easier to see visually:

<img width="2074" height="807" alt="pe_comparison" src="https://github.com/user-attachments/assets/b82e604f-b395-4e93-8dc0-d4ef9ff42fed" />


*The right panel is the money shot. RoPE doesn't scale anything (flat at 1.0). PI scales every
dimension uniformly (flat at 1/s). NTK-aware smoothly ramps. NTK-by-parts is surgical: leave
fast dims alone (they already fit), fully interpolate slow dims (they need the help), smooth
transition between.*

## Layout
├── config.py           # GPTConfig dataclass with every PE knob
├── layers.py           # CausalSelfAttention, MLP, Block
├── model.py            # GPT class with PE dispatch by config.pe_method
├── data.py             # DataLoaderLite for Tiny Shakespeare
├── train.py            # AdamW + cosine LR + warmup + grad clipping
├── generate.py         # 4 sampling strategies, KV-cache, PE method comparison
├── pe/
│   ├── rope.py
│   ├── pi.py
│   ├── ntk_aware.py
│   ├── ntk_by_parts.py
│   └── yarn.py
├── input.txt                       # Tiny Shakespeare
└── plot_pe_comparison.py           # generates pe_comparison.png


## Usage

```bash
# Train (default: pe_method="rope", block_size=256, max_steps=5000)
python train.py

# Generate 500 tokens with every PE method at scale_factor=2.0 (2× training context)
python generate.py

# Regenerate the PE comparison plot
python plot_pe_comparison.py
```

Swap PE methods by editing `pe_method` in `GPTConfig`. All 5 use the same interface.

## Training

- **Model:** 6 layer × 6 head × 384 embed × d_head 64
- **Data:** Tiny Shakespeare, tokenized with tiktoken GPT-2 (338K tokens, ~60 epochs at 5000 steps)
- **Optimizer:** AdamW, lr=3e-4, betas=(0.9, 0.95), weight decay 0.1
- **Schedule:** 100-step linear warmup → cosine decay to lr/10
- **Grad clip:** 1.0
- **Batch:** B=16, T=256 → 4096 tokens per step

Reaches loss ≈ 0.13 on Tiny Shakespeare by step 5000 (~memorization for a dataset this size).

## OOD cliff experiment

Same trained weights loaded into each PE variant at inference time. Generate 500 new tokens
(total context 506, i.e. 2× training). Prompt: `"To be, or not to be"`.

Qualitative findings (see `generations_*.txt`):

- **Plain RoPE**: cliff clearly visible past position 256. Text collapses into
  a `"Thou camest thy wife"` repetition attractor — the classic OOD failure.
- **PI, NTK-aware, NTK-by-parts, YaRN**: all avoid the cliff. Character-attribution
  format, character names, and rough grammatical structure hold through all 500 tokens.
- **Attractor choice differs by method.** RoPE → "Thou camest". PI → PETRUCHIO/HORTENSIO
  dialogue. NTK-aware → VINCENTIO court scene. NTK-by-parts → PETRUCHIO monologue.
  YaRN → similar to NTK-by-parts.
- At s=2, the theoretical hierarchy (NTK-by-parts > YaRN > NTK-aware > PI > RoPE) doesn't
  cleanly show up in generation quality — all four scaling methods work well enough that
  differences are subtle. Larger scale factors would sharpen the differences.

## Notes to my future self

- **PE scaling methods are inference-time context extenders.** They don't lower the loss
  within training range; they preserve quality *outside* it.
- **`register_buffer(..., persistent=False)` matters.** If cos/sin tables are persistent
  buffers, `load_state_dict` silently overwrites them — every PE method ends up using
  the trained checkpoint's tables regardless of which class you instantiated. All 5 PE
  files in this repo set `persistent=False`.
- **KV cache stores rotated keys.** Rotation happens *before* the cache concat, exactly
  once per key, at the position it was first computed. This is what preserves the
  relative-position invariant.
- **Position injection differs from vanilla GPT-2.** With RoPE there is no `wpe` embedding
  added to the input. Position enters *repeatedly*, inside every attention layer, applied
  only to Q and K (never V), via cos/sin rotation.

## Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — style benchmark
- [RoFormer](https://arxiv.org/abs/2104.09864) — the original RoPE paper
- Chen et al. — Position Interpolation
- bloc97 — NTK-aware and NTK-by-parts (reddit posts)
- Peng et al. — [YaRN](https://arxiv.org/abs/2309.00071)
