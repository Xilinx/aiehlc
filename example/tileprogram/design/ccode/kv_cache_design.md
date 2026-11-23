<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# KV Cache Autoregressive Decoding — Design Document

## Overview

`kv_cache.cc` demonstrates KV cache autoregressive text generation using the AIE tile programming model. It extends `attention.cc` with two distinct phases: **prefill** (batch-process a prompt) and **decode** (generate one token at a time, growing the KV cache each step).

This is the core optimization behind efficient LLM inference: during decode, only the new token's Q/K/V are projected, while all previous K and V vectors are reused from cache.

## Dimensions

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_SEQ` | 16 | Maximum sequence length (KV cache capacity) |
| `EMBED_DIM` | 8 | Embedding dimension per token |
| `PROMPT_LEN` | 4 | Number of prompt tokens |
| `GEN_LEN` | 4 | Number of tokens to generate |
| `VOCAB` | 8 | Vocabulary size |
| `SCALE_FACTOR` | 45 | 1/sqrt(8) in Q7 fixed-point (0.354 × 128) |

Data type: `int8`, Q7 fixed-point for attention scaling.

Mesh: 2×2 (4 AIE tiles).

## KV Cache Concept

```
PREFILL (process prompt at once):
  Embed 4 prompt tokens → project Q,K,V (full batch)
  → Store K,V in cache[0..3]
  → Attention for last prompt token → first output token

DECODE LOOP (4 steps, one token at a time):
  Embed new token → project q,k,v (single token)
  → Append k,v to cache → cache grows: 5,6,7,8
  → q * K_cache^T → softmax → * V_cache → output proj → residual
  → Score against vocab → argmax → next token
```

Key insight: during decode, we project only 1 new token but attend over ALL cached tokens. The KV cache avoids recomputing K and V for previous positions.

## Architecture

### Data Flow

```
                        ┌─────────────────────────────────────────────┐
                        │            PREFILL PHASE                    │
                        │                                             │
  prompt_ids ──→ embed_table ──→ prompt_emb [4×8]                    │
                                    │                                 │
                          ┌─────────┼─────────┐                      │
                          ↓         ↓         ↓                      │
                   linear_proj  linear_proj  linear_proj              │
                     (×Wq)       (×Wk)       (×Wv)                   │
                          ↓         ↓         ↓                      │
                     Q_buf[4×8]  K_buf[4×8]  V_buf[4×8]              │
                          │         │         │                      │
                          │    ┌────┘    ┌────┘                      │
                          │    ↓         ↓                            │
                          │  K_cache   V_cache                       │
                          │  [0..3]    [0..3]                        │
                          │    │         │                            │
                     q_last    │         │   (last row of Q_buf)     │
                          │    │         │                            │
                          ↓    ↓         │                            │
                    qk_scores_cached     │                            │
                          ↓              │                            │
                   approx_softmax_row    │                            │
                          ↓              ↓                            │
                     attn_apply_cached                                │
                          ↓                                           │
                   linear_proj_token (×Wo)                            │
                          ↓                                           │
                   residual_add_token (+ last prompt emb)             │
                          ↓                                           │
                    host_score_argmax → generated_ids[0]              │
                        └─────────────────────────────────────────────┘

                        ┌─────────────────────────────────────────────┐
                        │          DECODE LOOP (×4)                   │
                        │                                             │
  current_token ──→ token_embed ──→ token_emb [1×8]                  │
                                       │                              │
                             ┌─────────┼─────────┐                   │
                             ↓         ↓         ↓                   │
                     linear_proj_token (×Wq, ×Wk, ×Wv)               │
                             ↓         ↓         ↓                   │
                          q_tok     k_tok     v_tok                   │
                             │         │         │                   │
                             │    kv_cache_append (×2)                │
                             │         ↓         ↓                   │
                             │    K_cache[pos] V_cache[pos]           │
                             │    cache_len++                         │
                             │         │         │                   │
                             ↓         ↓         │                   │
                       qk_scores_cached          │                   │
                             ↓                   │                   │
                      approx_softmax_row         │                   │
                             ↓                   ↓                   │
                        attn_apply_cached                             │
                             ↓                                        │
                      linear_proj_token (×Wo)                         │
                             ↓                                        │
                      residual_add_token (+ token_emb)                │
                             ↓                                        │
                       host_score_argmax → next token                 │
                        └─────────────────────────────────────────────┘
```

### Cache Growth Over Time

```
Step         K/V Cache Contents           cache_len
─────────    ─────────────────────────    ─────────
Prefill      [p0][p1][p2][p3][ ][ ]...   4
Decode 0     [p0][p1][p2][p3][g0][ ]...  5
Decode 1     [p0][p1][p2][p3][g0][g1]..  6
Decode 2     [p0][p1][p2][p3][g0]...[g2] 7
Decode 3     [p0][p1][p2][p3][g0]...[g3] 8

p0-p3 = prompt token projections
g0-g3 = generated token projections
```

## Kernels

### Kernel 1: `linear_proj`

Full-sequence matrix multiply for prefill projections.

```
Signature: (in_feat, weights, out_feat, int seq_len)
Shapes:    [seq_len × D] × [D × D] → [seq_len × D]
Used for:  Prefill Q, K, V projections (3 launches)
```

### Kernel 2: `linear_proj_token`

Single-token matrix multiply for decode projections.

```
Signature: (in_tok, weights, out_tok)
Shapes:    [1 × D] × [D × D] → [1 × D]
Used for:  Decode Q, K, V, output projections (4 launches per step)
```

### Kernel 3: `kv_cache_append`

Copies a single vector into the cache at a given position.

```
Signature: (in_vec, cache_out, int pos)
Shapes:    [1 × D] → cache[pos × D .. (pos+1) × D]
Used for:  Growing the K and V caches (2 launches per decode step)
```

### Kernel 4: `qk_scores_cached`

Computes attention scores between a single query and all cached keys.

```
Signature: (in_q, in_kcache, out_scores, int cache_len)
Shapes:    [1 × D] × [cache_len × D]^T → [cache_len]
Formula:   scores[j] = (q · K_cache[j]) × SCALE_FACTOR >> 7
```

### Kernel 5: `approx_softmax_row`

Variable-length single-row softmax using piecewise-linear exp approximation.

```
Signature: (in_scores, out_attn, int len)
Shapes:    [len] → [len]
Approx:    x ≥ 0:   127
           x ≥ -32: 127 + 2x
           x ≥ -64: 63 + x
           x < -64: 0
Normalize: row sums to ~127
```

### Kernel 6: `attn_apply_cached`

Applies attention weights over the V cache to produce context.

```
Signature: (in_attn, in_vcache, out_context, int cache_len)
Shapes:    [cache_len] × [cache_len × D] → [1 × D]
Formula:   context[j] = (Σ_k attn[k] × V_cache[k][j]) >> 7
```

### Kernel 7: `residual_add_token`

Element-wise add with saturation for single-token residual connection.

```
Signature: (in_a, in_b, out)
Shapes:    [1 × D] + [1 × D] → [1 × D]
```

### Kernel 8: `token_embed`

Embedding table lookup for a single token ID.

```
Signature: (in_table, out_emb, int token_id)
Shapes:    table[token_id] → [1 × D]
```

## Buffer Layout

| Buffer | Shape | Bytes | Lifetime |
|--------|-------|-------|----------|
| `embed_table` | [8 × 8] | 64 | Entire program |
| `Wq, Wk, Wv, Wo` | [8 × 8] each | 256 total | Entire program |
| `K_cache` | [16 × 8] | 128 | Grows during decode |
| `V_cache` | [16 × 8] | 128 | Grows during decode |
| `prompt_emb` | [4 × 8] | 32 | Prefill only |
| `Q_buf, K_buf, V_buf` | [4 × 8] each | 96 total | Prefill only |
| `q_last` | [1 × 8] | 8 | Prefill attention |
| `token_emb` | [1 × 8] | 8 | Per decode step |
| `q_tok, k_tok, v_tok` | [1 × 8] each | 24 total | Per decode step |
| `scores` | [16] | 16 | Per attention call |
| `attn_weights` | [16] | 16 | Per attention call |
| `context` | [1 × 8] | 8 | Per attention call |
| `proj_out` | [1 × 8] | 8 | Per attention call |
| `result` | [1 × 8] | 8 | Per attention call |
| **Total** | | **~890** | |

## Host main() Flow

### Setup (lines ~370-430)
1. `aieSetDevice(0)`, `aieDim mesh(2, 2)`
2. Allocate all buffers
3. Initialize: identity-like weights, embedding table, `prompt_ids = {0, 3, 5, 2}`

### Prefill Phase (lines ~430-500)
4. Embed prompt tokens (host memcpy from embed_table)
5. `linear_proj<<<mesh>>>` for Q, K, V (3 launches, PROMPT_LEN=4 rows)
6. `aieDeviceSynchronize()` + memcpy K_buf, V_buf → K_cache, V_cache
7. Extract last query row → q_last
8. `qk_scores_cached<<<mesh>>>` (q_last, K_cache, scores, 4)
9. `approx_softmax_row<<<mesh>>>` (scores, attn_weights, 4)
10. `attn_apply_cached<<<mesh>>>` (attn_weights, V_cache, context, 4)
11. `linear_proj_token<<<mesh>>>` (context, Wo, proj_out)
12. `residual_add_token<<<mesh>>>` (last_prompt_emb, proj_out, result)
13. `aieDeviceSynchronize()`
14. Host: score result vs embed_table → argmax → first generated token
15. cache_len = 4

### Decode Loop (lines ~500-570, 4 iterations)
16. For each step:
    - `token_embed<<<mesh>>>` (embed_table, token_emb, current_token)
    - `linear_proj_token<<<mesh>>>` × 3 (q, k, v)
    - `kv_cache_append<<<mesh>>>` × 2 (k, v)
    - cache_len++
    - `qk_scores_cached<<<mesh>>>` + `approx_softmax_row<<<mesh>>>` + `attn_apply_cached<<<mesh>>>`
    - `linear_proj_token<<<mesh>>>` (output proj)
    - `residual_add_token<<<mesh>>>` (token_emb + proj_out)
    - `aieDeviceSynchronize()`
    - Host: score → argmax → next token

### Verification (lines ~570-700)
17. CPU reference mirrors exact same operations
18. Compare generated token sequences: PASS/FAIL
19. Print generated sequence, attention patterns (growing cache), efficiency message

### Kernel Launch Count
- Prefill: ~12 launches
- Decode: ~10 launches per step × 4 steps = ~40
- **Total: ~52 kernel launches**

## Next-Token Scoring

Scoring uses tied embedding weights on the host (not a kernel):

```
score(token_v) = result · embed_table[v]
next_token = argmax_v(score(token_v))
```

This is a dot-product between the output representation and each vocabulary embedding row, selecting the highest-scoring token.

## Verification Strategy

1. CPU reference functions replicate every kernel's logic bit-exactly
2. Both paths (AIE and CPU) run the full prefill + decode pipeline independently
3. Generated token sequences are compared element-by-element
4. Attention weight distributions are printed at each decode step to visualize growing cache utilization

## Comparison with attention.cc

| Aspect | attention.cc | kv_cache.cc |
|--------|-------------|-------------|
| Attention type | Full self-attention (all-to-all) | Cached autoregressive (query-to-cache) |
| Sequence handling | Fixed 8×8 matrix ops | Variable-length: prefill batch + single-token decode |
| KV storage | Computed fresh each time | Cached and grown incrementally |
| Phases | Single forward pass | Prefill + decode loop |
| Token generation | None | Full autoregressive loop with argmax |
| Kernel count | 5 | 8 |
| Key addition | — | kv_cache_append, token_embed, variable-length attention |

## Weight Initialization

```
Wq: 2 × identity   (amplify query signal)
Wk: 2 × identity   (amplify key signal)
Wv: identity + checkerboard   (mix features)
Wo: identity   (pass-through)
embed_table: 4 on diagonal + sparse 1s   (distinct per-token patterns)
```
