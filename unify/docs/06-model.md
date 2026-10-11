# Section 6 — The AI Model (covers original Sections 6, 7 and 16)

This is the longest file. Read it with a notebook open.

## 6.0 The whole model in one block

```
INPUT   x    (B, S, L)  float32     B windows, S sensors, L timesteps
        mask (B, S)     bool        True = real sensor, False = padding
   │
   │  reshape: fold sensors into batch  ── this is the universality trick
   ▼
        (B·S, 1, L)
   │
   │  SHARED 1-D CNN ENCODER  (same weights for every sensor, every dataset)
   │    Conv1d(1→32, k=5, pad=2) → BN → ReLU
   │    Conv1d(32→64, k=3, pad=1) → BN → ReLU → MaxPool(2)
   │    Conv1d(64→64, k=3, pad=1) → BN → ReLU
   │    AdaptiveAvgPool1d(1)      ── kills the L dependence
   ▼
        (B·S, 64)
   │
   │  reshape back
   ▼
        (B, S, 64)
   │
   │  + sensor embedding  E[sensor_ids]   (B, S, 64)
   ▼
   tokens (B, S, 64)
   │
   │  CROSS-ATTENTION:  M learned queries attend over the S sensor tokens
   │                    padded sensors masked with −inf
   ▼
   context (B, M·64)      with M=1 → (B, 64)
   │
   │  REGRESSION HEAD:  Linear(64→64) → ReLU → Dropout(0.2) → Linear(64→1) → Sigmoid
   ▼
OUTPUT  rul_norm (B, 1)   in [0,1]     ×  R_max_dataset  →  physical RUL
```

Parameter count: roughly **60k–90k**, i.e. ~250–350 KB in fp32. Small enough to live in the Pi's 1 MB L2 cache. That is deliberate.

---

## 6.1 Why a CNN?

**Plain language.** A convolution is a small pattern-detector that slides along the signal. Instead of learning "what does position 17 mean", it learns "what does a rising edge look like", and then finds rising edges anywhere.

**Technical.** A 1-D convolution with kernel size `k`, input channels `C_in`, output channels `C_out`:

```
y[c_out, t] = b[c_out] + Σ_{c_in} Σ_{j=0}^{k−1} w[c_out, c_in, j] · x[c_in, t + j − k//2]
```

Three properties earn its place:

1. **Translation equivariance / weight sharing** — the same filter applies at every time position. A trend that starts at t=5 or t=20 is detected by the same weights. A fully-connected layer would need to learn it separately at each position.
2. **Parameter efficiency** — a `Linear(30 → 64)` on one sensor is 1,984 parameters. A `Conv1d(1→64, k=5)` is 384, and it generalises better.
3. **Local-then-global hierarchy** — stacked convolutions build a receptive field. Layer 1 sees 5 timesteps, layer 2 sees ~7, layer 3 sees ~11 (with the pool, more). Early layers learn "slope, spike, level"; later layers learn "slope followed by plateau".

### Why **1-D** and not 2-D?

Your data is `(sensors × time)`. It looks like an image, so beginners reach for `Conv2d`. Don't — and be ready to explain why, because it is a favourite viva question:

- In an image, both axes are spatial and **translation-invariant**: shifting a cat 5 pixels right is still a cat. In your data, shifting along the *sensor* axis is meaningless — swapping sensor 3 and sensor 8 is not the same measurement. A 2-D kernel implicitly assumes neighbouring sensors are related, which is false; the ordering of columns in the CSV is arbitrary.
- A 2-D kernel spanning `k_s` sensors hard-codes the sensor count into the architecture. That is exactly problem **P1**.

So: convolve along time only, treat sensors as a *set*, and combine them with attention. That single sentence is the design thesis of the whole project.

### What each convolution layer learns (intuition to say out loud)

| Layer | Receptive field | Learns something like |
|---|---|---|
| Conv1 (k=5, 32 filters) | 5 timesteps | Local shapes: rising edge, falling edge, spike, flat, local mean/variance. Some filters approximate a derivative, others a smoother — the network rediscovers signal-processing primitives. |
| Conv2 (k=3, 64 filters) | ~11 timesteps | Combinations: "rise then plateau", "oscillation of period ~4", "sustained upward drift" |
| Conv3 (k=3, 64 filters) | ~19 timesteps | Window-scale summary: "monotone increase across the whole window", "stable with increasing variance" |
| GlobalAvgPool | whole window | Presence/strength of each pattern, position-independent |

**Why Global Average Pooling instead of flatten?** Flattening produces a vector of size `64 × L'`, which welds the model to a specific L and adds thousands of parameters. GAP averages over time → output size `64` regardless of L.

A precise caveat, because an examiner may push on it: GAP makes the *PyTorch* model accept any window length. The **exported ONNX deliberately fixes L**, because `dynamic_axes` marks only the batch and sensor axes. That is a choice, not an oversight — the Pi must use the window length the model was trained with, and a fixed axis makes a mismatch fail loudly instead of silently producing garbage. So state it as: *sensor-count-agnostic end to end; window-length-agnostic in principle, pinned at export for safety.*

---

## 6.2 The shared per-sensor encoder — the core trick

```python
class SensorEncoder(nn.Module):
    """Encodes ONE sensor's window into a D-dim descriptor. Weights shared across all sensors."""
    def __init__(self, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),  nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),          # (·, d_model, 1)
        )
    def forward(self, x):                     # x: (N, 1, L)
        return self.net(x).squeeze(-1)        # (N, d_model)
```

Used like this:

```python
B, S, L = x.shape
h = self.encoder(x.reshape(B * S, 1, L))     # (B·S, D)
h = h.reshape(B, S, -1)                      # (B, S, D)
```

### Why shared weights, in three arguments

1. **Sensor-count agnosticism (P1).** S only affects an internal reshape, never a weight shape. The same checkpoint accepts S=2 and S=21.
2. **Data efficiency.** With 21 sensors and shared weights, the encoder sees 21× more training examples than a per-sensor encoder would. In a field with 100 engines total, this is decisive.
3. **Transfer across datasets (P2).** "Detect a monotone upward drift" is the same operation whether the signal is turbine temperature or bearing RMS. Sharing forces the encoder to learn dataset-independent primitives.

**The obvious objection (and your answer).** *"Surely a temperature sensor and a vibration sensor need different processing?"* Correct — and that is what the **sensor embedding** restores. The encoder learns universal shape-primitives; the embedding tells the aggregator whose primitives these are. Separation of concerns: shape extraction is shared, identity is learned per-sensor.

---

## 6.3 Sensor embeddings

```python
self.sensor_emb = nn.Embedding(num_embeddings=64, embedding_dim=64)  # global vocab of 64 sensor IDs
tokens = h + self.sensor_emb(sensor_ids)     # (B,S,D) + (B,S,D)
```

**Plain language.** Every sensor in every dataset gets a permanent ID number and a learned 64-number "name tag". The tag is added to that sensor's descriptor so downstream layers know which sensor they are looking at.

**Technical.** `nn.Embedding` is a lookup table of shape `(V, D)` — a matrix whose row `i` is the vector for ID `i`, trained by backpropagation like any other parameter. It is exactly `one_hot(i) @ W`, implemented as an index for speed.

### The global sensor vocabulary

Maintain one file, loaded by both trainer and Pi:

```json
{
  "cmapss/T24": 1,  "cmapss/T30": 2,  "cmapss/T50": 3, "...": "...",
  "femto/acc_h_rms": 30, "femto/acc_h_kurt": 31, "femto/acc_v_rms": 32, "...": "...",
  "xjtu/acc_h_rms": 40,  "xjtu/acc_h_kurt": 41,  "...": "...",
  "live/mpu_ax_rms": 56, "live/mpu_ay_rms": 57, "...": "..."
}
```

**Design question worth thinking about:** should `femto/acc_h_rms` and `xjtu/acc_h_rms` share an ID? They are physically the same quantity from different rigs.

- **Separate IDs:** the model can learn rig-specific corrections. Safer.
- **Shared IDs:** forces true cross-dataset transfer, and — critically — lets you assign your **MPU-6050 features the same IDs as the XJTU features**, so the deployed model has actually *seen* that sensor ID during training. Without this, your live sensor maps to an untrained embedding row, which is random noise.

**Recommendation:** share IDs for physically identical features across the bearing datasets *and* your live rig, and add a separate small "domain embedding" (3–4 IDs: cmapss / femto / xjtu / live) added to every token in the batch. You get both transfer and rig-specific adaptation. This is the single most important detail for making your live demo defensible.

### Embedding dimension

Must equal `d_model` (64) so it can be added. Choosing D:

| D | Params (encoder ≈) | Notes |
|---|---|---|
| 32 | ~25k | Fast, may underfit; fine for a Pi-only variant |
| **64** | ~70k | ✔ Recommended. Standard for small time-series models |
| 128 | ~230k | Overfits on 100 engines unless heavily regularised |

Rule of thumb: with ~740 run-to-failure units total, keep total parameters under ~200k.

---

## 6.4 Attention from first principles

Skip nothing here; attention is what your examiner will drill into.

### The intuition

You have S descriptions (one per sensor). You want one summary. Averaging treats a dead constant sensor exactly like the one screaming about failure. Attention learns to compute a **weighted** average, where the weights are computed *from the data itself*, per-sample.

### The database analogy (use this in viva)

Attention is a soft dictionary lookup.

- **Query (Q)** — what you are looking for. "Which sensor shows late-stage degradation?"
- **Key (K)** — each item's advertisement of what it contains. "I am a temperature channel with a steep rise."
- **Value (V)** — the actual content you retrieve.

A hard dictionary returns the value of the one exactly-matching key. Attention scores *every* key against the query, softmaxes the scores into weights summing to 1, and returns the weighted blend of all values. Soft, differentiable, learnable.

### The equation

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Piece by piece:

- `Q ∈ R^{M × d_k}` — M queries
- `K ∈ R^{S × d_k}` — one key per sensor
- `V ∈ R^{S × d_v}` — one value per sensor
- `Q Kᵀ ∈ R^{M × S}` — raw similarity scores (dot product = unnormalised cosine similarity)
- `/√d_k` — **scaling**. Without it, for large `d_k` the dot products have variance ~`d_k`, pushing softmax into saturation where gradients vanish. Dividing by `√d_k` restores unit variance. (This is the "scaled" in *scaled dot-product attention*; a very common interview question.)
- `softmax(·)` row-wise → `A ∈ R^{M × S}`, each row non-negative and summing to 1
- `A V ∈ R^{M × d_v}` — the output: a convex combination of the values

### Self-attention vs cross-attention

| | Self-attention | Cross-attention |
|---|---|---|
| Q from | the same sequence as K, V | a **different** source |
| Output length | = input length (S) | = number of queries (M) |
| Use | contextualising tokens against each other | **aggregating a variable-size set into a fixed-size output** |

You need the second. Your input has variable S; your head needs fixed size. Cross-attention with M *learned* query vectors converts `S → M` regardless of S. (If you have seen Perceiver or DETR: same mechanism, "learned latent queries".)

Note: you can also use **self-attention among sensors first** (letting sensors inform each other — e.g. "temperature is up AND pressure is down"), then cross-attend to pool. That is one extra layer and usually worth 1–2 RMSE. Recommended as an upgrade once the baseline works.

---

## 6.5 Why cross-attention instead of concatenation

Put this table in your report; it directly answers your base-paper critique.

| Property | Concatenate `(B, S·D)` | Mean-pool | **Cross-attention** |
|---|---|---|---|
| Output shape depends on S | ✖ yes — fatal | ✔ no | ✔ no |
| Learns per-sensor importance | ✔ yes (via the linear layer) | ✖ no | ✔ yes, **and input-dependent** |
| Handles missing/failed sensors | ✖ no | partially | ✔ yes, via mask |
| Permutation behaviour | order-dependent (fragile) | invariant | invariant + identity via embedding |
| Interpretability | weights are static | none | ✔ attention map shows which sensor drove *this* prediction |
| Parameters | grows with S (`S·D×H`) | 0 | constant (`3·D²`) |

The interpretability point deserves emphasis: attention weights `A ∈ R^{1×S}` are a free per-prediction explanation. Plot them on your dashboard as a bar chart — "the model is currently looking 62 % at horizontal RMS, 21 % at kurtosis". Examiners love this, and it costs nothing because you already computed it. It also turns your black box into something a maintenance engineer would trust, which is the actual industrial objection to deep learning.

**Critical caveat to state honestly:** attention weights are *not* a rigorous causal explanation — there is a well-known literature ("Attention is not Explanation" / "Attention is not not Explanation") showing attention can be manipulated without changing predictions. Present them as an *indication of feature usage*, not proof. Saying this unprompted marks you as someone who reads papers.

---

## 6.6 The cross-attention module, with shapes and masking

```python
class CrossAttentionPool(nn.Module):
    """Aggregates a variable-length set of sensor tokens into a fixed-size context vector."""
    def __init__(self, d_model=64, n_queries=1, n_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)  # learned queries
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        self.h  = n_heads
        self.dk = d_model // n_heads

    def forward(self, tokens, mask):
        # tokens: (B, S, D)   mask: (B, S) bool, True = valid sensor
        B, S, D = tokens.shape
        M = self.q.shape[1]

        q = self.Wq(self.q.expand(B, -1, -1))          # (B, M, D)
        k = self.Wk(tokens)                            # (B, S, D)
        v = self.Wv(tokens)                            # (B, S, D)

        # split into heads: (B, h, ·, dk)
        q = q.view(B, M, self.h, self.dk).transpose(1, 2)
        k = k.view(B, S, self.h, self.dk).transpose(1, 2)
        v = v.view(B, S, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.dk ** 0.5     # (B, h, M, S)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)   # kill padding
        attn   = scores.softmax(dim=-1)                          # (B, h, M, S)

        out = attn @ v                                           # (B, h, M, dk)
        out = out.transpose(1, 2).reshape(B, M, D)               # (B, M, D)
        return self.Wo(out).reshape(B, M * D), attn              # (B, M·D), weights
```

### Shape trace with real numbers

```
B=256, S=21, D=64, M=1, h=4, dk=16

tokens          (256, 21, 64)
q  after Wq     (256,  1, 64) → heads (256, 4, 1, 16)
k  after Wk     (256, 21, 64) → heads (256, 4, 21, 16)
v  after Wv     (256, 21, 64) → heads (256, 4, 21, 16)
scores          (256, 4, 1, 21)
attn (softmax)  (256, 4, 1, 21)   each row sums to 1 over the 21 sensors
out = attn @ v  (256, 4, 1, 16)
merged          (256, 1, 64)
context         (256, 64)
```

Now run the identical code with S=2 (XJTU):

```
tokens (256, 2, 64) → scores (256, 4, 1, 2) → context (256, 64)   ← same output shape
```

**No weight has changed shape.** That is the entire "Any Dataset" claim, demonstrated in four lines of shape arithmetic. Be able to write this trace on a whiteboard from memory; it is the strongest thing you can do in a viva.

### Masking details that bite

- Use `-1e4`, not `-inf`. `-inf` produces `NaN` if an entire row is masked, and it exports to ONNX more reliably as a finite number. For fp16 use `-1e4`; `-1e9` overflows fp16.
- The mask must be applied **before** softmax, never after (masking after softmax leaves the weights not summing to 1).
- Guarantee at least one valid sensor per sample, or you get NaN. Assert it in the collate function.

### Multi-head — why 4 heads?

Splitting D into `h` independent subspaces lets different heads specialise: one might track energy features, another temperature-like trends, another whichever channel is currently anomalous. Cost is identical (same total D), and it consistently helps. `h` must divide `d_model`: 64/4 = 16. ✔

### Number of queries M

- M=1 → one context vector, minimal, `(B, 64)`.
- M=4 → four "views" of the sensor set, concatenated to `(B, 256)`. More capacity for the head, small extra cost.

Start with M=1. If the model underfits, try M=4.

---

## 6.7 The regression head

```python
self.head = nn.Sequential(
    nn.Linear(M * d_model, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 1), nn.Sigmoid(),     # output in [0,1] = normalised RUL
)
```

**Why so small?** The heavy lifting is done by the encoder and attention. A big head is the fastest route to overfitting with ~700 units of data.

**Why Sigmoid at the output?**

- Enforces `RUL_norm ∈ [0,1]` — physically impossible predictions become impossible by construction. No more negative RUL on the dashboard.
- Pairs naturally with the normalised-RUL design (§5.6).
- ⚠ Caveat: sigmoid saturates. If many labels sit exactly at 1.0 (all the capped healthy windows — which is most of your data!), gradients vanish there. Two mitigations: cap labels at 0.98 instead of 1.0, or drop the sigmoid and clamp at inference only. **Try both and pick empirically** — this is a real trade-off, not a settled question, and knowing that is worth saying.

### Activation functions — the quick reference

| Activation | Formula | Where you use it | Why |
|---|---|---|---|
| ReLU | max(0,x) | after every conv, in the head | Non-saturating for x>0, cheap, no vanishing gradient; the default |
| Leaky ReLU | max(0.01x, x) | alternative | Avoids dead units when many pre-activations go negative |
| GELU | x·Φ(x) | alternative in the head | Smoother, standard in transformers, slightly better sometimes |
| Sigmoid | 1/(1+e⁻ˣ) | final output only | Bounds output to [0,1] |
| Tanh | | not used | Saturating; superseded by ReLU in hidden layers |
| Softmax | eˣⁱ/Σeˣʲ | inside attention only | Turns scores into a probability distribution over sensors |

**Why non-linearity at all?** Without it, a stack of linear layers collapses to a single linear layer: `W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx`. Depth would be pointless. This is a guaranteed interview question — have the one-line proof ready.

### BatchNorm — and a deployment warning

`BatchNorm1d` normalises each channel across the batch, which stabilises and accelerates training. But:

- At **inference** it uses *running* mean/var accumulated during training. You must call `model.eval()` before export, or ONNX bakes in training-mode behaviour and your Pi predictions will be wrong. This is a classic, silent, extremely annoying bug.
- With batch size 1 at inference, BatchNorm in train mode divides by zero variance. `eval()` fixes it. Say it three times: **`model.eval()` before `torch.onnx.export`.**
- Alternative: `LayerNorm`, which is batch-size independent and therefore safer for edge deployment. Slight accuracy difference; worth testing.

---

## 6.8 Loss functions

### MSE (Mean Squared Error)

```
L_MSE = (1/N) Σ (ŷᵢ − yᵢ)²
```

- Differentiable everywhere, gradient `2(ŷ−y)` — clean and well-behaved.
- Penalises large errors quadratically → sensitive to outliers, which in prognostics are often mislabelled or pathological units.
- **Symmetric**: predicting 20 cycles early and 20 cycles late cost the same. Physically wrong.

### RMSE

`√MSE` — same units as RUL, so it is the *reporting* metric. Not usually the training loss (the sqrt adds nothing to the gradient direction and destabilises near zero).

### MAE / Huber

- MAE `|ŷ−y|` is outlier-robust but its gradient is constant, which slows late convergence.
- **Huber (smooth L1)** is quadratic near zero, linear far away. Best of both. `nn.SmoothL1Loss(beta=0.1)` on normalised targets is a very good default for this problem, and more stable than MSE on the noisy bearing datasets.

### The NASA scoring function

Defined for the PHM08 challenge. Let `d = ŷ − y` (positive = predicting **late**, i.e. more life than there is):

```
       ⎧ exp(−d/13) − 1     if d < 0   (early prediction — conservative, safe)
 s  =  ⎨
       ⎩ exp( d/10) − 1     if d ≥ 0   (late prediction  — dangerous)

Score = Σᵢ sᵢ         (lower is better; 0 is perfect)
```

**Asymmetry, quantified:**

| Error d | Early (d = −|d|) | Late (d = +|d|) | Ratio |
|---|---|---|---|
| 10 | 1.14 | 1.72 | 1.5× |
| 20 | 3.71 | 6.39 | 1.7× |
| 30 | 9.99 | 19.1 | 1.9× |
| 50 | 45.6 | 147 | 3.2× |

A late prediction of 50 cycles costs 3× an early one. That encodes the engineering reality: replacing a good part is wasteful; missing a failure is catastrophic.

```python
def nasa_score(y_hat, y):
    d = y_hat - y
    return torch.where(d < 0, torch.exp(-d/13) - 1, torch.exp(d/10) - 1).sum()
```

**Can you train on it directly?** It is differentiable, but exponential — a single bad prediction produces an enormous gradient and blows up training. Two workable approaches:

1. **Train on MSE/Huber, report the NASA score.** What most papers do. Safe.
2. **Combined loss:** `L = MSE + λ · mean(clamped NASA)`, with λ ≈ 0.01–0.1, and clamp `d` to e.g. ±60 before the exponential to bound the gradient. Gives some asymmetry benefit without instability. Recommended once your baseline trains cleanly.

⚠ **Compute the NASA score in physical units, on the capped scale**, never on normalised values — the constants 13 and 10 are in *cycles*. Getting this wrong produces a meaningless number that you cannot compare to literature.

⚠ It is also defined for C-MAPSS. For bearings it is not standard; report RMSE/MAE and optionally a scaled variant, and say clearly that you did.

### Asymmetry, the cheaper way

Simple weighted MSE gives you asymmetry with none of the exponential drama:

```python
w = torch.where(y_hat > y, 2.0, 1.0)      # 2× penalty for late predictions
loss = (w * (y_hat - y) ** 2).mean()
```

Two lines, stable, defensible. A very good answer to "how did you address the asymmetric cost of errors?"

---

## 6.9 Optimiser and backpropagation

### Backpropagation in one paragraph

The network is a composition of differentiable functions. The loss is a scalar. Backprop applies the chain rule from the loss backwards to every parameter, computing `∂L/∂θ` for all θ in one backward pass costing about the same as one forward pass. PyTorch builds the graph dynamically during `forward()`; `loss.backward()` walks it in reverse and accumulates `.grad` on every leaf tensor. That is all `autograd` is.

```python
optimizer.zero_grad()        # clear old gradients (they ACCUMULATE by default — classic bug)
loss = criterion(model(x, mask, ids), y)
loss.backward()              # compute ∂L/∂θ for every parameter
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stability
optimizer.step()             # θ ← θ − lr · update
```

### Optimisers

| Optimiser | Update rule (sketch) | Notes |
|---|---|---|
| SGD | `θ ← θ − lr·g` | Simple; needs careful LR tuning |
| SGD + momentum | `v ← βv + g; θ ← θ − lr·v` | Smooths noise, escapes shallow ravines |
| **Adam** | per-parameter adaptive LR from 1st and 2nd gradient moments | Fast convergence, insensitive to LR choice |
| **AdamW** ✔ | Adam with *decoupled* weight decay | Correct L2 regularisation; use this |

Adam maintains, per parameter: `m` (mean of gradients) and `v` (mean of squared gradients), bias-corrected:

```
m ← β₁m + (1−β₁)g          β₁ = 0.9
v ← β₂v + (1−β₂)g²         β₂ = 0.999
θ ← θ − lr · m̂ / (√v̂ + ε)   ε = 1e−8
```

Note this is why training memory is ~3× the model: weights + `m` + `v`. Exactly the point made in `02-edge-ai.md`.

**Use AdamW.** In plain Adam, weight decay gets divided by `√v̂` and therefore applies unevenly; AdamW decouples it. It is a one-word change and strictly better.

### Hyperparameters — a working starting set

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `d_model` | 64 | Fits the data size and the Pi's cache |
| Window `L` | 30 | Field convention; 30 s warm-up on the Pi |
| Batch size | 256 | Large enough for stable BatchNorm; fits laptop VRAM easily |
| Optimiser | AdamW | Decoupled weight decay |
| Learning rate | 1e-3 | Standard Adam starting point |
| LR schedule | Cosine annealing to 1e-5 over the run, 5-epoch linear warm-up | Warm-up prevents early divergence with attention |
| Weight decay | 1e-4 | Mild regularisation |
| Dropout | 0.2 (head), 0.1 (attention) | Small dataset → regularise |
| Epochs | 100, early stop patience 15 | You will overfit well before 100 |
| Grad clip | 1.0 | Attention + exponential-ish losses can spike |
| Loss | SmoothL1 on normalised RUL (+ optional 0.05× clamped NASA) | Stable, mildly asymmetric |
| Seed | fixed & logged | Prognostics results are high-variance |

### Training loop skeleton

```python
best = float("inf")
for epoch in range(epochs):
    model.train()
    for x, mask, ids, y, dset in train_loader:      # batches MIX datasets
        opt.zero_grad()
        y_hat, _ = model(x, mask, ids)
        loss = criterion(y_hat.squeeze(-1), y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

    model.eval()
    with torch.no_grad():
        val_rmse = evaluate(model, val_loader)      # per-dataset, in PHYSICAL units
    if val_rmse["mean"] < best:
        best = val_rmse["mean"]
        torch.save({"model": model.state_dict(), "cfg": cfg}, "best.pt")
```

**Watch for:** the mean across datasets can hide a collapse on one of them. Log per-dataset validation RMSE every epoch and plot all three curves. If FEMTO's curve goes up while C-MAPSS goes down, your loss is unbalanced — reweight per dataset (e.g. inverse to sample count).

---

## 6.10 Complete model, end to end

```python
class UnifyRUL(nn.Module):
    def __init__(self, d_model=64, n_sensor_ids=64, n_domains=4, n_queries=1, n_heads=4):
        super().__init__()
        self.encoder    = SensorEncoder(d_model)
        self.sensor_emb = nn.Embedding(n_sensor_ids, d_model)
        self.domain_emb = nn.Embedding(n_domains, d_model)
        self.pool       = CrossAttentionPool(d_model, n_queries, n_heads)
        self.head       = nn.Sequential(
            nn.Linear(n_queries * d_model, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x, mask, sensor_ids, domain_id):
        # x (B,S,L) float32 | mask (B,S) bool | sensor_ids (B,S) long | domain_id (B,) long
        B, S, L = x.shape
        h = self.encoder(x.reshape(B * S, 1, L)).reshape(B, S, -1)   # (B,S,D)
        h = h + self.sensor_emb(sensor_ids) + self.domain_emb(domain_id)[:, None, :]
        ctx, attn = self.pool(h, mask)                                # (B, M·D), (B,h,M,S)
        return self.head(ctx), attn                                   # (B,1), weights
```

### Full shape table (B=256, S=21, L=30, D=64, M=1, h=4)

| Stage | Shape | Elements | Params |
|---|---|---|---|
| input `x` | (256, 21, 30) | 161,280 | – |
| reshaped | (5376, 1, 30) | 161,280 | – |
| after Conv1 | (5376, 32, 30) | 5,160,960 | 192 |
| after Conv2 + pool | (5376, 64, 15) | 5,160,960 | 6,208 |
| after Conv3 | (5376, 64, 15) | 5,160,960 | 12,352 |
| after GAP | (5376, 64) | 344,064 | 0 |
| reshaped tokens | (256, 21, 64) | 344,064 | – |
| + sensor emb | (256, 21, 64) | 344,064 | 4,096 |
| + domain emb | (256, 21, 64) | 344,064 | 256 |
| attention scores | (256, 4, 1, 21) | 21,504 | – |
| context | (256, 64) | 16,384 | 16,640 (Wq,Wk,Wv,Wo) |
| head hidden | (256, 64) | 16,384 | 4,160 |
| output | (256, 1) | 256 | 65 |
| **Total params** | | | **≈ 44k** |

44k parameters ≈ **176 KB in fp32, 44 KB in int8**. Compare a typical CNN-LSTM RUL model at 1–4 M parameters. Yours is 25–100× smaller and it runs on a Pi in a couple of milliseconds. **That size number is a headline result — put it on a slide.**

---

## 6.11 Architecture comparison (original Section 16)

| Architecture | How it models time | Variable sensors? | Params | Edge-friendly? | C-MAPSS FD001 RMSE (typical) | Weakness for this project |
|---|---|---|---|---|---|---|
| MLP | none (flatten) | ✖ | 100k–1M | ✔ | 30–40 | No temporal structure; input welded to S·L |
| **1-D CNN** | local receptive fields | ✖ as usually written | 100k–1M | ✔✔ fast, parallel | 12–14 | Fixed sensor count if channels are conv inputs |
| **LSTM/GRU** | recurrent state | ✖ | 200k–2M | ✖ sequential, poor SIMD use | 13–16 | Slow on CPU, hard to parallelise, vanishing gradients on long windows |
| **CNN-LSTM** (your base paper #2) | CNN features → LSTM | ✖ hard-coded | 1–4M | ✖ | 11–13 | Best-of-both accuracy, worst-of-both deployment; sensor count fixed |
| **Transformer / DAST** (base paper #1) | full self-attention | ✖ fixed positional + input dim | 1–10M | ✖ O(L²), memory-hungry | 11–12 | Excellent accuracy, but fixed input dimension is exactly P1 |
| Informer | ProbSparse attention, O(L log L) | ✖ | 5–20M | ✖✖ | – | Built for long-horizon forecasting (L in thousands); overkill at L=30 |
| GNN | sensors as graph nodes | ✔ (naturally!) | 0.5–5M | ~ | 11–13 | Genuinely handles variable sensors — **but** needs a graph/adjacency you don't have; learned adjacency ≈ attention anyway |
| **UNIFY-RUL (yours)** | 1-D CNN per sensor + cross-attention over the sensor set | ✔✔ | **~44k** | ✔✔ | expect 13–17 | Slightly worse than specialists on any single dataset; no long-range temporal modelling beyond L |

### The honest positioning statement

Do not claim "our architecture is better." Claim this instead, which is true and defensible:

> *UNIFY-RUL trades 1–3 RMSE against dataset-specialised state of the art in exchange for three properties none of those models have: it accepts any sensor count without architectural change, it trains jointly across heterogeneous domains, and it deploys to a CPU-only ARM edge device at 44k parameters and single-digit-millisecond latency.*

That sentence is worth more marks than an inflated accuracy claim, and it will survive cross-examination. If an examiner pushes — "so it's worse?" — the answer is: "On FD001 alone, yes, by about 2 RMSE. But the DAST Transformer cannot run on XJTU at all without being rebuilt and retrained, and it will not fit our latency budget on a Pi. We optimised for a different objective, and we report both numbers."

### Note on GNNs

Graph neural networks are the closest genuine competitor for the "variable sensor count" property, and an examiner who knows the area may raise them. Your answer: a GNN needs an adjacency matrix expressing sensor relationships, which we do not have and which differs per dataset. Learning the adjacency turns the GNN into an attention mechanism (this equivalence is well known — a graph attention network over a fully-connected graph *is* self-attention). So we use attention directly, without the extra machinery.

---

## 6.12 Suggested improvements (ranked by value per hour of work)

1. **Quantile regression for uncertainty — solves your P4.** Change the head to output 3 numbers (e.g. the 10th, 50th, 90th percentiles) and train with the pinball loss:

   ```python
   def pinball(pred, y, qs=(0.1, 0.5, 0.9)):
       loss = 0
       for i, q in enumerate(qs):
           e = y - pred[:, i]
           loss = loss + torch.max(q * e, (q - 1) * e).mean()
       return loss
   ```

   Now the dashboard shows "RUL: 47 min (10–90 % interval: 31–68 min)" and you alarm only when the **10th percentile** crosses the threshold. That is a principled false-alarm control mechanism, costs about 15 lines, and converts P4 from unaddressed to solved. Highest-value change in this entire document.

2. **Temporal smoothing of predictions.** An EWMA over the last 10 predictions (α=0.3) plus N-consecutive-breach hysteresis on the alarm. Three lines. Removes almost all alarm flapping.

3. **Self-attention among sensors before pooling.** One extra `nn.MultiheadAttention` layer letting sensors see each other. Typically 1–2 RMSE improvement.

4. **Auxiliary health-index head.** Predict `1 − RUL_norm` monotonicity as an extra loss term, or add a monotonicity penalty `relu(HI_t − HI_{t−1})`. Multi-task regularisation, plus a meaningful dashboard bar for the unlabeled motor.

5. **Deep ensemble (3 models, different seeds).** Average the predictions; the spread is a second uncertainty estimate. Trivially parallel, 3× inference cost — still under 10 ms on the Pi. Reliable accuracy gain.

6. **Domain adversarial training (DANN).** A gradient-reversal branch that tries to predict which dataset a sample came from; the encoder is trained to *fool* it, producing domain-invariant features. This is the principled version of what you are attempting with joint training, and it is a strong, citable "advanced work" item.

7. **Temporal attention as well as sensor attention.** Currently GAP averages over time, discarding *when* within the window something happened. A small attention over the time axis before GAP recovers that.

---

## Summary

- Shared 1-D CNN encoder per sensor, folded into the batch dimension: `(B,S,L) → (B·S,1,L) → (B·S,D) → (B,S,D)`. This is the universality mechanism.
- Sensor (and domain) embeddings restore the identity the sharing erased.
- Cross-attention with M learned queries pools a variable-size sensor set into a fixed vector; masking handles padding and failed sensors.
- Small sigmoid-capped MLP head outputs normalised RUL in [0,1].
- Train with SmoothL1/MSE on normalised RUL; report RMSE in physical units and the NASA score on C-MAPSS. AdamW, 1e-3, cosine schedule, grad clip 1.0.
- ~44k parameters, ~176 KB fp32. Deliberately tiny.

## Key takeaways

1. The reshape `(B,S,L) → (B·S,1,L)` and the shape trace showing S=21 and S=2 producing identical output shapes are the two things you must be able to reproduce from memory.
2. `softmax(QKᵀ/√d_k)V` — know every symbol and why the `√d_k` is there.
3. Attention weights are free interpretability, but are an indication of usage, not a causal explanation.
4. The NASA score is asymmetric because late predictions are dangerous; compute it in physical units.
5. `model.eval()` before export, or BatchNorm silently ruins your deployment.

## Common interview questions

1. Derive the output shape of a `Conv1d` given input length, kernel, stride, padding, dilation. (`L_out = floor((L_in + 2p − d(k−1) − 1)/s) + 1`)
2. Why divide by `√d_k` in attention?
3. Difference between self-attention and cross-attention.
4. Why does a network need non-linear activations? Prove it.
5. Why is Adam usually faster than SGD, and what is its memory cost?
6. What is the difference between Adam and AdamW?
7. What does global average pooling do and why not flatten?
8. How does BatchNorm behave differently in train and eval mode?
9. Why is MSE a poor loss for asymmetric-cost problems?
10. Explain how gradient clipping prevents exploding gradients.

## Common viva questions

1. "Draw your architecture and label the tensor shapes." → Practise this until it takes 90 seconds.
2. "What happens to your model if I add a 22nd sensor?" → It runs. The encoder is shared; only the mask and the sensor-ID lookup change. If the new sensor ID was never trained, its embedding is untrained — so mention that you would map it to the nearest known ID or fine-tune the embedding alone.
3. "Why not just use an LSTM?" → Sequential, poor SIMD utilisation on Cortex-A72, 10–50× more parameters, and still sensor-count-fixed. Have the comparison table ready.
4. "What is the receptive field of your CNN?" → Compute it: k=5 then k=3 then pool(2) then k=3 → about 20 timesteps of the 30-step window. Know the number.
5. "How many parameters does your model have and how big is the file?" → 44k, ~176 KB fp32, ~44 KB int8. Have the exact number from your own code.
6. "Your attention has one query. Isn't that just a learned weighted average?" → Essentially yes for M=1, but the weights are computed from the input per-sample rather than being fixed — that is the crucial difference from a static weighting. Good, honest answer.
7. "How do you know the model isn't just learning the cycle counter?" → It cannot: the input is normalised sensor values, with no time index. Verify by shuffling the time axis and confirming performance collapses.

## Practical implementation tips

- **Overfit one batch first.** Take 8 samples, train 500 steps, confirm loss → ~0. If it cannot overfit 8 samples, there is a bug in the model or the data path. Do this before any real training run.
- Print shapes at every layer once (`print(h.shape)`), then delete. Ten minutes here saves ten hours later.
- Test with S=2 and S=21 in the *same* unit test, asserting identical output shape. Make that a `pytest` case; it is a direct test of your core claim.
- Log attention weights for a few validation samples every epoch. If they are uniform (1/S everywhere), the attention is not learning — check the mask and the scaling.
- Fix and log the random seed. Report mean ± std over 3 seeds; single-run numbers in prognostics are noise.

## Suggested improvements — see §6.12 above (ranked).
