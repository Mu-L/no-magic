# MicroTurboQuant Challenges

Test your understanding of data-oblivious vector quantization by predicting what happens in these scenarios. Work out the answer before revealing it.

---

### Challenge 1: When Rotation Hurts Instead of Helps

**Setup:** `absmax_quantize` (lines 160-182) computes `scale = max(|x_i|) / levels` per vector. `turboquant_encode` (lines 190-202) rotates first, then absmax-quantizes the rotated coordinates. The comment in the `ANISOTROPY` constant (lines 42-49) notes that rotation wins above a certain anisotropy threshold but loses above another.

**Question:** Suppose you quantize a perfectly 1-sparse unit vector `x = [1.0, 0.0, 0.0, ..., 0.0]` (1 non-zero coord, 31 zeros) at 4 bits. What does the baseline absmax quantization achieve? What does TurboQuant achieve? Why does one clearly win?

<details>
<summary>Reveal Answer</summary>

**Answer:** Baseline: `max(|x|) = 1.0`, `scale = 1.0 / 7 ≈ 0.143`. Non-zero coord quantizes to `round(1.0 / 0.143) = 7`, dequantizes to `7 * 0.143 = 1.0` — exact. Zero coords quantize to `0` and dequantize to `0` — exact. Reconstruction error: essentially zero. TurboQuant: `y = R @ x` has all 32 coordinates non-zero with typical magnitude `1/sqrt(32) ≈ 0.18`. Absmax scale over `y` is much smaller (`~0.55/7 = 0.08`), and quantization error is spread across all 32 coordinates. Total squared reconstruction error scales as `D * (scale/sqrt(12))^2 ≈ 0.013` — roughly `0.013` vs. baseline's `~0.00001`. TurboQuant is thousands of times worse here.

**Why:** Sparse unit vectors are the adversarial case for TurboQuant. The rotation `R` is dense (Gaussian entries), so `R @ x` destroys the sparsity that absmax was exploiting for free. Absmax on a 1-sparse vector allocates every integer level to the single non-zero coordinate; the 31 zeros cost zero bits of error. Rotation converts "one exact non-zero, 31 exact zeros" into "32 imprecise values", and the per-coord quantization error accumulates via the inverse rotation. This is the fundamental reason the paper targets dense inner-product queries over embedding spaces, not compression of structurally sparse data: when sparsity is genuine, absmax already has the data-oblivious optimality that TurboQuant provides only on dense inputs.

**Script reference:** `03-systems/microturboquant.py`, lines 42-49 (ANISOTROPY sweet-spot comment), lines 160-182 (absmax_quantize), lines 190-202 (turboquant_encode), lines 248-297 (sample_name_embeddings with signpost about sparse vectors)

</details>

---

### Challenge 2: Why Gram-Schmidt on Gaussians Produces a Uniform Rotation

**Setup:** `random_rotation` (lines 109-142) fills a D-by-D matrix with i.i.d. standard-Gaussian entries, then orthonormalizes columns via Gram-Schmidt. The docstring claims the result is distributed as Haar measure on O(D) — uniform over all orthogonal matrices.

**Question:** What property of the Gaussian distribution makes this work? What would go wrong if you replaced `gaussian_sample()` with `random.uniform(-1, 1)` on line 124?

<details>
<summary>Reveal Answer</summary>

**Answer:** The multivariate standard Gaussian is **rotation-invariant**: if `g ~ N(0, I_D)`, then `R g ~ N(0, I_D)` for any orthogonal `R`. This means every direction on the unit sphere is equally likely to be the normalized first Gaussian column, so after Gram-Schmidt the first column is uniform on the sphere — and by induction, the whole orthonormal basis is Haar-uniform. Uniform(-1, 1) is NOT rotation-invariant: the distribution is a hypercube, not a ball. Gram-Schmidt on uniform-hypercube columns biases the first column toward axis-aligned directions (corners of the cube have higher density along the axes). The resulting "rotation" is statistically biased — it preserves orthogonality but not uniformity.

**Why:** The key property is that the Gaussian density `exp(-||x||^2/2)` depends only on `||x||`, not on direction. Any distribution with this radial symmetry works — Gaussian is the simplest and the one with the cleanest sampling primitive in `random`. The uniform hypercube's density is constant on `[-1, 1]^D` and zero outside, which has directional structure: a point near `(1, 0, 0, ..., 0)` is inside the cube, but a point of the same norm near `(0.577, 0.577, 0.577, 0, ..., 0)` is also inside, with equal density. The radii differ but both are allowed — this means the sampled direction is NOT uniform on the sphere. For TurboQuant, a non-uniform rotation breaks the data-oblivious guarantee: the MSE bound derived in the paper assumes Haar-uniform R, and any other distribution gives data-dependent worst-case performance.

**Script reference:** `03-systems/microturboquant.py`, lines 109-142 (random_rotation via Gram-Schmidt), lines 101-104 (gaussian_sample helper), lines 145-156 (orthogonality_error check), lines 421-424 (main asserts orthogonality < 1e-10)

</details>

---

### Challenge 3: QJL Signed-Bit Inner-Product Estimator

**Setup:** `qjl_signs` (lines 214-223) stores `K` sign bits per vector: `sign((S @ x)_k)` for each row of a fixed Gaussian `S`. `qjl_estimate_inner_product` (lines 226-245) returns `agreement * pi / 2.0` where `agreement` is the mean of `sign_a[k] * sign_b[k]` across the K sign bits. The comment cites the identity `E[sign(<g,a>) * sign(<g,b>)] = 1 - 2 * arccos(rho) / pi` for Gaussian `g`.

**Question:** Two vectors have cosine similarity `rho = 0.5`. Using the exact identity, what is `E[agreement]`? Using the linear approximation `rho ~ (pi/2) * agreement`, what does the estimator return? How much bias does the linear approximation introduce, and when does it matter?

<details>
<summary>Reveal Answer</summary>

**Answer:** Exact: `E[agreement] = 1 - 2 * arccos(0.5) / pi = 1 - 2 * (pi/3) / pi = 1 - 2/3 = 0.333`. Linear estimator: `rho_est = 0.333 * pi / 2 ≈ 0.523`. The true `rho` is `0.5`, so the linear estimate overshoots by `0.023` — a `+4.7%` relative bias. For near-orthogonal vectors (`rho` near 0) the bias vanishes: `arccos(0) = pi/2`, so `E[agreement] = 0` and the linear estimator returns exactly 0. The bias grows as `rho` moves toward `±1`.

**Why:** The exact Grothendieck-identity-derived estimator inverts `arccos` explicitly: `rho = cos(pi/2 - pi/2 * agreement) = sin(pi/2 * agreement)`. The script uses the small-angle approximation `sin(x) ~ x` which is exact at `x=0` and has cubic error. For production retrieval over dense embeddings where most pairs are near-orthogonal (typical for high-dim normalized vectors), the bias is negligible. For applications where high-similarity pairs matter (near-duplicate detection, k-nearest neighbors with similar items), the bias is systematically positive and matters for ranking. The paper's production-grade QJL variant uses the full inverse at one trig call per pair; the script's signpost comment flags this as the production fix.

**Script reference:** `03-systems/microturboquant.py`, lines 214-223 (qjl_signs: sign bits of Gaussian projection), lines 226-245 (qjl_estimate_inner_product with pi/2 scaling), lines 384-407 (qjl_demo: empirical mean signed error), docstring of qjl_estimate_inner_product on lines 229-241 (math identity and signpost)

</details>

---

### Challenge 4: The Orthogonality-Error Assertion

**Setup:** `main` (lines 421-424) calls `random_rotation(32)` then asserts `orthogonality_error(R) < 1e-10`. The error metric is `max |R^T R - I|` across all `D*D` entries (lines 145-156).

**Question:** Why is `1e-10` the threshold and not `0`? Why is it not `1e-15` (machine epsilon for f64)? And what would go wrong in `turboquant_decode` (lines 204-211) if the assertion were replaced with `< 1e-3` and silently passed a slightly non-orthogonal matrix?

<details>
<summary>Reveal Answer</summary>

**Answer:** `1e-10` accounts for accumulated floating-point error in Gram-Schmidt. Each inner product and subtraction introduces relative error on the order of `eps = 2^-52 ≈ 2.2e-16`. Across `D=32` columns with `D-1` projections each, the accumulated error is roughly `D^2 * eps ≈ 32^2 * 2.2e-16 ≈ 2e-13`. The `1e-10` margin is roughly 3 orders of magnitude above this, accommodating numerically adversarial Gaussian draws. `1e-15` is too tight — classical Gram-Schmidt (not the modified variant) can accumulate error closer to `D * eps ~ 10^-14` even on well-conditioned inputs. If the threshold were `1e-3`: `R^T R ≈ I + epsilon_matrix` where `||epsilon|| < 1e-3`. Then `turboquant_decode` returns `R^T y_hat ≈ R^T (R x + quant_error) = (I + epsilon) x + R^T quant_error`. The `epsilon * x` term is a systematic reconstruction bias independent of quantization — a "rotation-back-is-wrong" error that would appear at every bit-width including 32-bit, making the IP-MSE non-zero even before any bits are spent.

**Why:** The entire TurboQuant guarantee depends on `R^T R = I` exactly, because this is what ensures `||x_hat - x||_2 = ||y_hat - y||_2` (rotation preserves L2 norm of any vector, including the error vector). A non-orthogonal `R` breaks this isometry: the quantization error is no longer spread uniformly across coordinates, and the MSE bound from the paper's analysis no longer applies. The hard assertion enforces this invariant at construction time, so later code can trust it without re-checking. The threshold also catches degenerate Gaussian draws (a column nearly in the span of earlier columns gets amplified by Gram-Schmidt renormalization), which is why `random_rotation` raises an exception rather than returning a questionable matrix on line 138.

**Script reference:** `03-systems/microturboquant.py`, lines 109-142 (random_rotation with degeneracy check at line 138), lines 145-156 (orthogonality_error: max entry-wise deviation from I), lines 204-211 (turboquant_decode applies R^T assuming exact orthogonality), lines 421-424 (main asserts < 1e-10 before using R)

</details>
