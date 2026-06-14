**Session 683 Update (2026-06-15) — FEATURE MODE:**

✅ **NoncentralBeta Distribution** — 87th total, 71st continuous — commit 3f08a28
- **Mode**: FEATURE MODE (counter: 683)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: NoncentralBeta(α, β, λ) — Poisson mixture of central Beta distributions
  * Parameters: α > 0 (shape 1), β > 0 (shape 2), λ ≥ 0 (noncentrality; λ=0 → Beta(α, β))
  * Support: [0, 1]
  * PDF: Σ_{j=0}^{MAX} w_j · beta_pdf(x; α+j, β) where w_j = Poisson(λ/2) weights — O(MAX_POISSON_TERMS)
  * CDF: Σ_j w_j · I_x(α+j, β) (regularizedBetaI) — O(MAX_POISSON_TERMS)
  * Mean: Σ_j w_j · (α+j)/(α+β+j) — exact via mixture sum
  * Variance: E[X²]-mean² where E[X²]=Σ_j w_j·(α+j)(α+j+1)/((α+β+j)(α+β+j+1))
  * Sample: J ~ Poisson(λ/2) via poissonKnuth; Gamma(α+J,1)/(Gamma(α+J,1)+Gamma(β,1))
  * MAX_POISSON_TERMS = 250; private betaPdfAt helper with log-space computation
  * Key exact values:
    - NoncentralBeta(1,1,2): PDF(0.5)≈0.90980; CDF(0.5)≈0.30327; Mean≈0.63212
    - NoncentralBeta(2,3,0)=Beta(2,3): CDF(0.5)=0.6875; Mean=0.4; Var=0.04
- **Tests**: 60 tests passing
- **Distribution count**: 87 total (71 continuous + 16 discrete)
- **Next Priority**: KolmogorovSmirnov or SinhNormal or ExponentialPower variants

**Session 682 Update (2026-06-14) — FEATURE MODE:**

✅ **HyperbolicSecant Distribution** — 86th total, 70th continuous — commit 2f0bb5a
- **Mode**: FEATURE MODE (counter: 682)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: HyperbolicSecant(μ, σ) — symmetric leptokurtic distribution
  * Parameters: μ ∈ ℝ (location), σ > 0 (scale = standard deviation)
  * Support: (−∞, +∞)
  * PDF: (1/(2σ))·sech(π(x-μ)/(2σ)) = 1/(2σ·cosh(π(x-μ)/(2σ))) — O(1)
  * logPDF: −ln(2σ) − ln(cosh(π(x-μ)/(2σ))) — O(1)
  * CDF: (2/π)·arctan(exp(π(x-μ)/(2σ))) — O(1)
  * Quantile: μ + (2σ/π)·ln(tan(πp/2)) — exact closed form — O(1)
  * Mean: μ (exact) — O(1)
  * Variance: σ² (exact; from MGF M(t)=sec(σt)·exp(μt)) — O(1)
  * Mode: μ — O(1)
  * Entropy: ≈ ln(4σ) via 200-point Simpson quadrature — O(200)
  * Sample: exact inverse CDF X = μ + (2σ/π)·ln(tan(πU/2)) — O(1)
  * Key values (μ=0,σ=1): pdf(0)=0.5; cdf(0)=0.5; quantile(0.5)=0; entropy≈1.3863=ln(4)
  * Characteristic function: φ(t)=sech(σt)·exp(iμt); excess kurtosis=2
  * Applications: financial returns modeling, Bayesian statistics, random matrix theory
- **Tests**: 34 HyperbolicSecant tests + 4180 total tests passing
- **Distribution count**: 86 total (70 continuous + 16 discrete)
- **Next Priority**: NoncentralBeta or Kolmogorov-Smirnov or ExponentialPower (already done as GeneralizedNormal)

**Session 681 Update (2026-06-14) — FEATURE MODE:**

✅ **ExponentialModifiedGaussian Distribution** — 85th total, 69th continuous — commit 32b470b
- **Mode**: FEATURE MODE (counter: 681)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: ExponentialModifiedGaussian(μ, σ, λ) — X = Normal(μ,σ²) + Exponential(λ)
  * Parameters: μ ∈ ℝ (Normal mean), σ > 0 (Normal std), λ > 0 (Exp rate)
  * Support: (-∞, +∞) — right-skewed
  * PDF: f(x) = λ·exp(λ(μ+λσ²/2−x))·Φ((x−μ−λσ²)/σ) — O(1)
  * logPDF: ln(λ) + λ(μ+λσ²/2−x) + ln(Φ((x−μ−λσ²)/σ)) — O(1)
  * CDF: Φ((x−μ)/σ) − exp(λ(μ+λσ²/2−x))·Φ((x−μ−λσ²)/σ) — O(1)
  * Mean: μ + 1/λ (exact) — O(1)
  * Variance: σ² + 1/λ² (exact) — O(1)
  * Mode: μ + λσ² + σ·z* where φ(z*)/Φ(z*) = λσ (inverse Mills ratio, bisection) — O(1)
  * Entropy: 200-point Simpson quadrature — O(N)
  * Quantile: bisection with dynamic bracket expansion, tol 1e-10, 100 iter — O(log(1/tol))
  * Sample: Y + Z where Y~N(μ,σ) via Box-Muller, Z~Exp(λ) via −ln(U)/λ
  * Key values (μ=0,σ=1,λ=1): mean=1, var=2; pdf(0)≈0.26157; cdf(1)≈0.53808
  * Applications: cognitive science (reaction times), chromatography, finance
- **Tests**: 44 tests passing
- **Distribution count**: 85 total (69 continuous + 16 discrete)
- **Next Priority**: HyperbolicSecant, NoncentralBeta, or ExponentialPower

**Session 678 Update (2026-06-14) — FEATURE MODE:**

✅ **GeneralizedNormal Distribution** — 83rd total, 67th continuous — commit 2329b39
- **Mode**: FEATURE MODE (counter: 678)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: GeneralizedNormal(μ, α, β) — Generalized Gaussian Distribution
  * Parameters: μ ∈ ℝ (location), α > 0 (scale), β > 0 (shape)
  * Support: (-∞, +∞)
  * PDF: β/(2·α·Γ(1/β))·exp(-(|x-μ|/α)^β) — O(1)
  * logPDF: ln(β) - ln(2) - ln(α) - logGamma(1/β) - (|x-μ|/α)^β — O(1)
  * CDF: 0.5 + sign(x-μ)·0.5·P(1/β, (|x-μ|/α)^β) via regularizedGammaP — O(1)
  * Quantile: bisection (100 iterations, 1e-10 tol), bracket expansion for heavy tails
  * Mean/Mode/Median: μ (exact by symmetry) — O(1)
  * Variance: α²·Γ(3/β)/Γ(1/β) = α²·exp(logGamma(3/β)-logGamma(1/β)) — O(1)
  * Entropy: 1/β - ln(β) + ln(2·α) + logGamma(1/β) — O(1)
  * Sample: G~Gamma(1/β,1) via Marsaglia-Tsang, S=±1 → μ + α·S·G^(1/β)
  * Special cases: β=1→Laplace(μ,α); β=2→Normal(μ,α/√2); β→∞→Uniform
  * Key values (μ=0,α=1,β=2): pdf(0)=1/√π≈0.5642; cdf(0)=0.5; var=0.5
  * Key values (μ=0,α=1,β=1): pdf(0)=0.5; var=2 (matches Laplace)
  * Uses regularizedGammaP for CDF, logGamma for normalization
- **Tests**: 49 tests passing
- **Distribution count**: 83 total (67 continuous + 16 discrete)
- **Next Priority**: PowerNormal, BetaRectangular, or Kumaraswamy-Normal

**Session 677 Update (2026-06-14) — FEATURE MODE:**

✅ **LogitNormal Distribution** — 82nd total, 66th continuous — commit 1986ac7
- **Mode**: FEATURE MODE (counter: 677)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: LogitNormal(μ, σ): Y s.t. logit(Y) ~ Normal(μ, σ²)
  * Parameters: μ ∈ ℝ (location), σ > 0 (scale)
  * Support: (0, 1)
  * PDF: f(y) = 1/(σ·y·(1-y)·√(2π))·exp(-(logit(y)-μ)²/(2σ²)) — O(1)
  * CDF: Φ((logit(y)-μ)/σ) — exact closed form, O(1)
  * Quantile: sigmoid(μ + σ·Φ⁻¹(p)) — exact closed form, O(1)
  * Median: sigmoid(μ) — exact, O(1)
  * Mean: E[sigmoid(X)] for X~N(μ,σ²) — 200-pt Simpson quadrature
  * Variance: E[sigmoid(X)²] - E[sigmoid(X)]² — 200-pt Simpson
  * Mode: 500-point grid scan (handles bimodal case for large σ where σ²>2)
  * Entropy: H = 0.5·ln(2πeσ²) + E[X - 2·softplus(X)] — 200-pt Simpson
  * Sample: X~N(μ,σ) via Box-Muller, Y=sigmoid(X)
  * Key values (μ=0,σ=1): cdf(0.5)=0.5; pdf(0.5)=4/√(2π)≈1.596; mean=0.5; var≈0.0434
  * Symmetry: pdf(y)=pdf(1-y) and cdf(y)+cdf(1-y)=1 for μ=0
  * CRITICAL: variance(0,1)≈0.0434 (NOT 0.0862 as test-writer initially guessed)
  * Uses local helpers: normalPdf, normalCdf, normalQuantile, sigmoidFn, softplus, logitFn
  * Uses global erfInv (line 1970) via normalQuantile
- **Tests**: 51 tests, all passing
- **Distribution count**: 82 total (66 continuous + 16 discrete)
- **Next Priority**: GeneralizedNormal or BetaPrimeType2 or Lindley variant

**Session 676 Update (2026-06-13) — FEATURE MODE:**

✅ **GeneralizedExtremeValue Distribution** — 81st total, 65th continuous — commits cdc42a2, 9329537
- **Mode**: FEATURE MODE (counter: 676)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: GeneralizedExtremeValue(μ, σ, ξ) — unified extreme value family
  * Parameters: μ ∈ ℝ (location), σ > 0 (scale), ξ ∈ ℝ (shape)
  * Special cases: ξ→0 = Gumbel; ξ>0 = Fréchet type; ξ<0 = Weibull type (bounded)
  * Support: ξ=0: ℝ; ξ>0: (μ−σ/ξ,+∞); ξ<0: (−∞, μ−σ/ξ)
  * Gumbel threshold: |ξ|<1e-10 → use ξ=0 formulas to avoid division-by-zero
  * CDF: ξ=0: exp(-exp(-z)); ξ≠0: exp(-t^{-1/ξ}), t=1+ξ(x−μ)/σ
  * PDF: ξ=0: (1/σ)·exp(-z)·exp(-exp(-z)); ξ≠0: (1/σ)·t^{-1/ξ-1}·exp(-t^{-1/ξ})
  * Quantile: ξ=0: μ−σ·ln(-lnp); ξ≠0: μ+σ/ξ·((-lnp)^{-ξ}−1)
  * Mean: ξ=0: μ+σγ; ξ<1: μ+σ(Γ(1-ξ)-1)/ξ; ξ≥1: +∞
  * Variance: ξ=0: σ²π²/6; ξ<0.5: σ²(Γ(1-2ξ)-Γ²(1-ξ))/ξ²; ξ∈[0.5,1): +∞; ξ≥1: NaN
  * Mode: ξ=0: μ; ξ≠0: μ+σ((1+ξ)^{-ξ}-1)/ξ
  * Entropy: ln(σ)+(1+ξ)γ+1 [unified formula for all ξ]
  * Sample: inverse CDF via quantile(U), U~Uniform(0,1)
  * Key values (0,1,0) Gumbel: cdf(0)=e^{-1}≈0.36788; mean=γ≈0.57722; var=π²/6≈1.64493
  * Key values (0,1,1) Fréchet: cdf(1)=e^{-0.5}≈0.60653; mean=+∞
  * Key values (0,1,-0.5) Weibull: cdf(1)=e^{-0.25}≈0.77880; mean≈0.22754; ub=2
  * γ = 0.5772156649015328 (Euler–Mascheroni constant)
- **Tests**: 49 tests, all passing
- **Distribution count**: 81 total (65 continuous + 16 discrete)
- **Next Priority**: LogitNormal or BetaOfSecondKind or GeneralizedNormal

**Session 675 Update (2026-06-13) — STABILIZATION MODE:**

✅ **Test Quality Audit + Cross-Compilation** — commit 4063b1f
- **Mode**: STABILIZATION MODE (counter: 675)
- **CI Status**: GREEN; 0 open issues
- **Cross-Compilation**: ✅ All 6 targets pass (x86_64-linux, aarch64-linux, x86_64-macos, aarch64-macos, x86_64-windows, wasm32-wasi)
- **Test Quality Audit**: Added 4 meaningful tests for NoncentralF and ReciprocalInverseGaussian
  * NoncentralF: exact variance at λ=0 = central F formula (1.5625 for d1=4,d2=10); variance increases monotonically with λ
  * RIG: RIG-IG duality (cdf_RIG(y) = 1 - cdf_IG(1/y)) tested at 4 points; variance decreases with λ (3→1→0.375)
- **Distribution count**: 80 total (64 continuous + 16 discrete) — no new distributions
- **Next Priority**: GeneralizedExtremeValue (GEV) or BetaPrimeOfSecondKind or LogitNormal

**Session 674 Update (2026-06-13) — FEATURE MODE:**

✅ **ReciprocalInverseGaussian Distribution** — 80th total, 64th continuous — commit 0f4b4ea
- **Mode**: FEATURE MODE (counter: 674)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: ReciprocalInverseGaussian(μ, λ) — Y=1/X where X~IG(μ,λ)
  * Parameters: μ>0 (location), λ>0 (shape)
  * Support: (0, ∞)
  * PDF: √(λ/(2πy)) · exp(-λ(1-μy)²/(2μ²y))
  * logPDF expanded: 0.5·ln(λ)-0.5·ln(2π)-0.5·ln(y) - λ/(2μ²y) + λ/μ - λy/2
  * CDF: CLOSED FORM — 1 - Φ(z1) - exp(2λ/μ)·Φ(-z2) [= 1 - IG_CDF(1/y; μ, λ)]
    where z1=√(λy)·(1/(μy)-1), z2=√(λy)·(1/(μy)+1)
    overflow guard: when 2λ/μ > 500, compute in log-space
  * Mean: 1/μ + 1/λ (exact closed form via GIG(-1/2) moment formula)
  * Variance: 1/(μλ) + 2/λ² (exact; derived from GIG(-1/2) second moment)
  * Mode: (-1 + √(1 + 4λ²/μ²)) / (2λ) (from d/dx log f = 0: quadratic λy²+y-λ/μ²=0)
  * Entropy: numerical Simpson 1000 pts, adaptive upper bound
  * Sample: 1/X via Michael-Schucany-Haas for IG(μ,λ)
  * Key: RIG(1,1): pdf(1)=1/√(2π)≈0.39894; mean=2; var=3; mode=(-1+√5)/2≈0.618; CDF(1)≈0.332
  * Key: RIG(2,4): mean=0.75; var=0.25; mode≈0.390
  * GIG connection: RIG(μ,λ) = GIG(p=1/2, a=λ/μ², b=λ) [p+1=3/2 for moments]
- **Tests**: 51 tests, all passing
- **Distribution count**: 80 total (64 continuous + 16 discrete)
- **Next Priority**: GeneralizedExtremeValue (GEV) or BetaPrimeOfSecondKind or LogitNormal

**Session 673 Update (2026-06-13) — FEATURE MODE:**

✅ **NoncentralF Distribution** — 79th total, 63rd continuous — commit 385d666
- **Mode**: FEATURE MODE (counter: 673)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: NoncentralF(d1, d2, λ) — Poisson mixture of central F distributions
  * Parameters: d1>0 (num df), d2>0 (denom df), λ≥0 (noncentrality; 0→central F)
  * Support: [0, ∞)
  * PDF/CDF: Σ_j w_j·F(d1+2j, d2) where w_j=exp(-λ/2)(λ/2)^j/j! (Poisson weights)
  * Max 250 Poisson terms; break early when w_j < 1e-15
  * Mean: d2·(d1+λ)/(d1·(d2-2)) for d2>2; NaN for d2≤2
  * Variance: 2(d2/d1)²[(d1+λ)²+(d1+2λ)(d2-2)] / [(d2-2)²(d2-4)] for d2>4; Inf for 2<d2≤4
  * Sample: J~Poisson(λ/2) via Knuth (normal approx for λ>60), then X~ChiSq(d1+2J)/Y~ChiSq(d2)
  * λ=0 identity: NCF(2,4,0) CDF(1) = I_{1/3}(1,2) = 5/9 ≈ 0.55556 ✓
  * λ=0 identity: NCF(2,4,0) PDF(1) = 64/216 ≈ 0.29630 ✓
  * Finite-difference test: pdf ≈ d(cdf)/dx to 1e-3 tolerance ✓
  * Helper naming: `noncentralFGammaSample` (avoids conflict with GeneralizedGamma's `gammaSampleMT`)
  * Helper naming: `poissonKnuth` for Poisson sampling
- **Tests**: 43 tests, all passing
- **Distribution count**: 79 total (63 continuous + 16 discrete)
- **Next Priority**: ReciprocalInverseGaussian or GeneralizedExtremeValue or BetaDistributionOfKind2

**Session 672 Update (2026-06-13) — FEATURE MODE:**

✅ **GeneralizedGamma Distribution** — 78th total, 62nd continuous — commit acf055b
- **Mode**: FEATURE MODE (counter: 672)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: GeneralizedGamma(a, d, p) — Stacy's generalized gamma family
  * Parameters: a > 0 (scale), d > 0 (shape/index), p > 0 (power/shape)
  * Support: (0, ∞)
  * PDF: (p / (a^d · Γ(d/p))) · x^(d-1) · exp(-(x/a)^p)
  * CDF: regularizedGammaP(d/p, (x/a)^p) — exact O(1) via series/CF
  * Quantile: bisection; p=0→0, p=1→+∞; adaptive upper bound expansion
  * Mean: a · exp(logΓ((d+1)/p) - logΓ(d/p))
  * Variance: a²·exp(logΓ((d+2)/p)-logΓ(d/p)) - mean²
  * Mode: a·((d-1)/p)^{1/p} for d>1; 0.0 for d≤1
  * Entropy: log(a/p) + logΓ(d/p) + d/p - ((d-1)/p)·ψ(d/p)
  * Sample: a · Gamma(d/p, rate=1)^{1/p} via Marsaglia-Tsang with boost trick
  * Special cases:
    - p=1 → Gamma(scale=a, shape=d): pdf(1;1,2,1)=e^{-1}≈0.36788
    - d=p → Weibull(scale=a, shape=p): pdf(1;2,2,2)=0.5·exp(-0.25)≈0.38941
    - d=1,p=2 → HalfNormal-like: pdf(1;1,1,2)=(2/√π)·e^{-1}≈0.41511
    - (a=1,d=1,p=1)=Exponential: mean=1, var=1, entropy=1
  * CRITICAL: GGamma(1,1,2) pdf(1) = (2/√π)·exp(-1)≈0.41511 NOT 0.73576
    (test-writer erroneously computed as Rayleigh; actual is HalfNormal with x^0 term)
- **Tests**: 77 tests passing
- **Distribution count**: 78 total (62 continuous + 16 discrete)
- **Next Priority**: NoncentralF or ReciprocalInverseGaussian or GeneralizedExtremumValue

**Session 671 Update (2026-06-13) — FEATURE MODE:**

✅ **NoncentralT Distribution** — 77th total, 61st continuous — commit 4689411
- **Mode**: FEATURE MODE (counter: 671)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: NoncentralT(ν, δ) — T = (Z+δ)/√(V/ν) where Z~N(0,1), V~χ²(ν)
  * Parameters: ν > 0 (degrees of freedom), δ ∈ ℝ (noncentrality)
  * Support: (−∞, +∞)
  * PDF/CDF: 300-point composite Simpson over ∫₀^∞ φ/Φ(t√(v/ν)−δ)·f_{χ²(ν)}(v) dv
  * CRITICAL insight: F(0;ν,δ) = Φ(−δ) exactly (t=0 → phi_arg = −δ, independent of v)
  * Mean: δ√(ν/2)·Γ((ν-1)/2)/Γ(ν/2) for ν>1, NaN for ν≤1
  * Variance: ν(1+δ²)/(ν-2) − mean² for ν>2, Inf for 1<ν≤2, NaN for ν≤1
  * Sample: Box-Muller Z + Marsaglia-Tsang Gamma(ν/2) for V; T=(Z+δ)/√(V/ν)
  * Symmetry: F(t;ν,δ)+F(−t;ν,−δ)=1 (exact math, ~1e-3 numerical tolerance)
  * quantile(p=0)=−∞, quantile(p=1)=+∞; bisection 100 iter, adaptive bound
  * Private gammaSample helper in struct (avoids module-level collision)
  * Zig gotcha: for-loop ranges require usize (non-negative) — use array literals for negative ranges
- **Tests**: 69 tests all passing (exit code 0)
- **Distribution count**: 77 total (61 continuous + 16 discrete)
- **Next Priority**: GeneralizedGamma or NoncentralF

**Session 670 Update (2026-06-12) — STABILIZATION MODE:**

✅ **All Systems Green** — commits 90252b7, 7fdfa0f
- **Mode**: STABILIZATION MODE (counter: 670)
- **CI Status**: GREEN (5/5 recent runs all success)
- **Cross-Compilation**: All 6 targets pass (x86_64-linux, aarch64-linux, x86_64-macos, aarch64-macos, x86_64-windows, wasm32-wasi)
- **Test Quality Audit**:
  * GompertzMakeham: 67→71 tests (+NaN quantile guard, +sample mean convergence, +mean monotonicity in c, +sf boundary)
  * Muth: 67→70 tests (+sample mean convergence to 1.0, +sf at negative x, +canonical mean=1 for 6 kappas)
- **Distribution count**: 76 total (60 continuous + 16 discrete)
- **Next Priority**: NoncentralT or GeneralizedGamma (FEATURE sessions)

**Session 669 Update (2026-06-12) — FEATURE MODE:**

✅ **Muth Distribution** — 76th total, 60th continuous — commit 43d8508
- **Mode**: FEATURE MODE (counter: 669)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: Muth(κ) — survival model with unusual property E[X]=1 for all κ
  * Parameter: κ ∈ (0, 1] (rejects κ≤0, κ>1, NaN)
  * Support: [0, ∞)
  * Hazard: h(x) = exp(κx) − κ
  * logSf: κx − (exp(κx) − 1)/κ
  * PDF: (exp(κx) − κ) · exp(κx − (exp(κx)−1)/κ)
  * CDF: 1 − exp(κx − (exp(κx)−1)/κ)
  * Mean: 1 (closed form — proven via substitution u=exp(κx)/κ: ∫S dx = e^(1/κ) · e^(-1/κ) = 1)
  * Mode: κ > (3−√5)/2 ≈ 0.382 → ln(κ(3+√5)/2)/κ; else 0
    (derived from critical point u²−3κu+κ²=0 → u=κ(3±√5)/2)
  * Variance/Entropy: numerical Simpson 1000 pts
  * Key values: pdf(0;κ=0.5)=0.5; pdf(1;κ=0.5)≈0.5177; cdf(1;κ=0.5)≈0.5493
  * mode(κ=0.5)≈0.5389; mode(κ=1)≈0.9624; mode(κ=0.3)=0
  * pdf(0;κ=1)=0 (hazard=0 at origin); logpdf(0;κ=1)=−∞
- **Tests**: 67 tests all passing (exit code 0)
- **Distribution count**: 76 total (60 continuous + 16 discrete)
- **Next Priority**: NoncentralT or GeneralizedGamma

**Session 668 Update (2026-06-12) — FEATURE MODE:**

✅ **GompertzMakeham Distribution** — 75th total, 59th continuous — commit 4213bb0
- **Mode**: FEATURE MODE (counter: 668)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: GompertzMakeham(c, eta, b) — survival/mortality model with Makeham term
  * Parameters: c ≥ 0 (background hazard), eta > 0 (scale), b > 0 (growth rate)
  * Support: [0, ∞); c=0 reduces to Gompertz(eta, b)
  * Hazard: h(x) = c + eta·exp(b·x)
  * PDF: (c + eta·exp(b·x)) · exp(−c·x − (eta/b)·(exp(b·x)−1))
  * CDF: 1 − exp(−c·x − (eta/b)·(exp(b·x)−1)); = 0 for x < 0
  * Quantile: bisection with adaptive upper bound (doubles until sf < 1e-12, max 200 iters)
  * Mode: c=0,η≤1: −ln(η)/b; c=0,η>1: 0; c>0,b≥4c: max(0,ln(u/η)/b) where u=(b−2c−√(b²−4bc))/2; c>0,b<4c: 0
  * Mean: E[X] = ∫S(x)dx numerical (Simpson 1000 pts, adaptive upper)
  * Variance: 2∫x·S(x)dx − mean² (Simpson)
  * Entropy: −∫f·ln(f)dx (Simpson, guard f≤0 → contrib=0)
  * CRITICAL: centered finite difference at x=0 (boundary) gives pdf/2, not pdf — use interior points
  * Key values: pdf(0;c=0,η=1,b=1)=1.0; pdf(0;c=0.5,η=1,b=1)=1.5; cdf(1;c=0,η=1,b=1)=1−exp(−(e−1))≈0.82079
  * mode(c=0,η=0.5,b=1)=ln(2)≈0.6931; mode(c=0,η=1,b=1)=0.0; mode(c=0,η=2,b=1)=0.0
- **Tests**: 67 tests all passing (exit code 0)
- **Distribution count**: 75 total (59 continuous + 16 discrete)
- **Next Priority**: NoncentralT, Muth, or GeneralizedGamma

**Session 667 Update (2026-06-12) — FEATURE MODE:**

✅ **RaisedCosine Distribution** — 74th total, 58th continuous — commit a8e0715
- **Mode**: FEATURE MODE (counter: 667)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: RaisedCosine(μ, s) — symmetric bounded distribution
  * Parameters: mu (location, any real), s (scale, s > 0)
  * Support: [μ−s, μ+s] — bounded, symmetric around μ
  * PDF: (1/(2s))·(1 + cos(π(x−μ)/s)); peaks at μ (pdf(μ) = 1/s), 0 at boundaries
  * logPDF: −ln(2s) + ln(1+cos(π(x−μ)/s)); −∞ for x outside support (including boundaries)
  * CDF: (1/2)·(1 + (x−μ)/s + sin(π(x−μ)/s)/π) — closed form O(1)
  * Quantile: bisection (64 steps); p=0→μ−s, p=1→μ+s; error.InvalidProbability for !(p>=0&&p<=1)
  * Mean: μ (exact, by symmetry); Variance: s²(1/3 − 2/π²); Mode: μ; Median: μ
  * Entropy: ln(4s) − 1 (exact closed form; can be negative for s < 1/4)
  * Sample: inverse CDF via bisection using rng.float(T)
  * CRITICAL: pdf(μ) = 1/s (not 1/(2s)!) because (1+cos(0)) = 2 cancels denominator factor
  * CRITICAL: logpdf(μ; s=1) = 0 (not -ln(2)) because -ln(2) + ln(2) = 0
  * Key values (μ=0, s=1): pdf(0)=1.0; pdf(0.5)=0.5; pdf(±1)=0; cdf(0)=0.5; var≈0.13069; entropy≈0.38629
  * Key values (μ=2, s=3): pdf(2)=1/3; cdf(2)=0.5; var≈1.17623; entropy≈1.48491
  * Symmetry: cdf(μ−d) + cdf(μ+d) = 1 (exact); pdf(μ−d) = pdf(μ+d) (exact)
- **Tests**: 63 tests all passing (exit code 0)
- **Distribution count**: 74 total (58 continuous + 16 discrete)
- **Next Priority**: GompertzMakeham, NoncentralT, or PowerFunction distribution

**Session 666 Update (2026-06-12) — FEATURE MODE:**

✅ **LogLaplace Distribution** — 73rd total, 57th continuous — commit 180fa3e
- **Mode**: FEATURE MODE (counter: 666)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: LogLaplace(μ, b) — X = exp(Y) where Y ~ Laplace(μ, b); support (0, ∞)
  * PDF: 1/(2bx)·exp(−|ln(x)−μ|/b); CDF split at x=exp(μ)
  * Quantile closed form; Mean: exp(μ)/(1−b²) for b<1; Entropy: 1+μ+ln(2b)
  * 61 tests passing

**Session 664 Update (2026-06-11) — FEATURE MODE:**

✅ **Benford Distribution** — 72nd total, 16th discrete — commit 813b3ac
- **Mode**: FEATURE MODE (counter: 664)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: Benford(T) — first-digit law, parameter-free distribution
  * Support: {1, 2, ..., 9}; no parameters
  * PMF: P(X=d) = log₁₀(1 + 1/d) for d∈{1..9}
  * CDF: F(d) = log₁₀(d+1); 0 for d<1; 1 for d>9
  * Mean ≈ 3.44024; Variance ≈ 6.05653; Mode = 1
  * Entropy ≈ 1.9934 nats (NOT 2.19322 — critical: my scratchpad was wrong due to arithmetic error; correct value is ~1.993)
  * Quantile: smallest d s.t. CDF(d)≥p; error.InvalidParameter for p<0, p>1, !isFinite
  * Sample: inverse CDF with uniform random variable
  * validate(): always passes (no parameters)
- **Note**: Entropy = -Σ log₁₀(1+1/d)·ln(log₁₀(1+1/d)) ≈ 1.9934 nats (verified algebraically and numerically)
- **Tests**: 71 tests all passing (exit code 0)
- **Distribution count**: 72 total (56 continuous + 16 discrete)
- **Next Priority**: GompertzMakeham or NoncentralT or Lindley extension

**Session 663 Update (2026-06-11) — FEATURE MODE:**

✅ **AsymmetricLaplace Distribution** — 71st total, 56th continuous — commits 8485c97, 0db7663
- **Mode**: FEATURE MODE (counter: 663)
- **CI Status**: GREEN; 0 open issues
- **Implementation**: AsymmetricLaplace(T) — two-sided exponential with different tail rates
  * Parameters: mu (location), sigma (scale, σ>0), kappa (asymmetry, κ>0)
  * κ=1 → symmetric Laplace; Support: (-∞, +∞)
  * PDF: κ/(σ(1+κ²)) · exp((x-μ)/(σκ)) for x<μ; · exp(-(x-μ)κ/σ) for x≥μ
  * CDF: κ²/(1+κ²) · exp((x-μ)/(σκ)) for x<μ; 1 - 1/(1+κ²) · exp(-(x-μ)κ/σ) for x≥μ
  * Quantile: closed-form; p_boundary = κ²/(1+κ²); p=0→-∞, p=1→+∞
  * Mean: μ + σ(1-κ²)/κ; Variance: σ²(1+κ⁴)/κ²
  * Mode: μ (always); Entropy: 1 + ln(σ(1+κ²)/κ) — closed form
  * Sample: mixture representation — left side X=μ-σκ·Exp(1) with prob κ²/(1+κ²); else X=μ+(σ/κ)·Exp(1)
  * ALD(0,1,1): pdf(0)=0.5, cdf(0)=0.5, mean=0, var=2, entropy=1+ln(2)
  * ALD(0,1,2): pdf(0)=0.4, cdf(0)=0.8, mean=-1.5, var=4.25, entropy=1+ln(2.5)
  * ALD(0,1,0.5): pdf(0)=0.4, cdf(0)=0.2, mean=1.5, var=4.25
- **Tests**: 69 tests all passing (exit code 0)
- **Distribution count**: 71 total (56 continuous + 15 discrete)
- **Next Priority**: Benford (discrete) or GompertzMakeham or NoncentralT

**Session 662 Update (2026-06-11) — FEATURE MODE:**

✅ **JohnsonSU Distribution** — 70th total, 55th continuous — commit f192783
- **Mode**: FEATURE MODE (counter: 662)
- **CI Status**: GREEN (latest run SUCCESS); 0 open issues
- **Implementation**: JohnsonSU(T) — 4-parameter sinh-normal transformation family
  * Parameters: xi (location), lambda (scale, >0), gamma (shape), delta (shape, >0)
  * Support: (-∞, +∞)
  * Transformation: Z = γ + δ·arcsinh((X-ξ)/λ) ~ N(0,1); X = ξ + λ·sinh((Z-γ)/δ)
  * PDF: (δ/(λ√(2π))) · exp(-½z²) / √(1+u²) where u=(x-ξ)/λ, z=γ+δ·arcsinh(u)
  * CDF: Φ(z) = 0.5*(1+erf(z/√2)) — exact O(1)
  * Quantile: ξ + λ·sinh((Φ⁻¹(p)-γ)/δ) — exact O(1) via erfInv
  * Mean: ξ - λ·exp(1/(2δ²))·sinh(γ/δ)
  * Variance: (λ²/2)·(exp(1/δ²)-1)·(exp(1/δ²)·cosh(2γ/δ)+1)
  * Mode: bisection on d/dx log f = 0 (CRITICAL: use > 0 not < 0 for same-sign check)
  * Entropy: log(λ/δ) + ½log(2πe) + E_Z[log(cosh((Z-γ)/δ))] — numerical Simpson
  * Sample: Box-Muller → Z ~ N(0,1) → X = ξ + λ·sinh((Z-γ)/δ) — exact O(1)
  * Special case γ=0: symmetric, mean=ξ, mode=ξ; variance=(λ²/2)·(w²-1) where w=exp(1/δ²)
- **Bug fixed in bisection**: Using `> 0` (same-sign check) instead of `< 0` prevents mode from converging to boundary when d_mid=0 (d_lo*d_mid=0, which is not > 0 so hi=mid keeps mode bracket tight)
- **Test fix**: cdf boundaries require x=±1000, not x=±100 (arcsinh(100)≈5.3, Φ(-5.3)≈6e-8 > 1e-10; need arcsinh(1000)≈7.6, Φ(-7.6)≈1.5e-14 < 1e-10)
- **Tests**: 69 tests all passing (exit code 0)
- **Distribution count**: 70 total (55 continuous + 15 discrete)
- **Next Priority**: Benford (discrete) or Asymmetric Laplace or GompertzMakeham

**Session 659 Update (2026-06-11) — FEATURE MODE:**

✅ **IrwinHall Distribution** — 68th total, 53rd continuous — commit 9236d2e
- **Mode**: FEATURE MODE (counter: 659)
- **CI Status**: GREEN (3 recent runs SUCCESS); 0 open issues
- **Implementation**: IrwinHall(T) — sum of n i.i.d. Uniform(0,1) RVs
  * Parameter: n >= 1 (u32); support: [0, n]
  * PDF: (1/(n-1)!) * Σ_{k=0}^{floor(x)} (-1)^k·C(n,k)·(x-k)^(n-1) — O(n)
  * CDF: (1/n!) * Σ_{k=0}^{floor(x)} (-1)^k·C(n,k)·(x-k)^n — O(n)
  * n=1 → Uniform(0,1); n=2 → Triangular on [0,2]
  * Mean = n/2; Variance = n/12; Mode = n/2
  * Entropy: numerical Simpson's rule; Quantile: bisection; Sample: sum of n uniforms
- **Tests**: 71 tests all passing (3006 total test blocks, exit code 0)
- **Distribution count**: 68 total (53 continuous + 15 discrete)
- **Next Priority**: Benford (discrete), Johnson SU, or Bates distribution

**Session 656 Update (2026-06-10) — FEATURE MODE:**

✅ **WignerSemicircle Distribution** — 65th total, 50th continuous — commit bdfdd96
- **Mode**: FEATURE MODE (counter: 656)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: WignerSemicircle(T) — semicircle law of random matrix theory
  * Parameter: R > 0 (radius/half-width); support: [-R, R]
  * PDF: (2/(πR²))·√(R²-x²) — semicircular shape
  * CDF: 1/2 + x·√(R²-x²)/(πR²) + arcsin(x/R)/π — exact closed form
  * Quantile: 64-step bisection (no closed form)
  * Mean: 0 (symmetric); Variance: R²/4; Mode: 0; Median: 0
  * Entropy: ln(πR) - 1/2
  * Sample: rejection sampling on unit disk — (U,V) ~ Uniform([-R,R]²), accept if U²+V²≤R², return U
  * Acceptance rate: π/4 ≈ 78.5%
  * Key values: pdf(0;R=1)=2/π; pdf(0;R=2)=1/π; cdf(0;R)=0.5
  * Test fix: test-writer wrote pdf(0;R=2)=1/(2π) but correct is 1/π (formula: (2/(πR²))·R = 2/(πR))
- **Tests**: 43 tests all passing (exit code 0)
- **Distribution count**: 65 total (50 continuous + 15 discrete)
- **Next Priority**: NoncentralT, ReciprocalInverseGaussian, or IrwinHall distribution

**Session 654 Update (2026-06-10) — FEATURE MODE:**

✅ **ToppLeone Distribution** — 64th total, 49th continuous — commit 204aa23
- **Mode**: FEATURE MODE (counter: 654)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: ToppLeone(α) — bounded [0,1] distribution
  * Parameter: α > 0 (shape); support: [0, 1]
  * CDF: (2x-x²)^α = (1-(1-x)²)^α — exact closed form
  * PDF: 2α(1-x)(2x-x²)^(α-1)
  * Quantile: x = 1 - √(1 - p^(1/α)) — exact closed form; O(1) sampling via inversion
  * Mode: 1-1/√(2α-1) for α≥1; 0 for 0<α<1
  * Mean: 1 - (√π/2)·exp(logΓ(α+1) - logΓ(α+3/2))
  * Variance: E[X²] - mean² where E[X²] = 1 - √π·exp(logΓ(α+1)-logΓ(α+3/2)) + 1/(α+1)
  * Entropy: numerical Simpson's rule (no closed form)
  * Note: DistributionError.OutOfSupport added for validateValue (alongside existing OutOfDomain)
  * Special case: α=1 → CDF=2x-x²; mean=1/3; mode=0
  * Stochastic dominance: higher α → CDF shifts right (u^α decreases for u∈(0,1))
- **Tests**: 39 tests all passing (exit code 0)
- **Distribution count**: 64 total (49 continuous + 15 discrete)
- **Next Priority**: NoncentralT or ReciprocalInverseGaussian or GompertzMakehamLaw

**Session 653 Update (2026-06-10) — FEATURE MODE:**

✅ **NoncentralChiSquared Distribution** — 63rd total, 48th continuous — commit dc5109f
- **Mode**: FEATURE MODE (counter: 653)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: NoncentralChiSquared(T) — X = Σᵢ(Yᵢ+μᵢ)² where Yᵢ~N(0,1), λ=Σμᵢ²
  * Parameters: k > 0 (degrees of freedom), lambda >= 0 (noncentrality); support: (0, ∞)
  * Special case: lambda=0 → ChiSquared(k) (exact formulas used)
  * PDF/CDF: Poisson-mixture series: Σ w_j · f_χ²_{k+2j}(x) where w_j = Poisson(λ/2) PMF
  * CDF exact via regularizedGammaP(k/2+j, x/2) accumulation
  * Smart j_start: series starts at floor(λ/2 - 6√(λ/2)) to avoid underflow for large λ
  * Mean = k + λ; Variance = 2(k + 2λ); Mode ≈ max(0, k+λ-2)
  * Entropy: approximate formula (no closed form) via 0.5·log(2πe(k+λ)) + ψ(k/2)
  * Sample: M ~ Poisson(λ/2) via Knuth inter-arrival, then 2·Gamma(k/2+M) Marsaglia-Tsang
  * Bug fixes applied: (1) lambda=0 → M=0 always (Poisson rate 0); (2) boost trick for alpha<1
  * CDF exact test: NoncentralChiSquared(4,0).cdf(4) = 1-3e^{-2} ≈ 0.59399 (tolerance 1e-6)
  * CDF exact test: NoncentralChiSquared(2,0).cdf(2) = 1-e^{-1} ≈ 0.63212 (tolerance 1e-6)
- **Tests**: 37 tests all passing (exit code 0)
- **Distribution count**: 63 total (48 continuous + 15 discrete)
- **Next Priority**: NoncentralT or ReciprocalInverseGaussian or Topp-Leone

**Session 652 Update (2026-06-09) — FEATURE MODE:**

✅ **Chi Distribution** — 62nd total, 47th continuous — commit acaecb6
- **Mode**: FEATURE MODE (counter: 652)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Chi(T) — generalization of HalfNormal, Rayleigh, Maxwell-Boltzmann
  * Parameter: k > 0 (degrees of freedom, continuous); support: [0, ∞)
  * X ~ Chi(k) iff X = √(X₁²+...+Xₖ²), Xᵢ ~ N(0,1) i.i.d. (equivalently X = √(ChiSquared(k)))
  * PDF: 2^(1-k/2) · x^(k-1) · exp(-x²/2) / Γ(k/2)
  * CDF: regularizedGammaP(k/2, x²/2) — exact closed form
  * CDF exact for k=2: 1 - exp(-x²/2)
  * Quantile: bisection; NaN guard: !(p≥0 && p≤1)
  * Mean: √2 · exp(logΓ((k+1)/2) - logΓ(k/2))
  * Variance: k - mean²
  * Mode: √(k-1) for k≥1; 0 otherwise
  * Entropy: logΓ(k/2) + k/2 - 0.5·ln(2) - (k-1)/2·ψ(k/2)
  * Sample: sqrt(2·Gamma(k/2, rate=1)) via Marsaglia-Tsang + boost trick for k/2 < 1
  * Key relationships: Chi(1) = HalfNormal(σ=1); Chi(2) = Rayleigh(σ=1); Chi(3) = MaxwellBoltzmann(a=1)
  * Chi(1).mean = √(2/π); Chi(2).mean = √(π/2); Chi(3).mean = 2√(2/π)
  * Test fix: pdf(100) underflows to 0 (exp(-5000)) — test max x=15, not 100
  * Test fix: cdf tolerance 1e-10 not 1e-12 (regularizedGammaP precision)
  * Test fix: quantile roundtrip excludes x=10 (cdf→1.0 → inf); q_large > 3.0 not 5.0
- **Tests**: 37 tests all passing (exit code 0)
- **Distribution count**: 62 total (47 continuous + 15 discrete)
- **Next Priority**: Reciprocal Inverse Gaussian or NoncentralChiSquared or Gumbel-Softmax

**Session 651 Update (2026-06-09) — FEATURE MODE:**

✅ **InverseGamma Distribution** — 61st total, 46th continuous — commit e33e407
- **Mode**: FEATURE MODE (counter: 651)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: InverseGamma(T) — conjugate prior for Normal variance in Bayesian analysis
  * Parameters: α (shape) > 0, β (scale) > 0; support: (0, ∞)
  * If X ~ InverseGamma(α, β), then 1/X ~ Gamma(α, rate=β)
  * PDF: β^α/Γ(α) · x^(-α-1) · exp(-β/x)
  * CDF: 1 - P(α, β/x) where P is regularizedGammaP; SF: P(α, β/x)
  * Quantile: bisection on CDF (no closed form); NaN guard: !(p≥0 && p≤1) check
  * Mode: β/(α+1); Mean: β/(α-1) for α>1 else NaN; Variance: β²/((α-1)²(α-2)) for α>2 else NaN
  * Entropy: α + ln(β) + logGamma(α) - (1+α)·ψ(α); digamma tolerance: 1e-7 for α=3
  * Sample: Marsaglia-Tsang generates Y ~ Gamma(α, rate=β), return 1/Y; boost trick for α<1
  * Key values: InverseGamma(1,1): cdf(1)=pdf(1)=e^(-1); InverseGamma(3,1): mode=0.25, mean=0.5, var=0.25
- **Tests**: 34 tests all passing (exit code 0)
- **Distribution count**: 61 total (46 continuous + 15 discrete)
- **Next Priority**: Chi distribution or Reciprocal Inverse Gaussian

**Session 650 Update (2026-06-09) — STABILIZATION MODE:**

✅ **ALL SYSTEMS GREEN** — bug fix committed
- **Mode**: STABILIZATION (counter: 650)
- **CI Status**: ✅ GREEN (all 5 recent runs successful)
- **Open Issues**: 0
- **Cross-Compilation**: ✅ All 6 targets pass
- **Bug Fixed**: Logistic.logpdf overflow for x << μ — softplus numerically stable form applied
  * When neg_z = -(x-μ)/s > 0: use `neg_z + log1p(exp(-neg_z))` instead of `log1p(exp(neg_z))`
- **Test Quality Audit**: LogUniform, Arcsine, Logistic — all PASS (validate, validateValue, Big-O docs, exact formulas)
- **Distribution count**: 60 total (45 continuous + 15 discrete)
- **Next Priority**: InverseGamma or Chi distribution (FEATURE mode)

**Session 649 Update (2026-06-06) — FEATURE MODE:**

✅ **Logistic Distribution** — 60th total, 45th continuous — commit f9438e4
- **Mode**: FEATURE MODE (counter: 649)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Logistic(T) — symmetric distribution on full real line (-∞,+∞)
  * Parameters: mu (location, any real), s (scale, s > 0)
  * PDF: exp(-(x-μ)/s) / (s·(1+exp(-(x-μ)/s))²) = sigmoid-based
  * CDF: 1/(1+exp(-(x-μ)/s)) — the logistic sigmoid function
  * Quantile: μ + s·ln(p/(1-p)) — exact logit transform; Q(0)=-∞, Q(1)=+∞
  * Mean = Mode = Median = μ (perfectly symmetric)
  * Variance: s²·π²/3; Entropy: ln(s)+2 (can be negative for s < e^(-2)≈0.135)
  * Sample: μ + s·ln(U/(1-U)) via inverse transform
  * Cached log_s=ln(s) for logpdf and entropy; full real line support (validateValue always OK)
  * Key values: pdf(μ)=1/(4s); cdf(μ)=0.5; quantile(0.75)=μ+s·ln(3)
- **Tests**: 88 tests passing
- **Note**: Entropy corrected — can be negative for small s (test fixed from incorrect entropy>0 assumption)
- **Distribution count**: 60 total (45 continuous + 15 discrete)
- **Next Priority**: InverseGamma or Chi distribution
**Session 646 Update (2026-06-05) — FEATURE MODE:**

✅ **LogUniform Distribution** — 58th total, 43rd continuous — commit 119bab6
- **Mode**: FEATURE MODE (counter: 646)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: LogUniform(T) — Reciprocal distribution; ln(X) ~ Uniform(ln(a), ln(b))
  * Parameters: a > 0, b > a (lower/upper bounds)
  * PDF: 1/(x·ln(b/a)); monotonically decreasing on [a,b]
  * CDF: ln(x/a)/ln(b/a); Quantile: a·exp(p·ln(b/a)) — exact closed form
  * Mean: (b-a)/ln(b/a); Variance: (b²-a²)/(2·ln(b/a)) - mean²
  * Mode: a (PDF decreasing); Median: √(ab); Entropy: 0.5·ln(ab)+ln(ln(b/a))
  * Sample: inverse transform — exp(Uniform(ln(a), ln(b)))
  * Cached log_ratio=ln(b/a) for efficiency; validate() + validateValue(x)
  * Key values: (a=1,b=e) entropy=0.5; median=√e; mean=e-1
- **Tests**: 115 new tests; 2454 total in distributions.zig, all passing
- **Distribution count**: 58 total (43 continuous + 15 discrete)
- **Next Priority**: Arcsine distribution

**Session 645 Update (2026-06-05) — STABILIZATION MODE:**

✅ **ALL SYSTEMS GREEN** — commit f730fc9
- **Mode**: STABILIZATION MODE (counter: 645)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Tests**: 5460/5467 passing (7 skipped)
- **Cross-Compilation**: ✅ All 6 targets pass
- **Critical Fix**: owensT in SkewNormal had factor-of-2 error + wrong domain; fixed with change-of-variables to integrate over [0,a]
- **validate() fix**: SkewNormal.validate() used testing.expect, replaced with DistributionError returns
- **Distribution count**: 57 total (42 continuous + 15 discrete)
- **Next Priority**: LogUniform/Reciprocal or Arcsine distribution (FEATURE mode)

✅ **Lomax Distribution** — 39th distribution, 24th continuous — commit 9069095
- **Mode**: FEATURE MODE (counter: 619)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Lomax(T) — Pareto Type II, heavy-tailed on [0, ∞)
  * Parameters: lambda (λ > 0, scale), kappa (κ > 0, shape)
  * Support: x ∈ [0, ∞); used in survival analysis, queueing theory, Internet traffic
  * PDF: (κ/λ)·(1 + x/λ)^(-(κ+1)); CDF: 1 - (1 + x/λ)^(-κ)
  * SF: (1 + x/λ)^(-κ); LogPDF: ln(κ/λ) - (κ+1)·ln(1 + x/λ)
  * Quantile: λ·((1-p)^(-1/κ) - 1); fully closed-form, no special functions
  * Mean: λ/(κ-1) for κ > 1, else +∞; Variance: κλ²/((κ-1)²(κ-2)) for κ > 2
  * Mode: always 0 (PDF monotonically decreasing); Median: λ·(2^(1/κ) - 1)
  * Entropy: ln(λ/κ) + 1 + 1/κ
  * Sample: λ·(U^(-1/κ) - 1) via inverse transform (U ~ Uniform(0,1))
- **Tests**: 65 new tests all passing (exit code 0)
- **Distribution count**: 39 total (24 continuous + 15 discrete)
- **Next Priority**: Gompertz or Rice distribution

**Session 618 Update (2026-06-01) — FEATURE MODE:**

✅ **Lévy Distribution** — 38th distribution, 23rd continuous — commit a6fe36d
- **Mode**: FEATURE MODE (counter: 618)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Levy(T) — one-sided heavy-tailed stable distribution (α=1/2)
  * Parameters: mu (location, any real), c (scale c > 0); support (μ, ∞)
  * PDF: sqrt(c/(2π)) · exp(-c/(2(x-μ))) / (x-μ)^1.5
  * CDF: erfc(sqrt(c/(2(x-μ)))) = 1 - erf(sqrt(c/(2(x-μ))))
  * Quantile: μ + c/(2·erfInv(1-p)²); p=0 → μ, p=1 → ∞
  * Mean: +∞; Variance: +∞ (heavy-tailed stable)
  * Mode: μ + c/3; Median: μ + c/(2·erfInv(0.5)²)
  * Entropy: (1 + 3γ + ln(16πc²)) / 2 (γ = Euler-Mascheroni ≈ 0.5772)
  * Sample: μ + c/Z² where Z ~ N(0,1) via Box-Muller
  * Uses existing erf/erfInv private helpers
- **Tests**: 73 new tests all passing (exit code 0)
- **Distribution count**: 38 total (23 continuous + 15 discrete)
- **Next Priority**: Gompertz, Rice, or Lomax distribution

**Session 616 Update (2026-05-31) — FEATURE MODE:**

✅ **Maxwell-Boltzmann Distribution** — 36th distribution, 21st continuous — commit 7746d6b
- **Mode**: FEATURE MODE (counter: 616)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: MaxwellBoltzmann(T) — particle speed in ideal gas; Chi(3) scaled by a
  * Parameter: a > 0 (scale; a = sqrt(kT/m) in kinetic theory)
  * Support: [0, ∞); characterization: sqrt(X²+Y²+Z²), X,Y,Z ~ iid N(0,a²)
  * PDF: sqrt(2/π) · x²/a³ · exp(-x²/(2a²)); CDF: closed form via erf
  * Quantile: Newton-Raphson (no closed form)
  * Mean: 2a·sqrt(2/π) ≈ 1.5958a; Mode: a·sqrt(2); Variance: a²(3-8/π)
  * Entropy: ln(a·sqrt(2π)) + γ - 0.5 (γ = Euler-Mascheroni)
  * Sample: norm of 3 × N(0,a²) via Box-Muller pairs
  * Bug note: variance formula: correct value is 3-8/π ≈ 0.45352091 NOT 0.45351549
  * Disk: found .zig-cache was 8.9GB (full disk); cleaned to free space
- **Tests**: 52 new tests all passing (4091/4098 total, 7 skipped)
- **Distribution count**: 36 total (21 continuous + 15 discrete)
- **Next Priority**: Lévy, LogLogistic, or Rice distribution

**Session 614 Update (2026-05-31) — FEATURE MODE:**

✅ **HalfNormal Distribution** — 35th distribution, 20th continuous — commits f7d4891, 332804c
- **Mode**: FEATURE MODE (counter: 614)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: HalfNormal(T) — |N(0,σ)|; bounded below at 0
  * Parameter: sigma (σ > 0); support [0, ∞)
  * PDF: √(2/π)/σ · exp(-x²/(2σ²))
  * CDF: erf(x/(σ·√2)); uses existing erf() helper
  * Quantile: σ·√2·erfInv(p); uses existing erfInv() helper
  * Mean: σ·√(2/π); Variance: σ²·(1-2/π)
  * Mode: 0 (always); Median: σ·√2·erfInv(0.5) ≈ 0.6745σ
  * Entropy: 0.5·ln(π/2) + ln(σ) + 0.5
  * Sample: |N(0,σ)| via Box-Muller transform
  * Test fix: tolerance adjusted 1e-14→1e-6/1e-7 for erf/erfInv (~3e-9 approx error)
  * Test fix: wrong expected for pdf(1, sigma=2) corrected (was 0.35196, should be 0.35207)
- **Tests**: 68 tests all passing (exit code 0)
- **Distribution count**: 35 total (20 continuous + 15 discrete)
- **Next Priority**: Maxwell-Boltzmann, Lévy, or LogLogistic distribution

**Session 613 Update (2026-05-31) — FEATURE MODE:**

✅ **Kumaraswamy Distribution** — 34th distribution, 19th continuous — commit 791e901
- **Mode**: FEATURE MODE (counter: 613)
- **CI Status**: push triggered new run
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Kumaraswamy(T) — bounded continuous on (0,1) with closed-form CDF/quantile
  * Parameters: a > 0, b > 0 (shape); flexible alternative to Beta distribution
  * PDF: a·b·x^(a-1)·(1-x^a)^(b-1)
  * CDF: 1-(1-x^a)^b (closed form, unlike Beta)
  * Quantile: (1-(1-p)^(1/b))^(1/a) (closed form, unlike Beta)
  * Mean: b·B(1+1/a, b) via lgamma
  * Variance: b·B(1+2/a, b) - mean²; exact for a=2,b=3: 201/4900
  * Mode: ((a-1)/(ab-1))^(1/a) for a>1 AND b>1; NaN otherwise
  * Entropy: (1-1/b) - ln(ab) + (1-1/a)·(γ + ψ(b+1)) using existing digamma
  * Sample: inverse transform X = (1-(1-U)^(1/b))^(1/a)
- **Tests**: 60 tests (all passing, exit code 0)
- **Distribution count**: 34 total (19 continuous + 15 discrete)
- **Next Priority**: HalfNormal, Maxwell-Boltzmann, or Lévy distribution

**Session 612 Update (2026-05-31) — FEATURE MODE:**

✅ **Rayleigh Distribution** — 33rd distribution, 18th continuous — commit 96f8bd0
- **Mode**: FEATURE MODE (counter: 612)
- **CI Status**: commit pushed, build pending
- **Open Issues**: 0 bugs, 0 feature requests
- **Implementation**: Rayleigh(T) — scale parameter σ>0; amplitude of 2D i.i.d. N(0,σ²) vectors
  * Special case of Weibull(k=2, λ=σ·√2)
  * Methods: init, pdf, logpdf, cdf, sf, quantile, mean, variance, mode, median, entropy, sample, validate
  * Sampling: inverse transform X = σ·sqrt(-2·ln(U))
  * entropy: 1 + γ/2 + ln(σ/√2), γ = Euler-Mascheroni constant
- **Tests**: 56 new Rayleigh tests (all passing, exit code 0)
- **Distribution count**: 33 total (18 continuous + 15 discrete)
- **Next Priority**: Kumaraswamy, HalfNormal, or Maxwell-Boltzmann

**Session 611 Update (2026-05-31) — FEATURE MODE:**

✅ **Von Mises Distribution** — 32nd distribution, 17th continuous — commit ef92e63
- **Distribution count**: 32 total (17 continuous + 15 discrete)

**Session 610 Update (2026-05-30) — STABILIZATION MODE:**

✅ **ALL SYSTEMS GREEN** — commit 280ddb2
- **Mode**: STABILIZATION MODE (counter: 610)
- **CI Status**: ✅ GREEN — latest run SUCCESS, 0 open issues
- **Cross-Compilation**: ✅ All 6 targets pass (x86_64/aarch64 linux/macos + x86_64-windows + wasm32-wasi)
- **Test Quality Audit** (sessions 606-609 distributions: Logarithmic, Skellam, Rademacher, Triangular):
  * Logarithmic: 37 tests — EXCELLENT, no issues
  * Skellam: 39 tests — EXCELLENT, no issues
  * Rademacher: 39 tests — fixed vacuous init test (replaced `_ = dist` with pmf assertions)
  * Triangular: 52 tests — HIGH QUALITY, seed issue already resolved in implementation (seed=42, n=50000, tol=2%)
- **Fix**: Rademacher init test now asserts pmf(-1)=0.5, pmf(1)=0.5, pmf(0)=0.0
- **Distribution count**: 31 total (16 continuous + 15 discrete)
- **Next Priority**: VonMises, Rayleigh, or Kumaraswamy (FEATURE mode)

**Session 608 Update (2026-05-30) — FEATURE MODE:**

✅ **Rademacher Distribution** — 30th distribution, 15th discrete — commit 488ed3f
- **Mode**: FEATURE MODE (counter: 608)
- **CI Status**: ✅ GREEN — 3 recent runs SUCCESS, 0 open issues
- **Tests**: ✅ All 40 Rademacher tests passing (3768 total, exit code 0)
- **Deliverable**: Rademacher(T) added to src/stats/distributions.zig (+438 lines total)
  * Stateless design — no parameters, no allocator
  * Support: {-1, +1} only
  * Methods: init O(1), pmf O(1), logpmf O(1), cdf O(1), sf O(1), quantile O(1),
             mean O(1), variance O(1), entropy O(1), mode O(1), sample O(1), validate O(1)
  * PMF: 0.5 for k ∈ {-1,+1}, else 0.0
  * LogPMF: -ln(2) for k ∈ {-1,+1}, else -inf
  * CDF: 0.0 (k<-1), 0.5 (-1≤k<1), 1.0 (k≥1)
  * Mean: 0.0 (exact), Variance: 1.0 (exact), Entropy: ln(2) (exact)
  * Mode: -1 by convention (bimodal)
  * Sample: rng.boolean() → ±1
  * Quantile: -1 for (0,0.5], +1 for (0.5,1], error for p≤0 or p>1
  * Error handling: quantile p≤0 or p>1 → error.InvalidParameter
  * 40 tests: init, pmf/logpmf concrete values, cdf/sf/quantile, mean/variance/entropy/mode,
              sampling binary output, empirical 50/50 frequency, f32 support, validate
- **Distribution count**: 30 total (15 continuous + 15 discrete)
  * Continuous: Normal, Uniform, Exponential, Laplace, Weibull, Pareto, LogNormal, Cauchy, Gumbel, Gamma, Beta, ChiSquared, StudentT, F, Dirichlet
  * Discrete: Poisson, Binomial, Bernoulli, Geometric, NegativeBinomial, Hypergeometric, Categorical, Multinomial, Zipf, BetaBinomial, DirichletMultinomial, DiscreteUniform, Logarithmic, Skellam, **Rademacher**
- **Next Priority**: PolyaUrn or Yule-Simon or Conway-Maxwell-Poisson

**Session 607 Update (2026-05-30) — FEATURE MODE:**

✅ **Skellam Distribution** — 29th distribution, 14th discrete — commit 686e5da
- **Mode**: FEATURE MODE (counter: 607)
- **CI Status**: ✅ GREEN — 3 recent runs SUCCESS, 0 open issues
- **Tests**: ✅ All 39 Skellam tests passing (3728 total, exit code 0)
- **Deliverable**: Skellam(T) added to src/stats/distributions.zig
  * Parameters: μ₁ > 0, μ₂ > 0; support all integers ℤ
  * PMF: exp(-(μ₁+μ₂)) × (μ₁/μ₂)^(k/2) × I_{|k|}(2√(μ₁μ₂))
  * Bessel function: series expansion I_n(x) = Σ(x/2)^(n+2m)/(m!×(m+n)!) with 500-iteration convergence
  * Mean = μ₁−μ₂, Variance = μ₁+μ₂ (exact, O(1))
  * Sample: Knuth Poisson difference (μ<30) or normal approximation (μ≥30)
  * CDF: cumulative PMF sum from (mean−5σ) to k
  * Quantile: binary search over ±5σ range
- **Distribution count**: 29 total (15 continuous + 14 discrete)
- **Next Priority**: Rademacher or PolyaUrn

**Session 606 Update (2026-05-30) — FEATURE MODE:**

✅ **Logarithmic Distribution** — 28th distribution, 13th discrete — commit ae5ef72
- **Mode**: FEATURE MODE (counter: 606)
- **CI Status**: ✅ GREEN — 3 recent runs SUCCESS, 0 open issues
- **Tests**: ✅ All 37 Logarithmic tests passing (3651 total, exit code 0)
- **Distribution count**: 28 total (15 continuous + 13 discrete)
- **Next Priority**: PolyaUrn or Skellam or Rademacher

**Session 605 Update (2026-05-30) — STABILIZATION MODE:**

✅ **ALL SYSTEMS GREEN** — commit 2293205
- **Mode**: STABILIZATION MODE (counter: 605)
- **CI Status**: ✅ GREEN — 5 recent runs all SUCCESS, 0 open issues
- **Cross-Compilation**: ✅ All 6 targets pass
- **Note**: "slices differ" in zig build test is INTENTIONAL — debug.zig self-tests verify expectSliceEqual error detection (exit code 0, not failures)
- **Distribution count**: 27 total (15 continuous + 12 discrete)
- **Next Priority**: Logarithmic distribution or PolyaUrn or Skellam (FEATURE mode)

**Session 604 Update (2026-05-29) — FEATURE MODE:**

✅ **DiscreteUniform Distribution** — 27th distribution, 12th discrete — commit c159d7a
- **Distribution count**: 27 total (15 continuous + 12 discrete)
