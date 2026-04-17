# Cluster CMD MCMC Fitting — Methods Comparison

This note summarizes the cluster-CMD isochrone fitter added in `madys/mcmc.py`
and contrasts it with two established population-level fitters:

- **brutus** (`brutus-package/src/brutus/analysis/populations.py`, `core/populations.py`)
- **isochrones** (`isochrones/isochrones/cluster.py`, `cluster_utils.py`, `priors.py`)

The three tools all target the same question — *"given a CMD of a supposedly
coeval stellar population, infer its age / distance / extinction / metallicity"* —
but they sit at very different points on the complexity spectrum.

---

## 1. What madys now does

**Entry point:** `madys.mcmc.run_cluster_mcmc`, backed by `cluster_log_likelihood`,
`cluster_log_prior`, `cluster_log_probability`.

**Sampled parameters (3):** `theta = (log10 age, mu, E(B-V))`

| Component | Implementation |
|-----------|----------------|
| Isochrone model | `IsochroneInterpolator` wraps `madys.IsochroneGrid` with per-filter `RectBivariateSpline(kx=1, ky=1)` in `(log_mass, log_age)` |
| Metallicity | Fixed at whatever grid is loaded (`MIST`, `PARSEC2`, …). Not sampled. |
| Binaries | Not modeled. Every observed point is assumed to be a single star. |
| Extinction | Per-filter `A_filter = SampleObject.extinction(E(B-V), filter)` (`3.16 * A_coeff * E(B-V)` with MADYS' stored coefficients; overridable via `extinction_fn`) |
| Distance | Apparent magnitude `m = M + mu + A_filter`; no parallax term |
| IMF | Salpeter power law (`alpha = 2.35` by default), uniform mass grid from the interpolator |
| Per-star likelihood | Gaussian in **magnitude** space for color and apparent magnitude, each with a separate 1D error |
| IMF marginalization | `logsumexp` over the mass axis: `log L_i = logsumexp(log_norm + log_w - 0.5 * chi2)` with `log_w ∝ -alpha * log_mass` |
| Priors | Uniform on all three parameters, bounds set by caller |
| Sampler | `emcee.EnsembleSampler` (affine-invariant), initial walker ball clipped to the prior box |
| Outputs | Chain + log-prob + medians / 68 % intervals via `cluster_mcmc_summary`; derived `age_Myr_median`, `distance_pc_median` |

### Mathematically
For N observed stars with color/magnitude `(c_i, m_i)` and uncertainties
`(σ_c,i, σ_m,i)`:

$$
\ln \mathcal{L}(\log t, \mu, E) = \sum_i \ln \int w(M)\,
\mathcal{N}\!\big(c_i \mid c_\mathrm{iso}(M;\log t, E)\big)\,
\mathcal{N}\!\big(m_i \mid M_\mathrm{iso}(M;\log t) + \mu + A_m(E)\big)\,\mathrm{d}M
$$

with $w(M) \propto M^{-\alpha}$, normalized per-star by $\sum_k w_k$.
In code this is a `logsumexp` over the discrete mass grid inside
`cluster_log_likelihood` (`madys/mcmc.py`).

### Pros
- **Zero external infrastructure**: runs with `madys + emcee + scipy` only.
- **Transparent, ~300 LoC** total for cluster code — easy to audit and extend.
- **Model-agnostic** at the grid level: MIST vs PARSEC2 comparison is a
  one-line change (demonstrated in `examples/MADYS_cluster_cmd_mcmc_tutorial.ipynb`).
- **Fast**: 3-parameter Gaussian likelihood with ~200-point mass grid, ~60 s
  per 32×1000 emcee run on a toy cluster.
- Interoperates directly with madys filter systems and extinction law.

### Cons / assumptions
- **Single metallicity.** You can't fit `[Fe/H]` with this fitter; swapping
  the grid changes it discretely.
- **No binaries.** Unresolved binaries brighten and redden stars in a
  correlated way; ignoring them biases age young and distance bright.
- **No outlier / membership model.** Field contaminants pull the posterior
  toward whatever isochrone best accommodates them. You must pre-clean the
  catalog.
- **No parallax / per-star distance scatter.** Every star shares one μ.
- **Uniform priors only.** No IMF slope fit, no dust-map prior, no
  Galactic-structure prior.
- **Gaussian in mag space**, not flux. Fine when `σ_m ≲ 0.1`, but underestimates
  the tail for faint/low-SNR stars (a well-known CMD-fitting caveat — see the
  brutus/isochrones implementations below, both of which avoid this).
- **Mass grid is log-uniform in the interpolator's mass array**, not EEP.
  This under-samples post-MS evolution where the isochrone folds back in
  mass; the toy tests had to add an explicit turnoff to keep the fit
  identifiable.

---

## 2. brutus — `analysis.populations.isochrone_population_loglike`

**Reference file:** `brutus-package/src/brutus/analysis/populations.py` (+ `core/populations.py`, `priors/`).

**Sampled parameters (6):** `theta = [Fe/H, log(age), A_V, R_V, distance, f_field]`

Designed to be called from an external sampler (emcee / dynesty). The central
design choice is the **"mixture-before-marginalization"** pipeline, in 5 steps:

1. **`generate_isochrone_population_grid`** — builds a 2D grid over
   `(EEP, SMF)` (SMF = secondary mass fraction for binaries, default 21 pts in
   `[0, 1]`; EEP default 1000 pts in `[202, 808]`). Each grid point has
   synthetic photometry, primary mass, and **geometric jacobians** `dm` and
   `dSMF`.
2. **`compute_isochrone_cluster_loglike`** — per (grid_point, star)
   log-likelihood using `brutus.utils.photometry.phot_loglike` in **flux
   space** (chi-square, or Gaussian with `dim_prior=False`), plus an
   **optional parallax term** (Gaussian on `1000/d`).
3. **`compute_isochrone_outlier_loglike`** — independent contaminant model
   (chi-square or uniform, stellar-parameter-aware, pluggable
   `outlier_model_func`).
4. **`apply_isochrone_mixture_model`** — at each grid point, mixes
   `w_c · P_c + w_o · P_o` with `w_c = P_cluster · (1 − f_field)`.
5. **`marginalize_isochrone_grid`** — `logsumexp` over `(EEP, SMF)` weighted
   by `dm · dSMF`, then summed over stars.

### Priors (in `brutus/priors/`)
- **`stellar.logp_imf`**: Kroupa-like broken power-law `(alpha_low=1.3,
  alpha_high=2.3, break=0.5 M☉)`, with binary term through `mgrid2`.
- **`galactic.logp_galactic_structure`**: exponential disk + flattened halo
  number density as a distance prior, Gaussian `p(Fe/H)` split disk/halo,
  age–metallicity relation `logp_age_from_feh`.
- **`extinction.logp_extinction`**: Gaussian pull toward a 3D dust map
  (Bayestar); uniform fallback outside coverage.
- **`astrometric.logp_parallax`**: Gaussian in parallax, with optional
  `s = 1/d²` scale-factor reparameterization.
- Luminosity function prior `logp_ps1_luminosity_function` available.

### Pros
- **Mathematically correct mixture treatment**: contamination weight is applied
  *before* integrating over stellar parameters. Traditional "fit, then mix"
  pipelines are biased when the outlier probability depends on mass/EEP.
- **Binaries as first-class citizens** through the SMF axis (adds fluxes, not
  magnitudes).
- **Flux-space photometric likelihood** (`phot_loglike`) — correct for
  low-SNR bands where magnitude Gaussians break.
- **Full astrophysical prior stack**: 3D dust map, Galactic structure,
  parallax, IMF, LF — all modular.
- Supports `R_V` variation (not just `A_V`), parallax per star, arbitrary
  outlier likelihood, and `return_components=True` for diagnostics.
- Grid jacobians (`dm`, `dSMF`) are explicit, not implicit — reduces
  quadrature bias.

### Cons
- **Heavy**: 900+ LoC just for `analysis.populations`, plus `StellarPop`,
  dust maps, NN bolometric corrections. Large install footprint.
- **MIST-centric**: `Isochrone` / `StellarPop` wrap MIST tracks and neural-net
  BCs. Swapping isochrone families requires rewrapping, not just a different
  `.h5`.
- **Computational cost**: default `(EEP × SMF)` grid is ~21 000 points × N
  stars × N filters per likelihood call. Fine for population work but slow
  for quick diagnostic fits.
- `field_fraction` is fit but `cluster_prob` (the prior membership) is a fixed
  hyperparameter — model can be sensitive to it.
- The five-function pipeline is stateless (good for MCMC) but awkward to
  interrogate interactively.

---

## 3. isochrones — `cluster.StarClusterModel`

**Reference files:** `isochrones/isochrones/cluster.py`,
`isochrones/isochrones/cluster_utils.py`, `isochrones/isochrones/priors.py`.

**Sampled parameters (7):** `[age, Fe/H, distance, A_V, alpha, gamma, f_B]`
where `alpha` is the IMF power-law slope, `gamma` the mass-ratio power-law
slope, and `f_B` the binary fraction.

### Likelihood structure
Per-star log-likelihood is a **double loop over EEPs** for a primary and
secondary (numba-parallelized in `calc_lnlike_grid`):

- Outer `j` = primary EEP, inner `k` ≤ `j` = secondary EEP (enforces
  `m2 ≤ m1`), with a mass-ratio floor `q_lo`.
- **Photometric likelihood per (j,k)** is the logaddexp of a single-star
  residual and a binary residual (primary+secondary fluxes added):
  `lnlike_phot = Σ_b logaddexp(ln f_B + lnL_binary, ln(1−f_B) + lnL_single)`.
  Gaussian in **mag** space but with explicit binary flux addition, not just
  a smeared error.
- Mass priors: `powerlaw_lnpdf(m_j; alpha, mass_lo, mass_hi)` plus
  `powerlaw_lnpdf(m_k/m_j; gamma, q_lo, 1)`.
- EEP→mass jacobian: `ln_dm_deeps = ln|dm/dEEP|`.
- `lnlike_prop` adds any non-photometric datum (e.g. parallax → Gaussian on
  `1000/d`).
- Outer marginalization: trapezoidal integral in k, then trapezoidal integral
  in j, for each star. Product across stars → total `lnlike`.

### Priors (`isochrones/isochrones/priors.py`)
Each parameter gets a class from a small zoo of `Prior` subclasses:

| Parameter | Default prior | Notes |
|-----------|---------------|-------|
| `age` (log₁₀ yr) | `FlatLogPrior((6, 10.15))` | flat in log-age |
| `Fe/H` | `FehPrior(halo_fraction)` | mixture of Casagrande-2011 disk (2-Gaussian) + halo Gaussian (μ=−1.5, σ=0.4) |
| `distance` | `PowerLawPrior(alpha=2.0, bounds=(0, max_distance))` | volumetric `d²` prior |
| `A_V` | `FlatPrior((0, max_AV))` | |
| `alpha` (IMF) | `FlatPrior((−4, −1))` | treated as a free nuisance parameter |
| `gamma` (q slope) | `GaussianPrior(0.3, 0.1)` | |
| `f_B` | `FlatPrior((0.0, 0.6))` | |
| EEP grid | `EEP_prior` | Jacobian-aware EEP prior available for per-star fits |

Other prior shapes in the module: `GaussianPrior`, `LogNormalPrior`,
`ChabrierPrior` (LogNormal + Salpeter broken prior), `QPrior`, `AVPrior`,
`DistancePrior`, `AgePrior`.

### Sampler
Defaults to **MultiNest** (`nlive`, MPI-friendly via `comm`) through
`mnest_prior`/`mnest_loglike`; `use_emcee=True` switches to emcee.

### Pros
- **IMF slope, mass-ratio distribution, binary fraction all fit simultaneously**
  with per-star binary photometry. State of the art for resolved-cluster
  analyses.
- **Physically-motivated, differentiated priors** on every parameter (halo
  fraction for metallicity, `d²` for distance, Gaussian for `γ`, etc.).
- **Nested sampling built in** → Bayesian evidence "for free", which matters
  for choosing between cluster models or including binaries.
- **numba-jit** inner loops → the double-EEP integral is tractable for
  thousands of stars.
- Clean reuse of the per-star `StarModel` machinery — you can run the same
  framework on 1 star or 10 000.

### Cons
- **Not pip-clean**: depends on MultiNest, numba, ChronoStar-style MIST
  wrappers, `StarCatalog` / `get_ichrone` infrastructure. Harder to install
  than madys or a plain emcee script.
- **MIST-dominated**: `get_ichrone("mist")` is the default; PARSEC/Dartmouth
  support exists but is less exercised.
- **7-parameter space** with strong degeneracies (α–f_B, age–A_V) needs
  `nlive ≳ 1000` and scales linearly with that — expensive.
- `calc_lnlike_grid` is `O(N_stars · N_EEP²)` per likelihood call — the numba
  jit saves it, but memory (`N_stars × N_EEP × N_EEP` lnlike grid) can bite on
  large catalogs.
- EEP-based parameterization requires tracks with an EEP mapping; not all
  stellar grids ship one.
- No first-class field-contamination mixture (user must pre-select
  members, or build one on top).

---

## 4. Side-by-side summary

| Feature | **madys.mcmc** | **brutus populations** | **isochrones cluster** |
|---|---|---|---|
| # sampled params | 3 | 6 | 7 |
| Params | log t, μ, E(B−V) | Fe/H, log t, A_V, R_V, d, f_field | log t, Fe/H, d, A_V, α_IMF, γ, f_B |
| Metallicity | fixed (grid choice) | fitted | fitted |
| Binaries | none | SMF axis, flux-added | EEP×EEP grid, flux-added |
| Outlier model | none | mixture before marginalization | none (pre-clean catalog) |
| Parallax | no | yes (per star) | yes (per star) |
| Extinction law | per-filter `madys.SampleObject.extinction` | variable `(A_V, R_V)` + Bayestar 3D | A_V flat |
| Photometric likelihood | Gaussian in **mag** | chi-square / Gaussian in **flux** | Gaussian in **mag**, binary fluxes added |
| Marginalization variable | mass (1D) | (EEP, SMF) grid with jacobians | (EEP1, EEP2) double trapz |
| IMF | Salpeter α=2.35 (fixed) | Kroupa broken (fixed) | power-law slope α is **fitted** |
| Priors | all uniform | IMF + Galactic + dust map + parallax | FlatLog/FehPrior (disk+halo)/PowerLaw/Gaussian |
| Sampler | emcee | emcee / dynesty (external) | MultiNest (default) / emcee |
| Isochrone families | MIST, PARSEC2, BHAC15, ATMO, ... (anything in madys) | MIST (+NN BCs) | MIST (primary), others via `get_ichrone` |
| LoC for cluster fitter | ~300 | ~900 + dependencies | ~600 + numba + priors module |
| Typical runtime | seconds–minutes | minutes | minutes–hours |
| Install footprint | `emcee`, `corner` | brutus + dust maps + NN weights + dynesty | MultiNest + numba + isochrones data |

---

## 5. When to pick which

- **madys.mcmc.run_cluster_mcmc** — fast, small-parameter sanity fits; when
  you want to compare isochrone *model families* (e.g. MIST vs PARSEC2 as
  in the tutorial) rather than pin down every astrophysical nuisance; when
  you have a cleaned, single-metallicity, mostly single-star sample and just
  want age, distance, reddening.

- **brutus populations** — production-quality population inference with
  field contamination, parallaxes, 3D dust-map priors, and unresolved
  binaries; ideal for deep Gaia/APOGEE-scale analyses where the outlier
  model matters.

- **isochrones cluster** — when the *binary population* and *IMF slope* are
  the science, not just a nuisance; when you want nested-sampling evidence
  to compare, e.g., "cluster with binaries" vs "cluster without"; when you
  already have well-vetted members and per-star parallaxes.

## 6. Natural extensions for madys.mcmc

Concrete ways to close the gap, in rough order of effort:

1. **Parallax / per-star distance scatter** — add a Gaussian term on
   `1000/d_i` next to the photometric likelihood; trivial given the existing
   `logsumexp` structure.
2. **Metallicity axis** — promote the interpolator to 3D `(log_mass, log_age,
   Fe/H)` and add `Fe/H` as a sampled parameter with either a flat or
   `FehPrior`-style mixture.
3. **Outlier fraction** — append a `f_field` parameter with a uniform-in-CMD
   outlier likelihood and apply `logaddexp` inside `cluster_log_likelihood`
   before summing stars (brutus-style mixture-before-marginalization).
4. **Binaries (SMF)** — add a 1D SMF grid and flux-add a secondary along the
   isochrone; the existing `logsumexp` pattern generalizes with one more axis.
5. **IMF slope α as a free parameter** — already a one-line change since
   `imf_alpha` is already threaded through `cluster_log_likelihood`.
6. **EEP-based mass grid** — switch from linear `mass_grid` to the
   IsochroneGrid's native EEP if/when madys exposes one, with explicit
   `dm/dEEP` jacobians as in isochrones.

Each of these can be added without rearchitecting the module: the current
`cluster_log_likelihood` is organized so that extra axes (SMF, Fe/H, outlier)
become extra `logsumexp` dimensions, and extra per-star data (parallax)
becomes an additive term.
