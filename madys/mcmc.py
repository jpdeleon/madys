"""
MADYS MCMC sampling.

Probabilistic isochrone fitting built on top of ``madys.IsochroneGrid``.
Uses emcee (Goodman & Weare 2010) to sample the posterior distribution
of ``(log10(age/Myr), log10(mass/Msun))`` given a set of absolute
photometric magnitudes with Gaussian errors.

Public API
----------
IsochroneInterpolator : fast 2D (log_age, log_mass) -> magnitude predictor.
log_prior, log_likelihood, log_probability : modular building blocks.
run_mcmc : convenience wrapper that drives an ``emcee.EnsembleSampler``.
mcmc_summary : posterior summary (median + 16th/84th percentiles).
plot_corner : helper for a corner plot of the posterior samples.
compare_models : fit the same star with two models and stack the corners.
cluster_log_likelihood / cluster_log_probability / run_cluster_mcmc :
    IMF-marginalized CMD fitter for a whole star cluster, sampling
    (log10 age, distance modulus, E(B-V)).
"""

import warnings
import numpy as np
from scipy.interpolate import RectBivariateSpline

try:
    from madys.madys import IsochroneGrid, SampleObject
except ImportError:  # pragma: no cover - allow flat-layout import during tests
    from madys import IsochroneGrid, SampleObject  # type: ignore

try:
    import emcee
except ImportError:  # pragma: no cover
    emcee = None


_LN2PI = float(np.log(2.0 * np.pi))


class IsochroneInterpolator:
    """Fast (log10 age, log10 mass) -> absolute-magnitude predictor.

    Wraps an :class:`IsochroneGrid` instance and fits a bivariate spline
    per filter so that the model magnitudes can be evaluated cheaply
    inside an MCMC likelihood.
    """

    def __init__(self, grid, filters=None):
        if not isinstance(grid, IsochroneGrid):
            raise TypeError("grid must be an IsochroneGrid instance.")
        self.grid = grid
        self.masses = np.asarray(grid.masses, dtype=float)
        self.ages = np.asarray(grid.ages, dtype=float)
        self.log_masses = np.log10(self.masses)
        self.log_ages = np.log10(self.ages)

        data = np.asarray(grid.data, dtype=float)
        if data.ndim != 3:
            raise ValueError("grid.data must be a 3D array [n_mass, n_age, n_filters].")

        all_filters = np.asarray(grid.filters)
        if filters is None:
            filters = all_filters
        self.filters = np.asarray(filters)

        idx = []
        for f in self.filters:
            w = np.where(all_filters == f)[0]
            if len(w) == 0:
                raise ValueError(
                    f"Filter '{f}' not in grid.filters {list(all_filters)}."
                )
            idx.append(int(w[0]))
        self._filter_index = np.array(idx, dtype=int)

        self._splines = []
        self._has_data = np.zeros(len(self.filters), dtype=bool)
        for k, j in enumerate(self._filter_index):
            mag_grid = data[:, :, j]
            mask = np.isfinite(mag_grid)
            if mask.sum() < 4 or len(self.log_ages) < 2 or len(self.log_masses) < 2:
                self._splines.append(None)
                continue
            safe = np.where(mask, mag_grid, 0.0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    spline = RectBivariateSpline(
                        self.log_masses, self.log_ages, safe, kx=1, ky=1
                    )
                except Exception:
                    spline = None
            self._splines.append(spline)
            self._has_data[k] = spline is not None

        self.log_age_bounds = (float(self.log_ages.min()), float(self.log_ages.max()))
        self.log_mass_bounds = (float(self.log_masses.min()), float(self.log_masses.max()))

    def __repr__(self):
        return (
            f"IsochroneInterpolator(model='{self.grid.model_version}', "
            f"filters={list(self.filters)}, "
            f"log_age={self.log_age_bounds}, log_mass={self.log_mass_bounds})"
        )

    def predict(self, log_age, log_mass):
        """Return predicted absolute magnitudes at (log_age, log_mass).

        Scalar inputs return a 1D array of shape ``(n_filters,)``; array
        inputs broadcast and produce shape ``(..., n_filters)``.
        """
        log_age = np.atleast_1d(log_age).astype(float)
        log_mass = np.atleast_1d(log_mass).astype(float)
        if log_age.shape != log_mass.shape:
            raise ValueError("log_age and log_mass must have the same shape.")
        flat_age = log_age.ravel()
        flat_mass = log_mass.ravel()
        out = np.full((flat_age.size, len(self.filters)), np.nan)
        for k, spline in enumerate(self._splines):
            if spline is None:
                continue
            out[:, k] = spline.ev(flat_mass, flat_age)
        out = out.reshape(log_age.shape + (len(self.filters),))
        if out.shape[:-1] == (1,):
            return out[0]
        return out

    def in_bounds(self, log_age, log_mass):
        la_lo, la_hi = self.log_age_bounds
        lm_lo, lm_hi = self.log_mass_bounds
        return (la_lo <= log_age <= la_hi) and (lm_lo <= log_mass <= lm_hi)


def log_prior(theta, interpolator, log_age_range=None, log_mass_range=None):
    """Uniform-in-log prior restricted to the grid dynamical range."""
    log_age, log_mass = float(theta[0]), float(theta[1])
    la_lo, la_hi = interpolator.log_age_bounds
    lm_lo, lm_hi = interpolator.log_mass_bounds
    if log_age_range is not None:
        la_lo = max(la_lo, float(log_age_range[0]))
        la_hi = min(la_hi, float(log_age_range[1]))
    if log_mass_range is not None:
        lm_lo = max(lm_lo, float(log_mass_range[0]))
        lm_hi = min(lm_hi, float(log_mass_range[1]))
    if not (la_lo <= log_age <= la_hi):
        return -np.inf
    if not (lm_lo <= log_mass <= lm_hi):
        return -np.inf
    return 0.0


def log_likelihood(theta, interpolator, phot, phot_err):
    """Gaussian log-likelihood ignoring filters with NaN photometry or model."""
    log_age, log_mass = float(theta[0]), float(theta[1])
    model = interpolator.predict(log_age, log_mass)
    phot = np.asarray(phot, dtype=float)
    phot_err = np.asarray(phot_err, dtype=float)
    good = np.isfinite(model) & np.isfinite(phot) & np.isfinite(phot_err) & (phot_err > 0)
    if not np.any(good):
        return -np.inf
    resid = (phot[good] - model[good]) / phot_err[good]
    return -0.5 * float(np.sum(resid ** 2 + _LN2PI + 2.0 * np.log(phot_err[good])))


def log_probability(theta, interpolator, phot, phot_err,
                    log_age_range=None, log_mass_range=None):
    lp = log_prior(theta, interpolator,
                   log_age_range=log_age_range,
                   log_mass_range=log_mass_range)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, interpolator, phot, phot_err)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def _build_interpolator(model_version, filters, mass_range=None, age_range=None,
                        n_steps=(200, 200), **grid_kwargs):
    kwargs = dict(grid_kwargs)
    if mass_range is not None:
        kwargs['mass_range'] = list(mass_range)
    if age_range is not None:
        kwargs['age_range'] = list(age_range)
    kwargs.setdefault('n_steps', list(n_steps))
    grid = IsochroneGrid(model_version, list(filters), **kwargs)
    return IsochroneInterpolator(grid, filters=filters)


def run_mcmc(phot, phot_err, filters, model_version,
             mass_range=(0.05, 2.0), age_range=(1.0, 1000.0),
             n_walkers=32, n_steps=2000, burn_in=500,
             initial=None, initial_scatter=0.05,
             progress=False, seed=None, interpolator=None,
             n_grid_steps=(200, 200), **grid_kwargs):
    """Run an emcee MCMC fit for a single target's absolute photometry.

    Parameters
    ----------
    phot, phot_err : array-like
        Absolute magnitudes and 1-sigma errors, one entry per filter.
    filters : sequence of str
        Photometric filters corresponding to ``phot``.
    model_version : str
        Any model accepted by :class:`IsochroneGrid` (e.g. ``'mist'``,
        ``'parsec2'``).
    mass_range, age_range : tuple of float
        Linear bounds for the isochrone grid AND the uniform prior.
    n_walkers, n_steps, burn_in : int
        Sampler configuration.
    initial : length-2 array or None
        Initial guess ``(log10(age), log10(mass))``. If ``None``,
        the midpoint of the grid is used.
    initial_scatter : float
        Scatter (same units as theta, i.e. dex) applied around ``initial``.
    interpolator : IsochroneInterpolator or None
        Optional pre-built interpolator. When supplied, the grid kwargs
        are ignored.
    **grid_kwargs :
        Additional kwargs forwarded to :class:`IsochroneGrid`
        (``feh``, ``v_vcrit``, ``fspot``, ``B``, ``he``, ``afe``, ...).

    Returns
    -------
    result : dict
        Keys ``sampler`` (the ``emcee.EnsembleSampler``), ``samples``
        (post-burn flat chain, shape ``[n_samples, 2]``),
        ``interpolator``, ``filters``, ``model_version``.
    """
    if emcee is None:
        raise ImportError("emcee is required for run_mcmc; pip install emcee.")

    phot = np.asarray(phot, dtype=float)
    phot_err = np.asarray(phot_err, dtype=float)
    filters = list(filters)
    if phot.shape != phot_err.shape:
        raise ValueError("phot and phot_err must have the same shape.")
    if phot.shape[-1] != len(filters):
        raise ValueError("length of phot must match len(filters).")

    if interpolator is None:
        interpolator = _build_interpolator(
            model_version, filters,
            mass_range=mass_range, age_range=age_range,
            n_steps=n_grid_steps, **grid_kwargs
        )

    log_age_range = (float(np.log10(age_range[0])), float(np.log10(age_range[1])))
    log_mass_range = (float(np.log10(mass_range[0])), float(np.log10(mass_range[1])))

    if initial is None:
        initial = np.array([
            0.5 * (log_age_range[0] + log_age_range[1]),
            0.5 * (log_mass_range[0] + log_mass_range[1]),
        ])
    initial = np.asarray(initial, dtype=float)

    rng = np.random.default_rng(seed)
    p0 = initial + initial_scatter * rng.standard_normal((n_walkers, 2))

    for i in range(n_walkers):
        p0[i, 0] = np.clip(p0[i, 0], *log_age_range)
        p0[i, 1] = np.clip(p0[i, 1], *log_mass_range)

    sampler = emcee.EnsembleSampler(
        n_walkers, 2, log_probability,
        args=(interpolator, phot, phot_err),
        kwargs=dict(log_age_range=log_age_range,
                    log_mass_range=log_mass_range),
    )
    sampler.run_mcmc(p0, n_steps, progress=progress)

    discard = min(burn_in, max(0, n_steps - 1))
    samples = sampler.get_chain(discard=discard, flat=True)

    return {
        'sampler': sampler,
        'samples': samples,
        'interpolator': interpolator,
        'filters': filters,
        'model_version': model_version,
        'log_age_range': log_age_range,
        'log_mass_range': log_mass_range,
    }


def mcmc_summary(result, param_labels=('log10_age_Myr', 'log10_mass_Msun')):
    """Return posterior medians with +/- 68% credible intervals."""
    samples = result['samples'] if isinstance(result, dict) else np.asarray(result)
    out = {}
    for i, label in enumerate(param_labels):
        chain = samples[:, i]
        q16, q50, q84 = np.percentile(chain, [16, 50, 84])
        out[label] = {
            'median': float(q50),
            'lower_1sigma': float(q50 - q16),
            'upper_1sigma': float(q84 - q50),
        }
    out['age_Myr_median'] = float(10 ** out[param_labels[0]]['median'])
    out['mass_Msun_median'] = float(10 ** out[param_labels[1]]['median'])
    return out


def plot_corner(result, labels=None, truths=None, **kwargs):
    """Thin wrapper around :mod:`corner.corner`."""
    try:
        import corner
    except ImportError as exc:  # pragma: no cover
        raise ImportError("corner is required for plot_corner; pip install corner.") from exc

    samples = result['samples'] if isinstance(result, dict) else np.asarray(result)
    if labels is None:
        labels = [r"$\log_{10}(\mathrm{age}/\mathrm{Myr})$",
                  r"$\log_{10}(M/M_\odot)$"]
    return corner.corner(samples, labels=labels, truths=truths, **kwargs)


def compare_models(phot, phot_err, filters, model_versions,
                   mass_range=(0.05, 2.0), age_range=(1.0, 1000.0),
                   **kwargs):
    """Run :func:`run_mcmc` against several models and collect summaries.

    Parameters
    ----------
    model_versions : sequence of str
        Model names (e.g. ``['mist', 'parsec2']``).

    Returns
    -------
    dict
        Maps each model name to the ``run_mcmc`` result enriched with a
        ``summary`` entry.
    """
    results = {}
    for model in model_versions:
        res = run_mcmc(phot, phot_err, filters, model,
                       mass_range=mass_range, age_range=age_range, **kwargs)
        res['summary'] = mcmc_summary(res)
        results[model] = res
    return results


def mcmc_from_sample(sample_obj, index, model_version, **kwargs):
    """Convenience wrapper: run MCMC on a specific row of a :class:`SampleObject`.

    Absolute photometry with ``phot_err > ph_cut`` is ignored.
    """
    if not isinstance(sample_obj, SampleObject):
        raise TypeError("sample_obj must be a madys.SampleObject instance.")
    if index < 0 or index >= len(sample_obj):
        raise IndexError(f"index {index} outside sample of size {len(sample_obj)}.")

    phot = np.asarray(sample_obj.abs_phot[index], dtype=float).copy()
    phot_err = np.asarray(sample_obj.abs_phot_err[index], dtype=float).copy()
    filters = list(sample_obj.filters)

    ph_cut = kwargs.pop('ph_cut', 0.2)
    bad = ~np.isfinite(phot) | ~np.isfinite(phot_err) | (phot_err > ph_cut) | (phot_err <= 0)
    phot[bad] = np.nan
    phot_err[bad] = np.nan

    return run_mcmc(phot, phot_err, filters, model_version, **kwargs)


# ---------------------------------------------------------------------------
# Cluster-CMD fitter
# ---------------------------------------------------------------------------


def _default_extinction_fn(ebv, filter_name):
    """Default ``A_filter`` in magnitudes using MADYS' stored coefficients."""
    return float(SampleObject.extinction(ebv, filter_name))


def _isochrone_cmd(interpolator, log_age, ebv, color_filters, mag_filter,
                   mass_grid=None, extinction_fn=None):
    """Return (mass, color, mag) for the cluster isochrone at a given age."""
    if extinction_fn is None:
        extinction_fn = _default_extinction_fn
    if mass_grid is None:
        mass_grid = interpolator.masses

    mass_grid = np.asarray(mass_grid, dtype=float)
    log_mass = np.log10(mass_grid)
    log_age_arr = np.full_like(log_mass, float(log_age))
    abs_mags = interpolator.predict(log_age_arr, log_mass)  # (n_mass, n_filters)

    filters = list(interpolator.filters)
    c1, c2 = color_filters
    f_mag = mag_filter
    try:
        c1_idx = filters.index(c1)
        c2_idx = filters.index(c2)
        m_idx = filters.index(f_mag)
    except ValueError as exc:
        raise ValueError(
            f"interpolator does not cover {color_filters!r} / {mag_filter!r}; "
            f"available filters: {filters}."
        ) from exc

    A_c1 = float(extinction_fn(ebv, c1))
    A_c2 = float(extinction_fn(ebv, c2))
    A_m = float(extinction_fn(ebv, f_mag))

    color_iso = (abs_mags[:, c1_idx] - abs_mags[:, c2_idx]) + (A_c1 - A_c2)
    mag_iso_abs = abs_mags[:, m_idx]  # before distance/reddening
    return mass_grid, color_iso, mag_iso_abs, A_m


def cluster_log_likelihood(theta, interpolator, color, mag, color_err, mag_err,
                           color_filters, mag_filter,
                           mass_grid=None, imf_alpha=2.35,
                           extinction_fn=None):
    """Gaussian likelihood marginalized over an IMF-weighted mass axis.

    ``theta`` is ``(log10_age_Myr, distance_modulus, E(B-V))``. For each
    observed star, the likelihood is the IMF-weighted integral of a 2D
    Gaussian in CMD space evaluated along the reddened, distance-shifted
    isochrone.
    """
    log_age, mu, ebv = (float(x) for x in theta)
    if ebv < 0:
        return -np.inf

    mass_grid, color_iso, mag_iso_abs, A_m = _isochrone_cmd(
        interpolator, log_age, ebv, color_filters, mag_filter,
        mass_grid=mass_grid, extinction_fn=extinction_fn,
    )
    mag_iso = mag_iso_abs + mu + A_m

    good = np.isfinite(color_iso) & np.isfinite(mag_iso)
    if good.sum() < 2:
        return -np.inf

    color_iso = color_iso[good]
    mag_iso = mag_iso[good]
    masses = mass_grid[good]

    dm = np.gradient(masses)
    dm = np.clip(dm, 0, None)
    w = masses ** (-float(imf_alpha)) * dm
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return -np.inf
    log_w = np.log(w / w_sum)

    color = np.asarray(color, dtype=float)
    mag = np.asarray(mag, dtype=float)
    color_err = np.asarray(color_err, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    good_obs = (
        np.isfinite(color) & np.isfinite(mag)
        & np.isfinite(color_err) & np.isfinite(mag_err)
        & (color_err > 0) & (mag_err > 0)
    )
    if not np.any(good_obs):
        return -np.inf

    color = color[good_obs]
    mag = mag[good_obs]
    color_err = color_err[good_obs]
    mag_err = mag_err[good_obs]

    dc = color[:, None] - color_iso[None, :]
    dm2 = mag[:, None] - mag_iso[None, :]
    chi2 = (dc / color_err[:, None]) ** 2 + (dm2 / mag_err[:, None]) ** 2
    log_norm = -_LN2PI - np.log(color_err) - np.log(mag_err)
    log_arg = log_norm[:, None] + log_w[None, :] - 0.5 * chi2

    max_log = np.max(log_arg, axis=1)
    bad = ~np.isfinite(max_log)
    if np.any(bad):
        return -np.inf
    log_lik_per_star = max_log + np.log(np.sum(np.exp(log_arg - max_log[:, None]), axis=1))
    total = float(np.sum(log_lik_per_star))
    if not np.isfinite(total):
        return -np.inf
    return total


def cluster_log_prior(theta, interpolator, mu_range, ebv_range,
                      log_age_range=None):
    log_age, mu, ebv = (float(x) for x in theta)
    la_lo, la_hi = interpolator.log_age_bounds
    if log_age_range is not None:
        la_lo = max(la_lo, float(log_age_range[0]))
        la_hi = min(la_hi, float(log_age_range[1]))
    if not (la_lo <= log_age <= la_hi):
        return -np.inf
    if not (float(mu_range[0]) <= mu <= float(mu_range[1])):
        return -np.inf
    if not (float(ebv_range[0]) <= ebv <= float(ebv_range[1])):
        return -np.inf
    if ebv < 0:
        return -np.inf
    return 0.0


def cluster_log_probability(theta, interpolator, color, mag, color_err, mag_err,
                            color_filters, mag_filter,
                            mu_range, ebv_range, log_age_range=None,
                            mass_grid=None, imf_alpha=2.35,
                            extinction_fn=None):
    lp = cluster_log_prior(theta, interpolator, mu_range, ebv_range,
                           log_age_range=log_age_range)
    if not np.isfinite(lp):
        return -np.inf
    ll = cluster_log_likelihood(
        theta, interpolator, color, mag, color_err, mag_err,
        color_filters, mag_filter,
        mass_grid=mass_grid, imf_alpha=imf_alpha,
        extinction_fn=extinction_fn,
    )
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_cluster_mcmc(color, mag, color_err, mag_err,
                     color_filters, mag_filter, model_version,
                     mu_range, ebv_range=(0.0, 1.0),
                     age_range=(1.0, 1e4), mass_range=(0.1, 10.0),
                     n_walkers=32, n_steps=2000, burn_in=500,
                     initial=None, initial_scatter=(0.1, 0.1, 0.02),
                     seed=None, interpolator=None,
                     imf_alpha=2.35, mass_grid=None,
                     extinction_fn=None,
                     progress=False,
                     n_grid_steps=(200, 200), **grid_kwargs):
    """Sample ``(log10 age, distance modulus, E(B-V))`` from a cluster CMD.

    Parameters
    ----------
    color, mag : array-like, shape (n_stars,)
        Observed cluster color and magnitude.
    color_err, mag_err : array-like, shape (n_stars,)
        Per-star 1-sigma errors in color and magnitude.
    color_filters : 2-tuple of str
        Filter names defining the color axis, e.g. ``('Gbp', 'Grp')``.
    mag_filter : str
        Filter name of the y-axis magnitude, e.g. ``'G'``.
    model_version : str
        Isochrone family: ``'mist'``, ``'parsec2'``, ...
    mu_range : tuple of float
        Uniform prior bounds on the distance modulus.
    ebv_range : tuple of float
        Uniform prior bounds on the colour excess.
    age_range, mass_range : tuple of float
        Grid bounds used both to construct the :class:`IsochroneGrid` and
        to clip the log-age prior.
    interpolator : IsochroneInterpolator or None
        Optional pre-built interpolator covering ``color_filters`` and
        ``mag_filter``.
    imf_alpha : float
        Power-law IMF exponent (Salpeter: 2.35).
    **grid_kwargs :
        Forwarded to :class:`IsochroneGrid` (``feh``, ``v_vcrit``, ...).

    Returns
    -------
    dict with keys ``sampler``, ``samples``, ``interpolator``,
    ``color_filters``, ``mag_filter``, ``model_version``,
    ``log_age_range``, ``mu_range``, ``ebv_range``.
    """
    if emcee is None:
        raise ImportError("emcee is required for run_cluster_mcmc; pip install emcee.")

    color = np.asarray(color, dtype=float)
    mag = np.asarray(mag, dtype=float)
    color_err = np.asarray(color_err, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    if not (color.shape == mag.shape == color_err.shape == mag_err.shape):
        raise ValueError("color, mag, color_err, mag_err must share shape (n_stars,).")
    if color.ndim != 1:
        raise ValueError("cluster CMD inputs must be 1D arrays.")

    all_filters = [color_filters[0], color_filters[1], mag_filter]
    if interpolator is None:
        interpolator = _build_interpolator(
            model_version, all_filters,
            mass_range=mass_range, age_range=age_range,
            n_steps=n_grid_steps, **grid_kwargs,
        )

    log_age_range = (float(np.log10(age_range[0])), float(np.log10(age_range[1])))
    mu_range = (float(mu_range[0]), float(mu_range[1]))
    ebv_range = (float(ebv_range[0]), float(ebv_range[1]))

    if initial is None:
        initial = np.array([
            0.5 * (log_age_range[0] + log_age_range[1]),
            0.5 * (mu_range[0] + mu_range[1]),
            0.5 * (ebv_range[0] + ebv_range[1]),
        ])
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (3,):
        raise ValueError("initial must have shape (3,) for (log_age, mu, ebv).")

    scatter = np.broadcast_to(np.asarray(initial_scatter, dtype=float), (3,))
    rng = np.random.default_rng(seed)
    p0 = initial + scatter * rng.standard_normal((n_walkers, 3))
    p0[:, 0] = np.clip(p0[:, 0], *log_age_range)
    p0[:, 1] = np.clip(p0[:, 1], *mu_range)
    p0[:, 2] = np.clip(p0[:, 2], *ebv_range)

    sampler = emcee.EnsembleSampler(
        n_walkers, 3, cluster_log_probability,
        args=(interpolator, color, mag, color_err, mag_err,
              color_filters, mag_filter,
              mu_range, ebv_range),
        kwargs=dict(log_age_range=log_age_range,
                    mass_grid=mass_grid, imf_alpha=imf_alpha,
                    extinction_fn=extinction_fn),
    )
    sampler.run_mcmc(p0, n_steps, progress=progress)
    discard = min(burn_in, max(0, n_steps - 1))
    samples = sampler.get_chain(discard=discard, flat=True)

    return {
        'sampler': sampler,
        'samples': samples,
        'interpolator': interpolator,
        'color_filters': color_filters,
        'mag_filter': mag_filter,
        'model_version': model_version,
        'log_age_range': log_age_range,
        'mu_range': mu_range,
        'ebv_range': ebv_range,
    }


def cluster_mcmc_summary(result):
    """Posterior medians and +/- 68% intervals for the cluster parameters."""
    samples = result['samples'] if isinstance(result, dict) else np.asarray(result)
    labels = ('log10_age_Myr', 'distance_modulus', 'ebv')
    out = {}
    for i, label in enumerate(labels):
        chain = samples[:, i]
        q16, q50, q84 = np.percentile(chain, [16, 50, 84])
        out[label] = {
            'median': float(q50),
            'lower_1sigma': float(q50 - q16),
            'upper_1sigma': float(q84 - q50),
        }
    out['age_Myr_median'] = float(10 ** out['log10_age_Myr']['median'])
    out['distance_pc_median'] = float(10 ** (1.0 + out['distance_modulus']['median'] / 5.0))
    return out
