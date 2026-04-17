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
