"""Tests for :mod:`madys.mcmc`.

The isochrone grids shipped with MADYS are too large/optional to rely
on in CI, so the tests construct a synthetic grid that mimics the
:class:`madys.IsochroneGrid` interface used by ``mcmc.py`` (namely the
``data``, ``masses``, ``ages``, ``filters`` and ``model_version``
attributes plus ``isinstance`` compatibility via direct patching).
"""

import os
import sys
import types

import numpy as np
import pytest


madys_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if madys_root not in sys.path:
    sys.path.insert(0, madys_root)


def _make_fake_grid_module():
    """Stub ``madys.madys`` so ``mcmc.py`` can import it without the heavy deps."""

    class IsochroneGrid:
        def __init__(self, model_version, filters, **kwargs):
            self.model_version = model_version
            self.filters = np.asarray(list(filters))
            mass_range = kwargs.get('mass_range', [0.05, 2.0])
            age_range = kwargs.get('age_range', [1.0, 1000.0])
            n_mass, n_age = kwargs.get('n_steps', [50, 50])
            self.masses = np.geomspace(mass_range[0], mass_range[1], n_mass)
            self.ages = np.geomspace(age_range[0], age_range[1], n_age)
            lm = np.log10(self.masses)[:, None]
            la = np.log10(self.ages)[None, :]
            mags = 5.0 - 5.0 * lm + 0.3 * la
            self.data = np.stack([mags + 0.05 * i for i in range(len(self.filters))],
                                 axis=-1)
            self.model_version_info = kwargs

    class SampleObject:
        pass

    mod = types.ModuleType('madys.madys')
    mod.IsochroneGrid = IsochroneGrid
    mod.SampleObject = SampleObject
    pkg = types.ModuleType('madys')
    pkg.__path__ = []
    pkg.IsochroneGrid = IsochroneGrid
    pkg.SampleObject = SampleObject
    return pkg, mod


@pytest.fixture(scope='module')
def mcmc_module(monkeypatch_module):
    pkg, sub = _make_fake_grid_module()
    monkeypatch_module.setitem(sys.modules, 'madys', pkg)
    monkeypatch_module.setitem(sys.modules, 'madys.madys', sub)
    if 'madys.mcmc' in sys.modules:
        del sys.modules['madys.mcmc']
    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'madys.mcmc',
        os.path.join(madys_root, 'madys', 'mcmc.py'),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['madys.mcmc'] = module
    return module


@pytest.fixture(scope='module')
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def interp(mcmc_module):
    pkg = sys.modules['madys']
    grid = pkg.IsochroneGrid('fake', ['G', 'Bp', 'Rp'],
                             mass_range=[0.1, 1.5],
                             age_range=[1.0, 1000.0],
                             n_steps=[40, 40])
    return mcmc_module.IsochroneInterpolator(grid)


class TestIsochroneInterpolator:
    def test_predict_shape_scalar(self, interp):
        pred = interp.predict(np.log10(10.0), np.log10(1.0))
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))

    def test_predict_matches_analytic(self, interp):
        log_age = np.log10(100.0)
        log_mass = np.log10(1.0)
        pred = interp.predict(log_age, log_mass)
        expected_base = 5.0 - 5.0 * log_mass + 0.3 * log_age
        expected = expected_base + 0.05 * np.arange(3)
        np.testing.assert_allclose(pred, expected, atol=1e-3)

    def test_predict_array(self, interp):
        la = np.array([np.log10(10.0), np.log10(100.0)])
        lm = np.array([np.log10(0.5), np.log10(1.0)])
        pred = interp.predict(la, lm)
        assert pred.shape == (2, 3)

    def test_in_bounds(self, interp):
        la_mid = 0.5 * (interp.log_age_bounds[0] + interp.log_age_bounds[1])
        lm_mid = 0.5 * (interp.log_mass_bounds[0] + interp.log_mass_bounds[1])
        assert interp.in_bounds(la_mid, lm_mid)
        assert not interp.in_bounds(interp.log_age_bounds[1] + 1.0, lm_mid)
        assert not interp.in_bounds(la_mid, interp.log_mass_bounds[0] - 1.0)

    def test_filter_subset(self, mcmc_module):
        pkg = sys.modules['madys']
        grid = pkg.IsochroneGrid('fake', ['G', 'Bp', 'Rp'],
                                 mass_range=[0.1, 1.5],
                                 age_range=[1.0, 1000.0],
                                 n_steps=[20, 20])
        interp = mcmc_module.IsochroneInterpolator(grid, filters=['Bp'])
        pred = interp.predict(np.log10(100.0), np.log10(1.0))
        assert pred.shape == (1,)

    def test_unknown_filter_raises(self, mcmc_module):
        pkg = sys.modules['madys']
        grid = pkg.IsochroneGrid('fake', ['G'],
                                 mass_range=[0.1, 1.5],
                                 age_range=[1.0, 1000.0],
                                 n_steps=[20, 20])
        with pytest.raises(ValueError, match="Filter 'J' not in"):
            mcmc_module.IsochroneInterpolator(grid, filters=['J'])


class TestLogProbability:
    def test_prior_inside_is_zero(self, mcmc_module, interp):
        la_mid = 0.5 * (interp.log_age_bounds[0] + interp.log_age_bounds[1])
        lm_mid = 0.5 * (interp.log_mass_bounds[0] + interp.log_mass_bounds[1])
        assert mcmc_module.log_prior([la_mid, lm_mid], interp) == 0.0

    def test_prior_outside_is_neg_inf(self, mcmc_module, interp):
        la_lo = interp.log_age_bounds[0] - 1.0
        lm_mid = 0.5 * (interp.log_mass_bounds[0] + interp.log_mass_bounds[1])
        assert mcmc_module.log_prior([la_lo, lm_mid], interp) == -np.inf

    def test_prior_honors_extra_restrictions(self, mcmc_module, interp):
        la_mid = 0.5 * (interp.log_age_bounds[0] + interp.log_age_bounds[1])
        lm_mid = 0.5 * (interp.log_mass_bounds[0] + interp.log_mass_bounds[1])
        value = mcmc_module.log_prior(
            [la_mid, lm_mid], interp,
            log_age_range=(la_mid + 0.1, la_mid + 0.2),
        )
        assert value == -np.inf

    def test_likelihood_peaks_at_truth(self, mcmc_module, interp):
        la_true, lm_true = np.log10(50.0), np.log10(0.8)
        model = interp.predict(la_true, lm_true)
        err = 0.02 * np.ones_like(model)

        ll_true = mcmc_module.log_likelihood([la_true, lm_true], interp, model, err)
        ll_off = mcmc_module.log_likelihood([la_true + 0.2, lm_true + 0.2],
                                            interp, model, err)
        assert np.isfinite(ll_true)
        assert ll_true > ll_off

    def test_likelihood_handles_nan_photometry(self, mcmc_module, interp):
        la, lm = np.log10(20.0), np.log10(0.5)
        model = interp.predict(la, lm)
        model_with_nan = model.copy()
        model_with_nan[1] = np.nan
        err = 0.05 * np.ones_like(model)
        value = mcmc_module.log_likelihood([la, lm], interp, model_with_nan, err)
        assert np.isfinite(value)


class TestRunMcmc:
    def test_end_to_end_recovers_injected_values(self, mcmc_module, interp):
        la_true, lm_true = np.log10(40.0), np.log10(0.6)
        truth_mags = interp.predict(la_true, lm_true)
        phot_err = 0.02 * np.ones_like(truth_mags)
        rng = np.random.default_rng(42)
        phot = truth_mags + phot_err * rng.standard_normal(truth_mags.shape)

        result = mcmc_module.run_mcmc(
            phot, phot_err, ['G', 'Bp', 'Rp'], 'fake',
            mass_range=(0.1, 1.5), age_range=(1.0, 1000.0),
            n_walkers=16, n_steps=600, burn_in=200,
            seed=7, interpolator=interp,
        )

        samples = result['samples']
        assert samples.ndim == 2 and samples.shape[1] == 2
        assert samples.shape[0] >= 16 * 400

        med_la, med_lm = np.median(samples, axis=0)
        assert abs(med_la - la_true) < 0.2
        assert abs(med_lm - lm_true) < 0.1

        summary = mcmc_module.mcmc_summary(result)
        assert 'log10_age_Myr' in summary
        assert summary['age_Myr_median'] == pytest.approx(
            10 ** summary['log10_age_Myr']['median']
        )

    def test_mismatched_lengths_raise(self, mcmc_module, interp):
        with pytest.raises(ValueError):
            mcmc_module.run_mcmc(
                [10.0, 11.0], [0.05], ['G', 'Bp'], 'fake',
                interpolator=interp, n_walkers=8, n_steps=10, burn_in=0,
            )
