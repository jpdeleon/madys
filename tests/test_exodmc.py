import pytest
import numpy as np
import copy


def iter_eccentric_anomaly(M, e, nIter=100):
    E = copy.deepcopy(M)
    for _ in range(nIter):
        delta = (M - (E - e*np.sin(E))) / (1 - e*np.cos(E))
        E += delta
    return E


def cropped_gaussian(mu, sigma, norb, limits=[0, 1]):
    if (mu < limits[0]) | (mu > limits[1]):
        raise ValueError(r'mu must be between {limits[0]} and {limits[1]}')
    if sigma > 5 * (limits[1] - limits[0]):
        return np.ones(norb)/(limits[1] - limits[0])
    
    Phi = lambda x, mu, sigma: 0.5 * (1 + erf((x-mu)/(np.sqrt(2) * sigma)))
    expected_invalid = Phi(limits[0], mu, sigma) + (limits[1] - Phi(1, mu, sigma))
    scaling_factor = int(2/(1 - expected_invalid))
    a1 = np.random.normal(mu, sigma, scaling_factor * norb)
    a1 = a1[(a1 > limits[0]) & (a1 < limits[1])][:norb]
    return a1


from scipy.special import erf


class TestIterEccentricAnomaly:
    def test_iter_eccentric_anomaly_circular_orbit(self):
        M = np.array([0.0, np.pi/4, np.pi/2, np.pi])
        e = 0.0
        E = iter_eccentric_anomaly(M, e)
        np.testing.assert_allclose(E, M, atol=1e-10)

    def test_iter_eccentric_anomaly_small_eccentricity(self):
        M = np.array([0.0, np.pi/2, np.pi])
        e = 0.1
        E = iter_eccentric_anomaly(M, e)
        assert np.all(E >= M - e)
        assert np.all(E <= M + e)

    def test_iter_eccentric_anomaly_single_value(self):
        M = 0.5
        e = 0.3
        E = iter_eccentric_anomaly(M, e)
        assert isinstance(E, (float, np.floating))

    def test_iter_eccentric_anomaly_high_eccentricity(self):
        M = np.array([0.0, 0.5, 1.0])
        e = 0.8
        E = iter_eccentric_anomaly(M, e)
        assert len(E) == len(M)


class TestCroppedGaussian:
    def test_cropped_gaussian_basic(self):
        mu = 0.5
        sigma = 0.1
        norb = 100
        result = cropped_gaussian(mu, sigma, norb)
        assert len(result) == norb
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_cropped_gaussian_mu_at_boundary(self):
        mu = 0.0
        sigma = 0.1
        norb = 50
        result = cropped_gaussian(mu, sigma, norb, limits=[0, 1])
        assert len(result) == norb
        assert np.all(result >= 0)

    def test_cropped_gaussian_large_sigma(self):
        mu = 0.5
        sigma = 100
        norb = 50
        result = cropped_gaussian(mu, sigma, norb, limits=[0, 1])
        assert len(result) == norb

    def test_cropped_gaussian_invalid_mu(self):
        mu = 1.5
        sigma = 0.1
        norb = 10
        with pytest.raises(ValueError):
            cropped_gaussian(mu, sigma, norb, limits=[0, 1])