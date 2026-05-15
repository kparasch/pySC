"""Tests for pySC.core.rng: reproducible random number generation."""
import numpy as np
import pytest

from pySC.core.rng import RNG


def test_rng_determinism():
    rng1 = RNG(seed=42)
    rng2 = RNG(seed=42)
    seq1 = [rng1.normal() for _ in range(50)]
    seq2 = [rng2.normal() for _ in range(50)]
    np.testing.assert_array_equal(seq1, seq2)


def test_rng_different_seeds():
    rng1 = RNG(seed=1)
    rng2 = RNG(seed=2)
    seq1 = [rng1.normal() for _ in range(50)]
    seq2 = [rng2.normal() for _ in range(50)]
    assert seq1 != seq2


def test_rng_state_serialization():
    rng = RNG(seed=42)
    # Advance the state
    _ = [rng.normal() for _ in range(10)]
    # Serialize and restore
    data = rng.model_dump()
    restored = RNG.model_validate(data)
    # Both should produce the same next values
    next_original = [rng.normal() for _ in range(20)]
    next_restored = [restored.normal() for _ in range(20)]
    np.testing.assert_array_equal(next_original, next_restored)


def test_normal_trunc_within_bounds():
    rng = RNG(seed=7)
    sigma = 2.0
    scale = 3.0
    loc = 10.0
    samples = rng.normal_trunc(loc=loc, scale=scale, sigma_truncate=sigma, size=5000)
    # All samples must be within loc +/- sigma_truncate * scale
    lower = loc - sigma * scale
    upper = loc + sigma * scale
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)


def test_normal_trunc_without_truncation():
    """With sigma_truncate=None and no default_truncation, normal_trunc behaves like normal."""
    rng_trunc = RNG(seed=99)
    rng_plain = RNG(seed=99)
    vals_trunc = [rng_trunc.normal_trunc(loc=0, scale=1, sigma_truncate=None) for _ in range(50)]
    vals_plain = [rng_plain.normal(loc=0, scale=1) for _ in range(50)]
    np.testing.assert_array_equal(vals_trunc, vals_plain)


def test_normal_trunc_with_size():
    rng = RNG(seed=42)
    result = rng.normal_trunc(size=100, sigma_truncate=3)
    assert isinstance(result, np.ndarray)
    assert len(result) == 100


def test_normal_distribution():
    rng = RNG(seed=0)
    samples = rng.normal(loc=5, scale=2, size=10000)
    assert abs(np.mean(samples) - 5) < 0.1
    assert abs(np.std(samples) - 2) < 0.1


def test_uniform_distribution():
    rng = RNG(seed=0)
    samples = rng.uniform(0, 1, size=10000)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)
    assert abs(np.mean(samples) - 0.5) < 0.05


def test_randomize_rng():
    rng = RNG(seed=42)
    seeded_vals = [rng.normal() for _ in range(20)]
    rng.randomize_rng()
    random_vals = [rng.normal() for _ in range(20)]
    # Extremely unlikely to match after re-seeding from OS entropy
    assert seeded_vals != random_vals
