import numpy as np
import pytest

from spectratwin.randomness.seed import derive_subseed, new_generator


def test_derive_subseed_is_deterministic():
    a = derive_subseed(42, "scene", "0001")
    b = derive_subseed(42, "scene", "0001")
    assert a == b


def test_derive_subseed_distinguishes_labels():
    assert derive_subseed(42, "scene") != derive_subseed(42, "camera")


def test_derive_subseed_distinguishes_master_seed():
    assert derive_subseed(1, "scene") != derive_subseed(2, "scene")


def test_derive_subseed_rejects_negative_master_seed():
    with pytest.raises(ValueError):
        derive_subseed(-1, "scene")


def test_derive_subseed_requires_a_label():
    with pytest.raises(ValueError):
        derive_subseed(42)


def test_new_generator_reproduces_same_draws():
    gen_a = new_generator(7, "sample", "0001")
    gen_b = new_generator(7, "sample", "0001")
    assert np.array_equal(gen_a.random(5), gen_b.random(5))
