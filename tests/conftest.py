import jax
import numpy as np
import pytest

from ripen.data import SyntheticConfig, dataset_info, simulate

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(scope="session")
def synthetic_items():
    return simulate(SyntheticConfig(n_items=36, t_max=20.0, seed=1))


@pytest.fixture(scope="session")
def synthetic_info(synthetic_items):
    return dataset_info(synthetic_items)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
