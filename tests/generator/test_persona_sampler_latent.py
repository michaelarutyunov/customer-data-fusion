"""Tests for z-driven latent deviation in persona_sampler."""

from __future__ import annotations

import numpy as np

from generator.persona_sampler import project, sample_persona
from schemas.persona import PersonaConfig


def test_latent_not_none() -> None:
    config = sample_persona("price_lex", random_seed=1)
    assert isinstance(config, PersonaConfig)
    assert config.latent is not None


def test_project_bounded() -> None:
    assert 0.0 <= project(z_axis=-3.0, base=0.5) <= 1.0
    assert 0.0 <= project(z_axis=3.0, base=0.5) <= 1.0


def test_project_monotone() -> None:
    assert project(-2, 0.5) < project(0, 0.5) < project(2, 0.5)


def test_correlated_deviations() -> None:
    price = []
    brand = []
    for seed in range(200):
        config = sample_persona("price_lex", random_seed=seed)
        price.append(config.transactions.price_sensitivity)
        brand.append(config.transactions.brand_loyalty)

    price_arr = np.array(price)
    brand_arr = np.array(brand)

    # Driven by different z axes → not perfectly correlated.
    corr = np.corrcoef(price_arr, brand_arr)[0, 1]
    assert abs(corr) < 0.9

    # There IS per-participant variation.
    assert price_arr.std() > 0.05
