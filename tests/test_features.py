"""Tests for colour feature extraction (synthetic images, no dataset needed)."""

import numpy as np
import pytest

from avocado_ripeness.features import (FEATURE_NAMES, LIGHTING_INVARIANT_FEATURES,
                                       adjust_brightness, extract_features, fruit_mask)


def make_image(fruit_rgb, size=100, radius=30, bg=(235, 235, 240)):
    """White background with a filled circle of colour ``fruit_rgb``."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bg
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius ** 2
    img[inside] = fruit_rgb
    return img, inside


def test_mask_finds_fruit_and_ignores_background():
    img, inside = make_image((60, 140, 40))
    mask = fruit_mask(img)
    # at least 95 % agreement with the true circle
    assert np.mean(mask == inside) > 0.95


def test_feature_vector_shape_and_names():
    img, _ = make_image((60, 140, 40))
    f = extract_features(img)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(f))


def test_green_fruit_has_green_hue_and_high_green_fraction():
    img, _ = make_image((60, 140, 40))          # green
    f = dict(zip(FEATURE_NAMES, extract_features(img)))
    assert 80 < f["hue_mean_deg"] < 130
    assert f["frac_green"] > 0.9
    assert f["frac_purple"] < 0.05


def test_purple_fruit_has_purple_hue():
    img, _ = make_image((90, 40, 70))           # dark purple
    f = dict(zip(FEATURE_NAMES, extract_features(img)))
    assert f["hue_mean_deg"] > 270 or f["hue_mean_deg"] < 30
    assert f["frac_purple"] > 0.9


def test_circular_mean_handles_hue_wraparound():
    # half the fruit at hue ~350 deg, half at ~10 deg -> mean should be ~0, not ~180
    img, inside = make_image((150, 30, 50))     # red-purple, hue ~350
    left = inside & (np.arange(100)[None, :] < 50)
    img[left] = (150, 50, 30)                   # orange-red, hue ~8
    f = dict(zip(FEATURE_NAMES, extract_features(img)))
    h = f["hue_mean_deg"]
    assert min(h, 360 - h) < 20


def test_empty_frame_raises():
    blank = np.full((100, 100, 3), 240, np.uint8)
    with pytest.raises(ValueError, match="No fruit"):
        extract_features(blank)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        extract_features(np.zeros((50, 50), np.uint8))


def test_lighting_invariant_features_barely_change_with_brightness():
    img, _ = make_image((60, 120, 40), bg=(250, 250, 250))
    idx = [FEATURE_NAMES.index(n) for n in LIGHTING_INVARIANT_FEATURES]
    f1 = extract_features(img)[idx]
    f2 = extract_features(adjust_brightness(img, 0.8))[idx]
    hue_i = LIGHTING_INVARIANT_FEATURES.index("hue_mean_deg")
    assert abs(f1[hue_i] - f2[hue_i]) < 3          # hue within 3 degrees
    assert np.allclose(np.delete(f1, hue_i), np.delete(f2, hue_i), atol=0.05)
