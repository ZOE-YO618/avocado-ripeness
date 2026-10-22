"""
Colour feature extraction for avocado photographs.

Pipeline for one image:

1. Load the RGB photo and (optionally) downscale it to ``size`` x ``size``
   pixels to cut computation time.
2. Convert RGB -> HSV (hue, saturation, value/brightness).
3. Segment the fruit from the plain white studio background with a simple
   threshold mask: background pixels are bright AND unsaturated, so a pixel
   belongs to the fruit if it is saturated OR dark.
4. Summarise the colour of the fruit pixels as a fixed-length feature vector.

Hue is an *angle* (0-360 deg), so it cannot be averaged directly: the mean of
350 deg and 10 deg should be 0 deg, not 180 deg. We therefore use the circular
mean, computed from the mean of cos(h) and sin(h).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

#: Names of the features returned by :func:`extract_features`, in order.
FEATURE_NAMES = [
    "hue_cos",        # mean cos(hue)  -> together with hue_sin encodes mean hue angle
    "hue_sin",        # mean sin(hue)
    "hue_mean_deg",   # circular mean hue in degrees [0, 360)
    "sat_mean",       # mean saturation, 0-1
    "val_mean",       # mean brightness (HSV value), 0-1
    "sat_std",        # spread of saturation over the fruit surface
    "val_std",        # spread of brightness over the fruit surface
    "frac_green",     # fraction of fruit pixels with hue in the green band
    "frac_purple",    # fraction of fruit pixels with hue in the red/purple band
    "frac_dark",      # fraction of fruit pixels that are very dark (V < 0.25)
]

#: Subset of features that do not depend on overall image brightness.
#: Scaling all RGB values by a constant leaves hue and saturation unchanged
#: (apart from clipping), so these features are robust to lighting level.
LIGHTING_INVARIANT_FEATURES = [
    "hue_cos", "hue_sin", "hue_mean_deg", "sat_mean", "sat_std",
    "frac_green", "frac_purple",
]

# Thresholds (OpenCV 8-bit HSV scale: H 0-179, S and V 0-255)
_BG_SAT_MAX = 40      # background is unsaturated ...
_BG_VAL_MIN = 140     # ... and bright
_MIN_FRUIT_FRACTION = 0.02  # less than 2 % fruit pixels -> segmentation failed


def load_image(path: str | Path, size: int | None = 200) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array, optionally resized.

    Parameters
    ----------
    path : str or Path
        Path to a JPEG/PNG file.
    size : int or None
        If given, the image is resized to ``size`` x ``size`` pixels
        (INTER_AREA, which averages pixels and avoids aliasing).
        ``None`` keeps the original resolution (800 x 800 in this dataset).

    Returns
    -------
    numpy.ndarray
        Array of shape (H, W, 3), dtype uint8, RGB channel order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be decoded as an image.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if size is not None:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fruit_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True on avocado pixels.

    A pixel is treated as background if it is both bright and unsaturated
    (the white lightbox). The mask is cleaned with a morphological opening and
    only the largest connected region is kept, which removes small specks and
    shadows at the image border.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB uint8 image of shape (H, W, 3).

    Returns
    -------
    numpy.ndarray
        Boolean array of shape (H, W).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    background = (s <= _BG_SAT_MAX) & (v >= _BG_VAL_MIN)
    mask = (~background).astype(np.uint8)

    # Remove isolated noise pixels
    kernel_size = max(3, rgb.shape[0] // 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component (the fruit)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Compute the colour feature vector of one avocado image.

    Parameters
    ----------
    rgb : numpy.ndarray
        RGB uint8 image of shape (H, W, 3).

    Returns
    -------
    numpy.ndarray
        1-D float array with ``len(FEATURE_NAMES)`` entries, in the order of
        :data:`FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If the input is not an (H, W, 3) image, or if fewer than 2 % of the
        pixels are classified as fruit (empty frame / failed segmentation).
        Raising here is deliberate: a kiosk should ask the user to re-scan
        rather than silently return a ripeness for an empty tray.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {rgb.shape}")

    mask = fruit_mask(rgb)
    if mask.mean() < _MIN_FRUIT_FRACTION:
        raise ValueError("No fruit detected in image (segmentation found <2% fruit pixels)")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    h_deg = hsv[..., 0][mask] * 2.0          # OpenCV stores hue as 0-179 -> degrees
    s = hsv[..., 1][mask] / 255.0
    v = hsv[..., 2][mask] / 255.0

    h_rad = np.deg2rad(h_deg)
    c, sn = np.cos(h_rad).mean(), np.sin(h_rad).mean()
    hue_mean_deg = np.rad2deg(np.arctan2(sn, c)) % 360.0

    frac_green = np.mean((h_deg >= 60) & (h_deg < 150))
    frac_purple = np.mean((h_deg >= 270) | (h_deg < 30))
    frac_dark = np.mean(v < 0.25)

    return np.array([
        c, sn, hue_mean_deg,
        s.mean(), v.mean(), s.std(), v.std(),
        frac_green, frac_purple, frac_dark,
    ])


def features_from_file(path: str | Path, size: int | None = 200) -> np.ndarray:
    """Convenience wrapper: load an image file and return its features."""
    return extract_features(load_image(path, size=size))


def adjust_brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Scale image brightness to simulate a different lighting level.

    Used for robustness testing: ``factor`` < 1 darkens, > 1 brightens.
    Values are clipped to the valid 0-255 range.
    """
    out = rgb.astype(np.float64) * factor
    return np.clip(out, 0, 255).astype(np.uint8)
