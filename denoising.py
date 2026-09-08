"""Denoising algorithms used while preparing segmentation datasets."""

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter

try:
    from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


EPS = 1e-8
DENOISING_METHODS = ("none", "gaussian", "median", "bilateral", "tv", "rdpad")


def normalize_to_float01(image, eps=EPS):
    image = image.astype(np.float32)
    mn = float(np.nanmin(image))
    mx = float(np.nanmax(image))

    if mx - mn < eps:
        return np.zeros_like(image, dtype=np.float32), mn, mx

    return ((image - mn) / (mx - mn)).astype(np.float32), mn, mx


def restore_from_float01(image, mn, mx):
    return (image.astype(np.float32) * (mx - mn) + mn).astype(np.float32)


def denoise_gaussian_2d(image, sigma=1.0):
    return gaussian_filter(
        image.astype(np.float32),
        sigma=float(sigma),
    ).astype(np.float32)


def denoise_median_2d(image, size=3):
    return median_filter(
        image.astype(np.float32),
        size=int(size),
    ).astype(np.float32)


def denoise_bilateral_2d(image, sigma_color=0.05, sigma_spatial=3.0):
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image jest wymagany dla bilateral filter.")
    image01, mn, mx = normalize_to_float01(image)
    if mx - mn < EPS:
        return image.astype(np.float32)

    output01 = denoise_bilateral(
        image01,
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        channel_axis=None,
    )
    return restore_from_float01(output01, mn, mx)


def denoise_tv_2d(image, weight=0.08, eps=0.0002, max_num_iter=200):
    if not SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image jest wymagany dla total variation denoising."
        )
    image01, mn, mx = normalize_to_float01(image)
    if mx - mn < EPS:
        return image.astype(np.float32)

    output01 = denoise_tv_chambolle(
        image01,
        weight=float(weight),
        eps=float(eps),
        max_num_iter=int(max_num_iter),
        channel_axis=None,
    )
    return restore_from_float01(output01, mn, mx)


def _shift_reflect(image, dy, dx):
    padded = np.pad(image, ((2, 2), (2, 2)), mode="reflect")
    height, width = image.shape
    return padded[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]


def _rdpad_q2_modified_13_point(image):
    image = np.maximum(image.astype(np.float32), EPS)
    offsets = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
    ]
    values = np.stack(
        [_shift_reflect(image, dy, dx) for dy, dx in offsets], axis=0
    )
    value_sum = np.sum(values, axis=0)
    sum_squared_differences = np.zeros_like(image, dtype=np.float32)

    for index in range(values.shape[0] - 1):
        differences = values[index + 1 :] - values[index : index + 1]
        sum_squared_differences += np.sum(differences * differences, axis=0)

    q2 = (sum_squared_differences / 12.0) / (value_sum * value_sum + EPS)
    return np.maximum(q2, 0.0).astype(np.float32)


def _estimate_q0_squared(q2, mode="median", percentile=25.0):
    finite_values = q2[np.isfinite(q2)]
    if finite_values.size == 0:
        return np.float32(1e-4)

    if mode == "mean":
        value = float(np.mean(finite_values))
    elif mode == "percentile":
        value = float(np.percentile(finite_values, percentile))
    else:
        value = float(np.median(finite_values))

    return np.float32(max(value, EPS))


def _rdpad_diffusion_coefficient(q2, q0_squared):
    q0_squared = np.float32(max(float(q0_squared), EPS))
    ratio = (q2 - q0_squared) / (q0_squared * (1.0 + q2) + EPS)
    ratio = np.maximum(ratio, 0.0)
    coefficient = np.zeros_like(q2, dtype=np.float32)
    mask = ratio <= 1.0
    coefficient[mask] = 0.5 * np.square(1.0 - np.square(ratio[mask]))
    return np.clip(coefficient, 0.0, 0.5).astype(np.float32)


def _divergence_4n(image, coefficient):
    north = np.roll(image, 1, axis=0) - image
    south = np.roll(image, -1, axis=0) - image
    west = np.roll(image, 1, axis=1) - image
    east = np.roll(image, -1, axis=1) - image

    divergence = (
        np.roll(coefficient, 1, axis=0) * north
        + np.roll(coefficient, -1, axis=0) * south
        + np.roll(coefficient, 1, axis=1) * west
        + np.roll(coefficient, -1, axis=1) * east
    )
    divergence[[0, -1], :] = 0.0
    divergence[:, [0, -1]] = 0.0
    return divergence.astype(np.float32)


def denoise_rdpad_2d(
    image,
    iterations=50,
    timestep=0.15,
    q0_mode="median",
    q0_percentile=25.0,
    preserve_range=True,
):
    image = image.astype(np.float32)
    original_min = float(np.nanmin(image))
    original_max = float(np.nanmax(image))
    work = image - original_min + 1e-3
    work_max = float(np.max(work))
    if work_max > 0:
        work = work / work_max

    timestep = float(timestep)
    if timestep <= 0.0 or timestep > 0.25:
        raise ValueError("timestep powinien być w zakresie (0, 0.25].")

    for _ in range(int(iterations)):
        q2 = _rdpad_q2_modified_13_point(work)
        q0_squared = _estimate_q0_squared(q2, q0_mode, q0_percentile)
        coefficient = _rdpad_diffusion_coefficient(q2, q0_squared)
        work = np.maximum(work + timestep * _divergence_4n(work, coefficient), 0.0)

    output = work * work_max if work_max > 0 else work
    output = output - 1e-3 + original_min
    if preserve_range:
        output = np.clip(output, original_min, original_max)
    return output.astype(np.float32)


def denoise_2d(image, method="none"):
    """Apply one supported method using its reconstruction-project defaults."""
    method = str(method).strip().lower()
    functions = {
        "none": lambda value: value.astype(np.float32),
        "gaussian": denoise_gaussian_2d,
        "median": denoise_median_2d,
        "bilateral": denoise_bilateral_2d,
        "tv": denoise_tv_2d,
        "rdpad": denoise_rdpad_2d,
    }
    if method not in functions:
        raise ValueError(
            f"Unknown denoising method: {method}. Available: {DENOISING_METHODS}"
        )
    return functions[method](image)
