import cv2
import numpy as np


def filter_posterize(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(roi)
    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)
    return out


def filter_halftone_dots(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius
    out = np.full_like(roi, 245)
    out[dot_mask] = (15, 15, 15)
    return out


def filter_rgb_glitch(roi: np.ndarray) -> np.ndarray:
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 0.72).astype(np.uint8)
    return out


def filter_thermal_heatmap(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filter_vintage_sepia(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    sepia_kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sepia = cv2.transform(roi, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    vignette = np.clip(1 - 0.5 * (dist / max_dist), 0, 1)[..., None]

    out = (sepia * vignette).astype(np.uint8)
    noise = np.random.randint(0, 25, out.shape, dtype=np.uint8)
    out = cv2.add(out, noise)
    return out


def filter_soft_white_glow(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    out = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)
    return out


def filter_pink_halftone(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out = np.full_like(roi, (215, 190, 245))
    out[dot_mask] = (55, 20, 130)
    return out


def filter_grid_overlay(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    color = (235, 235, 235)

    overlay = out.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    return out


def filter_ndvi(roi: np.ndarray) -> np.ndarray:
    """Pseudo-NDVI vegetation-health proxy from RGB (green channel stands in for NIR)."""
    b, g, r = cv2.split(roi.astype(np.float32))
    ndvi = (g - r) / (g + r + 1e-6)
    ndvi_norm = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_SUMMER)


def filter_night_vision(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    noise = np.random.randint(0, 20, gray.shape, dtype=np.uint8)
    gray = cv2.add(gray, noise)
    out = np.zeros_like(roi)
    out[:, :, 1] = gray
    return out


def filter_haze_index(roi: np.ndarray) -> np.ndarray:
    """Dark-channel-prior based haze/visibility density map."""
    b, g, r = cv2.split(roi)
    dark_channel = cv2.min(cv2.min(b, g), r)
    haze_norm = cv2.normalize(dark_channel, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(255 - haze_norm, cv2.COLORMAP_BONE)


FILTERS = [
    filter_grid_overlay,
    filter_posterize,
    filter_halftone_dots,
    filter_rgb_glitch,
    filter_thermal_heatmap,
    filter_vintage_sepia,
    filter_soft_white_glow,
    filter_pink_halftone,
    filter_ndvi,
    filter_night_vision,
    filter_haze_index,
]

FILTER_NAMES = {
    filter_grid_overlay: "Grid Overlay",
    filter_posterize: "Posterize",
    filter_halftone_dots: "Halftone Dots",
    filter_rgb_glitch: "RGB Glitch Shift",
    filter_thermal_heatmap: "Thermal Heatmap",
    filter_vintage_sepia: "Vintage Sepia",
    filter_soft_white_glow: "Soft White Glow",
    filter_pink_halftone: "Pink Halftone",
    filter_ndvi: "Pseudo-NDVI (Vegetation)",
    filter_night_vision: "Night Vision",
    filter_haze_index: "Haze Index",
}
