from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np


PHOTO_IMAGE = "photo_image"
PHOTO_LANDSCAPE_BOUNDS = (854, 480)
PHOTO_PORTRAIT_BOUNDS = (480, 854)
PHOTO_JPEG_QUALITY = 82


@dataclass(frozen=True)
class EncodedPhotoAsset:
    image_bytes: bytes
    asset_format: str
    width_px: int
    height_px: int
    original_width_px: int
    original_height_px: int
    channel_count: int
    bit_depth: int


def encode_photo_asset(
    image_array: Any,
    *,
    max_width_px: int = PHOTO_LANDSCAPE_BOUNDS[0],
    max_height_px: int = PHOTO_LANDSCAPE_BOUNDS[1],
    jpeg_quality: int = PHOTO_JPEG_QUALITY,
) -> EncodedPhotoAsset:
    image_module = _load_pillow_image_module()
    oriented_array = _orient_image_array(image_array)
    if oriented_array.ndim == 2:
        normalized_array = _normalize_scalar_plane_to_uint8(oriented_array)
        image = image_module.fromarray(normalized_array, mode="L")
        channel_count = 1
    elif oriented_array.ndim == 3:
        normalized_array = _normalize_multichannel_to_uint8(oriented_array)
        if normalized_array.ndim == 2:
            image = image_module.fromarray(normalized_array, mode="L")
            channel_count = 1
        else:
            image = image_module.fromarray(normalized_array, mode="RGB")
            channel_count = 3
    else:
        raise ValueError(f"Unsupported image array shape: {oriented_array.shape}")

    original_width_px, original_height_px = image.size
    target_size = _fit_photo_size(
        width_px=original_width_px,
        height_px=original_height_px,
        max_width_px=max_width_px,
        max_height_px=max_height_px,
    )
    if target_size != image.size:
        resample = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
        image = image.resize(target_size, resample=resample)

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(jpeg_quality),
        optimize=True,
        progressive=True,
    )
    encoded_bytes = buffer.getvalue()
    width_px, height_px = image.size
    return EncodedPhotoAsset(
        image_bytes=encoded_bytes,
        asset_format="jpeg",
        width_px=int(width_px),
        height_px=int(height_px),
        original_width_px=int(original_width_px),
        original_height_px=int(original_height_px),
        channel_count=int(channel_count),
        bit_depth=8,
    )


def _load_pillow_image_module():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("Pillow is required for WITec photo import") from exc
    return Image


def _fit_photo_size(*, width_px: int, height_px: int, max_width_px: int, max_height_px: int) -> tuple[int, int]:
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid image size: {(width_px, height_px)}")

    bounds = (
        (int(max_width_px), int(max_height_px))
        if width_px >= height_px
        else (int(max_height_px), int(max_width_px))
    )
    scale = min(1.0, bounds[0] / width_px, bounds[1] / height_px)
    return (
        max(1, int(round(width_px * scale))),
        max(1, int(round(height_px * scale))),
    )


def _orient_image_array(image_array: Any) -> np.ndarray:
    array = np.asarray(image_array)
    if array.ndim == 2:
        return np.ascontiguousarray(array.T)
    if array.ndim == 3:
        return np.ascontiguousarray(array.transpose(1, 0, 2))
    raise ValueError(f"Unsupported image array shape: {array.shape}")


def _normalize_multichannel_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.shape[2] <= 0:
        raise ValueError(f"Unsupported channel count: {array.shape}")

    if array.shape[2] == 1:
        return _normalize_scalar_plane_to_uint8(array[:, :, 0])

    working = array[:, :, :4]
    if working.shape[2] == 4:
        alpha = _normalize_scalar_plane_to_uint8(working[:, :, 3]).astype(np.float32) / 255.0
        rgb = np.stack(
            [_normalize_scalar_plane_to_uint8(working[:, :, channel]) for channel in range(3)],
            axis=2,
        ).astype(np.float32)
        composited = np.rint((rgb * alpha[:, :, None]) + (255.0 * (1.0 - alpha[:, :, None]))).astype(np.uint8)
        return composited

    channels = [
        _normalize_scalar_plane_to_uint8(working[:, :, channel])
        for channel in range(min(3, working.shape[2]))
    ]
    while len(channels) < 3:
        channels.append(channels[-1].copy())
    return np.stack(channels[:3], axis=2)


def _normalize_scalar_plane_to_uint8(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        minimum = int(np.min(array))
        maximum = int(np.max(array))
        if minimum >= 0 and maximum <= 255:
            return np.asarray(array, dtype=np.uint8)

    working = np.asarray(array, dtype=np.float32)
    finite_values = working[np.isfinite(working)]
    if finite_values.size == 0:
        return np.zeros(working.shape, dtype=np.uint8)

    if np.issubdtype(array.dtype, np.integer) and finite_values.size > 1:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
    else:
        low, high = np.percentile(finite_values, [1, 99]).astype(float).tolist()

    if not np.isfinite(low):
        low = float(np.min(finite_values))
    if not np.isfinite(high):
        high = float(np.max(finite_values))
    if high <= low:
        return np.zeros(working.shape, dtype=np.uint8)

    normalized = np.clip((working - low) / (high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(normalized * 255.0).astype(np.uint8)
