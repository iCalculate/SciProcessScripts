from __future__ import annotations

import numpy as np

from backend.services.media_assets import encode_photo_asset


def test_encode_photo_asset_resizes_grayscale_to_480p_bounds() -> None:
    image = np.linspace(0, 4095, num=1600 * 1200, dtype=np.float32).reshape(1600, 1200)

    encoded = encode_photo_asset(image)

    assert encoded.asset_format == "jpeg"
    assert encoded.original_width_px == 1600
    assert encoded.original_height_px == 1200
    assert encoded.width_px <= 854
    assert encoded.height_px <= 480
    assert encoded.channel_count == 1
    assert encoded.bit_depth == 8
    assert len(encoded.image_bytes) > 0


def test_encode_photo_asset_converts_rgba_to_rgb() -> None:
    image = np.zeros((400, 300, 4), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 3] = 200

    encoded = encode_photo_asset(image)

    assert encoded.asset_format == "jpeg"
    assert encoded.channel_count == 3
    assert encoded.width_px == 400
    assert encoded.height_px == 300
