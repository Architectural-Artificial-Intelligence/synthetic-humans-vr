import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import make_scene_thumbnails as T


def test_thumbnail_has_no_metadata():
    info = T.ensure_info()
    im, _ = T.make_thumbnail(1, info[info["scene"] == 1].iloc[0])
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=92); buf.seek(0)
    back = Image.open(buf)
    assert len(back.getexif()) == 0
    assert "xmp" not in back.info and "exif" not in back.info
    assert back.size == (T.OUT_W, T.OUT_H)


def test_originals_do_carry_gps_so_stripping_matters():
    assert len(Image.open(T.PHOTOS / "1.jpg").getexif().get_ifd(0x8825)) > 0


def test_blur_box_changes_only_the_box():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    before = img.copy()
    T.blur_box(img, (20, 20, 60, 60))
    assert not np.array_equal(img[20:60, 20:60], before[20:60, 20:60])
    assert np.array_equal(img[70:, 70:], before[70:, 70:])


def test_projection_centre_pixel_matches_panorama_centre():
    pano = np.zeros((200, 400, 3), np.uint8); pano[95:105, 195:205] = 255      # bright spot at lon 0, lat 0
    view = T.equirect_to_rect(pano, 0, 0, 90, 80, 60)
    assert view[30, 40].min() > 200
