import io
import tempfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from panel_flow.extraction import Extractor, calculate_book_hash


class TestCalculateBookHash:
    def test_hash_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            result = calculate_book_hash(path)
            assert result.startswith("sha256:")
            assert len(result) == 7 + 64  # "sha256:" + 64 hex chars
        finally:
            path.unlink()

    def test_same_content_same_hash(self) -> None:
        content = b"identical content"
        hashes = []

        for _ in range(2):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(content)
                f.flush()
                path = Path(f.name)
            try:
                hashes.append(calculate_book_hash(path))
            finally:
                path.unlink()

        assert hashes[0] == hashes[1]


class TestExtractor:
    def test_cbz_extraction(self) -> None:
        # Create a minimal CBZ (zip with images)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cbz") as f:
            cbz_path = Path(f.name)

        try:
            with zipfile.ZipFile(cbz_path, "w") as z:
                # Create a simple test image
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr("page001.png", buf.getvalue())

                img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
                buf2 = io.BytesIO()
                img2.save(buf2, format="PNG")
                z.writestr("page002.png", buf2.getvalue())

            extractor = Extractor(cbz_path)
            pages = list(extractor.iter_pages())

            assert len(pages) == 2
            assert pages[0][0] == 0  # first index
            assert pages[1][0] == 1  # second index
            assert isinstance(pages[0][1], Image.Image)
        finally:
            cbz_path.unlink()

    def test_cbz_filters_non_images(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cbz") as f:
            cbz_path = Path(f.name)

        try:
            with zipfile.ZipFile(cbz_path, "w") as z:
                # Add an image
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr("page001.png", buf.getvalue())
                # Add non-image files that should be ignored
                z.writestr("metadata.xml", b"<xml/>")
                z.writestr("readme.txt", b"readme")

            extractor = Extractor(cbz_path)
            pages = list(extractor.iter_pages())

            assert len(pages) == 1  # only the image
        finally:
            cbz_path.unlink()

    def test_unsupported_format(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as f:
            path = Path(f.name)

        try:
            extractor = Extractor(path)
            with pytest.raises(ValueError, match="Unsupported file format"):
                list(extractor.iter_pages())
        finally:
            path.unlink()

    def test_zip_extension_treated_as_cbz(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            zip_path = Path(f.name)

        try:
            with zipfile.ZipFile(zip_path, "w") as z:
                img = Image.new("RGB", (100, 100), color=(0, 0, 255))
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                z.writestr("page.jpg", buf.getvalue())

            extractor = Extractor(zip_path)
            pages = list(extractor.iter_pages())
            assert len(pages) == 1
        finally:
            zip_path.unlink()

    def test_single_image_jpg(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            img_path = Path(f.name)

        try:
            # Create a test image
            img = Image.new("RGB", (200, 150), color=(100, 150, 200))
            img.save(img_path, format="JPEG")

            extractor = Extractor(img_path)
            pages = list(extractor.iter_pages())

            assert len(pages) == 1
            assert pages[0][0] == 0  # index 0
            assert isinstance(pages[0][1], Image.Image)
            assert pages[0][1].size == (200, 150)
        finally:
            img_path.unlink()

    def test_single_image_png(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            # Create a test image
            img = Image.new("RGB", (300, 200), color=(255, 128, 64))
            img.save(img_path, format="PNG")

            extractor = Extractor(img_path)
            pages = list(extractor.iter_pages())

            assert len(pages) == 1
            assert pages[0][0] == 0
            assert isinstance(pages[0][1], Image.Image)
            assert pages[0][1].size == (300, 200)
        finally:
            img_path.unlink()
