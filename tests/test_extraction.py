import io
import tempfile
import zipfile
from pathlib import Path

import fitz
import pytest
from PIL import Image

from panelizer.extraction import Extractor, calculate_book_hash


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

    def test_page_count_and_get_page_cbz(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cbz") as f:
            cbz_path = Path(f.name)

        try:
            with zipfile.ZipFile(cbz_path, "w") as z:
                for i in range(2):
                    img = Image.new("RGB", (120, 90), color=(10 * i, 0, 0))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    z.writestr(f"page{i:03d}.png", buf.getvalue())

            extractor = Extractor(cbz_path)
            assert extractor.page_count() == 2

            p0 = extractor.get_page(0)
            assert isinstance(p0, Image.Image)
            assert p0.size == (120, 90)

            p1 = extractor.get_page(1)
            assert isinstance(p1, Image.Image)
            assert p1.size == (120, 90)
        finally:
            cbz_path.unlink()

    def test_page_count_and_get_page_single_image(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            img = Image.new("RGB", (64, 48), color=(1, 2, 3))
            img.save(img_path, format="PNG")

            extractor = Extractor(img_path)
            assert extractor.page_count() == 1
            p0 = extractor.get_page(0)
            assert p0.size == (64, 48)
        finally:
            img_path.unlink()

    def test_get_page_out_of_range_raises(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            img = Image.new("RGB", (32, 32), color=(255, 255, 255))
            img.save(img_path, format="PNG")

            extractor = Extractor(img_path)
            with pytest.raises(IndexError):
                extractor.get_page(1)
        finally:
            img_path.unlink()

    def test_pdf_page_count_and_get_page(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            pdf_path = Path(f.name)

        try:
            doc = fitz.open()
            doc.new_page(width=200, height=120)
            doc.save(pdf_path)
            doc.close()

            extractor = Extractor(pdf_path)
            assert extractor.page_count() == 1

            img = extractor.get_page(0)
            assert isinstance(img, Image.Image)
            assert img.size[0] > 0 and img.size[1] > 0
        finally:
            pdf_path.unlink()
