import io
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple

import fitz  # PyMuPDF
from PIL import Image

# Maximum dimension (width or height) for rasterized output.
# Bounds memory usage regardless of source PDF DPI.
MAX_OUTPUT_DIMENSION = 2000


class Extractor:
    def __init__(self, file_path: Path, *, max_dimension: int = MAX_OUTPUT_DIMENSION):
        self.file_path = file_path
        self.suffix = file_path.suffix.lower()
        self.max_dimension = max_dimension
        self._cbz_names: List[str] | None = None
        self._pdf_doc: fitz.Document | None = None
        self._cbz_file: zipfile.ZipFile | None = None

    def __enter__(self) -> "Extractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release cached file handles."""
        if self._pdf_doc is not None:
            self._pdf_doc.close()
            self._pdf_doc = None
        if self._cbz_file is not None:
            self._cbz_file.close()
            self._cbz_file = None

    def _get_pdf_doc(self) -> fitz.Document:
        """Return cached PDF document, opening if needed."""
        if self._pdf_doc is None:
            self._pdf_doc = fitz.open(self.file_path)
        return self._pdf_doc

    def _get_cbz_file(self) -> zipfile.ZipFile:
        """Return cached ZipFile, opening if needed."""
        if self._cbz_file is None:
            self._cbz_file = zipfile.ZipFile(self.file_path, "r")
        return self._cbz_file

    def _pdf_zoom_factor(self, page: fitz.Page) -> float:
        """Calculate zoom factor to cap output at max_dimension."""
        rect = page.rect
        max_src = max(rect.width, rect.height)
        if max_src <= 0:
            return 1.0
        # If source is already smaller than target, use 1:1
        if max_src <= self.max_dimension:
            return 1.0
        return self.max_dimension / max_src

    def page_count(self) -> int:
        if self.suffix in {".cbz", ".zip"}:
            return len(self._cbz_image_names())
        if self.suffix == ".pdf":
            return len(self._get_pdf_doc())
        if self.suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return 1
        raise ValueError(f"Unsupported file format: {self.suffix}")

    def get_page(self, index: int) -> Image.Image:
        """Return a single page image by 0-based index."""
        if index < 0:
            raise IndexError("Page index must be >= 0")

        if self.suffix in {".cbz", ".zip"}:
            names = self._cbz_image_names()
            if index >= len(names):
                raise IndexError(f"Page index out of range (got {index}, max {len(names) - 1})")

            z = self._get_cbz_file()
            with z.open(names[index]) as f:
                img_data = f.read()
                return Image.open(io.BytesIO(img_data)).convert("RGB")

        if self.suffix == ".pdf":
            doc = self._get_pdf_doc()
            if index >= len(doc):
                raise IndexError(f"Page index out of range (got {index}, max {len(doc) - 1})")
            page = doc[index]
            zoom = self._pdf_zoom_factor(page)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data)).convert("RGB")

        if self.suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            if index != 0:
                raise IndexError("Single-image input only has page 0")
            return Image.open(self.file_path).convert("RGB")

        raise ValueError(f"Unsupported file format: {self.suffix}")

    def iter_pages(self) -> Iterator[Tuple[int, Image.Image]]:
        """Yields (index, PIL Image) for each page in the book."""
        if self.suffix == ".cbz" or self.suffix == ".zip":
            yield from self._iter_cbz()
        elif self.suffix == ".pdf":
            yield from self._iter_pdf()
        elif self.suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield from self._iter_image()
        else:
            raise ValueError(f"Unsupported file format: {self.suffix}")

    def _iter_cbz(self) -> Iterator[Tuple[int, Image.Image]]:
        z = self._get_cbz_file()
        names = self._cbz_image_names()
        for i, name in enumerate(names):
            with z.open(name) as f:
                img_data = f.read()
                yield i, Image.open(io.BytesIO(img_data)).convert("RGB")

    def _iter_pdf(self) -> Iterator[Tuple[int, Image.Image]]:
        doc = self._get_pdf_doc()
        for i in range(len(doc)):
            page = doc[i]
            zoom = self._pdf_zoom_factor(page)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = pix.tobytes("png")
            yield i, Image.open(io.BytesIO(img_data)).convert("RGB")

    def _iter_image(self) -> Iterator[Tuple[int, Image.Image]]:
        """Process a single image file."""
        img = Image.open(self.file_path).convert("RGB")
        yield 0, img

    def _cbz_image_names(self) -> List[str]:
        if self._cbz_names is not None:
            return self._cbz_names

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        z = self._get_cbz_file()
        names = [n for n in z.namelist() if Path(n).suffix.lower() in image_extensions]
        self._cbz_names = sorted(names)
        return self._cbz_names
