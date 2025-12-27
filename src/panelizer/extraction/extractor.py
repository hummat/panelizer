import io
import zipfile
from pathlib import Path
from typing import Iterator, Tuple

import fitz  # PyMuPDF
from PIL import Image


class Extractor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.suffix = file_path.suffix.lower()

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
        with zipfile.ZipFile(self.file_path, "r") as z:
            # Filter for image extensions and sort naturally
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            names = sorted([n for n in z.namelist() if Path(n).suffix.lower() in image_extensions])

            for i, name in enumerate(names):
                with z.open(name) as f:
                    img_data = f.read()
                    yield i, Image.open(io.BytesIO(img_data)).convert("RGB")

    def _iter_pdf(self) -> Iterator[Tuple[int, Image.Image]]:
        doc = fitz.open(self.file_path)
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better detection
            img_data = pix.tobytes("png")
            yield i, Image.open(io.BytesIO(img_data)).convert("RGB")
        doc.close()

    def _iter_image(self) -> Iterator[Tuple[int, Image.Image]]:
        """Process a single image file."""
        img = Image.open(self.file_path).convert("RGB")
        yield 0, img
