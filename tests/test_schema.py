from datetime import datetime

import pytest
from pydantic import ValidationError

from panel_flow.schema import (
    BookData,
    BookMetadata,
    DetectionSource,
    Page,
    PageOverride,
    Panel,
    PanelOverride,
    ReadingDirection,
)


class TestDetectionSource:
    def test_values(self) -> None:
        assert DetectionSource.CV == "cv"
        assert DetectionSource.YOLO == "yolo"
        assert DetectionSource.SAM == "sam"
        assert DetectionSource.VLM == "vlm"
        assert DetectionSource.MANUAL == "manual"


class TestReadingDirection:
    def test_values(self) -> None:
        assert ReadingDirection.LTR == "ltr"
        assert ReadingDirection.RTL == "rtl"


class TestPanel:
    def test_create(self) -> None:
        panel = Panel(id="p-0", bbox=(10, 20, 100, 200), confidence=0.85)
        assert panel.id == "p-0"
        assert panel.bbox == (10, 20, 100, 200)
        assert panel.confidence == 0.85

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Panel(id="p-0", bbox=(0, 0, 10, 10), confidence=1.5)
        with pytest.raises(ValidationError):
            Panel(id="p-0", bbox=(0, 0, 10, 10), confidence=-0.1)


class TestPage:
    def test_create_minimal(self) -> None:
        page = Page(index=0, size=(800, 600), source=DetectionSource.CV, order_confidence=None)
        assert page.index == 0
        assert page.size == (800, 600)
        assert page.panels == []
        assert page.order == []
        assert page.user_override is False

    def test_create_with_panels(self) -> None:
        panels = [Panel(id="p-0", bbox=(0, 0, 100, 100), confidence=0.9)]
        page = Page(
            index=1,
            size=(800, 600),
            panels=panels,
            order=["p-0"],
            order_confidence=0.95,
            source=DetectionSource.MANUAL,
            user_override=True,
        )
        assert len(page.panels) == 1
        assert page.order == ["p-0"]
        assert page.order_confidence == 0.95
        assert page.user_override is True


class TestOverrides:
    def test_panel_override(self) -> None:
        override = PanelOverride(bbox=(10, 20, 30, 40))
        assert override.bbox == (10, 20, 30, 40)

    def test_panel_override_empty(self) -> None:
        override = PanelOverride()
        assert override.bbox is None

    def test_page_override(self) -> None:
        override = PageOverride(order=["p-1", "p-0"])
        assert override.order == ["p-1", "p-0"]


class TestBookMetadata:
    def test_create(self) -> None:
        meta = BookMetadata(tool_version="0.1.0")
        assert meta.reading_direction == ReadingDirection.LTR
        assert meta.tool_version == "0.1.0"
        assert isinstance(meta.created, datetime)

    def test_rtl_direction(self) -> None:
        meta = BookMetadata(reading_direction=ReadingDirection.RTL, tool_version="0.1.0")
        assert meta.reading_direction == ReadingDirection.RTL


class TestBookData:
    def test_create(self) -> None:
        page = Page(index=0, size=(800, 600), source=DetectionSource.CV, order_confidence=None)
        meta = BookMetadata(tool_version="0.1.0")
        book = BookData(book_hash="sha256:abc123", pages=[page], metadata=meta)
        assert book.version == 1
        assert book.book_hash == "sha256:abc123"
        assert len(book.pages) == 1
        assert book.overrides == {}
