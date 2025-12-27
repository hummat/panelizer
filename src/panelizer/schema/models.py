from datetime import UTC, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class DetectionSource(str, Enum):
    CV = "cv"
    YOLO = "yolo"
    SAM = "sam"
    VLM = "vlm"
    MANUAL = "manual"


class ReadingDirection(str, Enum):
    LTR = "ltr"
    RTL = "rtl"


class Panel(BaseModel):
    id: str
    bbox: Tuple[int, int, int, int] = Field(..., description="[x, y, width, height]")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Page(BaseModel):
    index: int
    size: Tuple[int, int] = Field(..., description="[width, height]")
    panels: List[Panel] = Field(default_factory=list)
    order: List[str] = Field(default_factory=list, description="Ordered list of panel IDs")
    order_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: DetectionSource
    user_override: bool = False
    gutters: Optional[Tuple[int, int]] = Field(None, description="[x_gutter, y_gutter] in pixels")
    processing_time: Optional[float] = Field(None, description="Detection time in seconds")


class PanelOverride(BaseModel):
    bbox: Optional[Tuple[int, int, int, int]] = None


class PageOverride(BaseModel):
    order: Optional[List[str]] = None


class BookMetadata(BaseModel):
    reading_direction: ReadingDirection = ReadingDirection.LTR
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_version: str


class BookData(BaseModel):
    version: int = 1
    book_hash: str
    pages: List[Page]
    overrides: Dict[str, Union[PanelOverride, PageOverride]] = Field(default_factory=dict)
    metadata: BookMetadata
