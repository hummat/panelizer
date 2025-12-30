import io
import json
import mimetypes
import threading
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Dict, Generic, Optional, Tuple, TypeVar
from urllib.parse import parse_qs, urlparse

from PIL import Image

from ..cv.debug import DebugContext
from ..cv.detector import CVDetector
from ..extraction.extractor import Extractor
from ..extraction.utils import calculate_book_hash
from ..ml import is_ml_available
from ..ordering import order_panels
from ..schema import DetectionSource, ReadingDirection

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple thread-safe LRU cache with bounded size."""

    def __init__(self, maxsize: int = 20):
        self._maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _json_bytes(obj: Dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


# Default cache size: ~20 pages balances memory vs navigation locality
DEFAULT_CACHE_SIZE = 20


@dataclass
class CVSettings:
    """CV detection settings exposed in debug mode."""

    min_panel_ratio: float = 0.1
    min_segment_ratio: float = 0.05
    # Post-processing (off by default to expose raw detection quality)
    panel_expansion: bool = False
    small_panel_grouping: bool = False
    big_panel_grouping: bool = False
    panel_splitting: bool = False  # Expensive O(n²) polygon splitting, often no-op
    # Preprocessing
    use_denoising: bool = True
    use_canny: bool = False
    use_morphological_close: bool = False
    # LSD (Line Segment Detector) settings
    max_segments: int = 500
    prefer_axis_aligned: bool = True  # Prefer horizontal/vertical gutters
    use_lsd_nfa: bool = False  # Use NFA quality scores (slower but more accurate)
    # Performance
    skip_scoring: bool = False  # Skip confidence scoring (faster CV-only mode)
    max_dimension: int = 2000  # Downscale images larger than this (0 = no limit)


@dataclass
class PreviewConfig:
    file_path: Path
    reading_direction: ReadingDirection
    host: str = "127.0.0.1"
    port: int = 0
    cache_size: int = DEFAULT_CACHE_SIZE
    debug: bool = False
    debug_dir: Optional[Path] = None
    # ML options
    use_ml: bool = False
    ml_fallback: bool = True
    confidence_threshold: float = 0.7
    device: Optional[str] = None  # None = auto


class PreviewApp:
    def __init__(self, config: PreviewConfig) -> None:
        self.config = config
        self.extractor = Extractor(config.file_path)
        self.cv_settings = CVSettings()
        self.cv_detector = self._build_cv_detector()
        self.ml_detector = None
        self.ml_fallback = config.ml_fallback

        # Initialize ML detector if needed
        if config.use_ml or config.ml_fallback:
            if is_ml_available():
                from ..ml import YOLODetector

                self.ml_detector = YOLODetector(device=config.device)
            elif config.ml_fallback:
                self.ml_fallback = False  # Disable fallback if deps not installed

        self._book_hash: Optional[str] = None
        self._page_png_cache: LRUCache[int, bytes] = LRUCache(config.cache_size)
        self._page_json_cache: LRUCache[int, Dict] = LRUCache(config.cache_size)
        self._lock = threading.Lock()

    def _build_cv_detector(self) -> CVDetector:
        """Build CV detector from current settings."""
        s = self.cv_settings
        return CVDetector(
            min_panel_ratio=s.min_panel_ratio,
            min_segment_ratio=s.min_segment_ratio,
            panel_expansion=s.panel_expansion,
            small_panel_grouping=s.small_panel_grouping,
            big_panel_grouping=s.big_panel_grouping,
            panel_splitting=s.panel_splitting,
            use_denoising=s.use_denoising,
            use_canny=s.use_canny,
            use_morphological_close=s.use_morphological_close,
            max_segments=s.max_segments,
            prefer_axis_aligned=s.prefer_axis_aligned,
            use_lsd_nfa=s.use_lsd_nfa,
            skip_scoring=s.skip_scoring,
            max_dimension=s.max_dimension,
        )

    def get_settings(self) -> Dict:
        """Get current CV settings as dict."""
        s = self.cv_settings
        return {
            "min_panel_ratio": s.min_panel_ratio,
            "min_segment_ratio": s.min_segment_ratio,
            "panel_expansion": s.panel_expansion,
            "small_panel_grouping": s.small_panel_grouping,
            "big_panel_grouping": s.big_panel_grouping,
            "panel_splitting": s.panel_splitting,
            "use_denoising": s.use_denoising,
            "use_canny": s.use_canny,
            "use_morphological_close": s.use_morphological_close,
            "max_segments": s.max_segments,
            "prefer_axis_aligned": s.prefer_axis_aligned,
            "use_lsd_nfa": s.use_lsd_nfa,
            "skip_scoring": s.skip_scoring,
            "max_dimension": s.max_dimension,
        }

    def update_settings(self, updates: Dict) -> None:
        """Update CV settings and rebuild detector."""
        s = self.cv_settings
        if "min_panel_ratio" in updates:
            s.min_panel_ratio = float(updates["min_panel_ratio"])
        if "min_segment_ratio" in updates:
            s.min_segment_ratio = float(updates["min_segment_ratio"])
        if "panel_expansion" in updates:
            s.panel_expansion = updates["panel_expansion"] in (True, "true", "1", 1)
        if "small_panel_grouping" in updates:
            s.small_panel_grouping = updates["small_panel_grouping"] in (True, "true", "1", 1)
        if "big_panel_grouping" in updates:
            s.big_panel_grouping = updates["big_panel_grouping"] in (True, "true", "1", 1)
        if "panel_splitting" in updates:
            s.panel_splitting = updates["panel_splitting"] in (True, "true", "1", 1)
        if "use_denoising" in updates:
            s.use_denoising = updates["use_denoising"] in (True, "true", "1", 1)
        if "use_canny" in updates:
            s.use_canny = updates["use_canny"] in (True, "true", "1", 1)
        if "use_morphological_close" in updates:
            s.use_morphological_close = updates["use_morphological_close"] in (True, "true", "1", 1)
        if "max_segments" in updates:
            s.max_segments = int(updates["max_segments"])
        if "prefer_axis_aligned" in updates:
            s.prefer_axis_aligned = updates["prefer_axis_aligned"] in (True, "true", "1", 1)
        if "use_lsd_nfa" in updates:
            s.use_lsd_nfa = updates["use_lsd_nfa"] in (True, "true", "1", 1)
        if "skip_scoring" in updates:
            s.skip_scoring = updates["skip_scoring"] in (True, "true", "1", 1)
        if "max_dimension" in updates:
            s.max_dimension = int(updates["max_dimension"])

        # Rebuild detector and clear cache
        self.cv_detector = self._build_cv_detector()
        self._page_json_cache.clear()

    def book_info(self) -> Dict:
        with self._lock:
            if self._book_hash is None:
                self._book_hash = calculate_book_hash(self.config.file_path)

        try:
            tool_version = pkg_version("panelizer")
        except PackageNotFoundError:
            tool_version = "0.1.0"

        return {
            "file_name": self.config.file_path.name,
            "book_hash": self._book_hash,
            "page_count": self.extractor.page_count(),
            "reading_direction": self.config.reading_direction.value,
            "tool_version": tool_version,
            "debug": self.config.debug,
        }

    def set_debug(self, enabled: bool) -> None:
        """Toggle debug mode. Clears cache when changed."""
        if self.config.debug != enabled:
            self.config.debug = enabled
            self._page_json_cache.clear()  # Re-detect with/without debug

    def _get_debug_dir(self, page_index: int) -> Path:
        """Get the debug output directory for a page."""
        base = self.config.debug_dir or Path(f"{self.config.file_path}.debug")
        return base / f"page-{page_index:04d}"

    def debug_steps(self, page_index: int) -> list:
        """Get list of debug steps for a page (from cached detection or disk)."""
        debug_dir = self._get_debug_dir(page_index)
        if not debug_dir.exists():
            return []

        # Find all debug images
        steps = []
        for img_path in sorted(debug_dir.glob("*.jpg")):
            # Parse filename like "00-input.jpg" -> "input"
            name = img_path.stem
            if "-" in name:
                name = name.split("-", 1)[1]
            steps.append({"name": name, "file": img_path.name})
        return steps

    def _clear_debug_dir(self, page_index: int) -> None:
        """Clear debug directory for a page before re-running detection."""
        debug_dir = self._get_debug_dir(page_index)
        if debug_dir.exists():
            for f in debug_dir.glob("*.jpg"):
                f.unlink()
            for f in debug_dir.glob("*.html"):
                f.unlink()

    def debug_image(self, page_index: int, filename: str) -> Optional[bytes]:
        """Get a debug image by filename."""
        debug_dir = self._get_debug_dir(page_index)
        img_path = debug_dir / filename
        # Security: ensure we're not escaping debug_dir
        if not img_path.resolve().is_relative_to(debug_dir.resolve()):
            return None
        if not img_path.exists():
            return None
        return img_path.read_bytes()

    def page_png(self, index: int, *, refresh: bool = False) -> bytes:
        if not refresh:
            cached = self._page_png_cache.get(index)
            if cached is not None:
                return cached

        img = self.extractor.get_page(index)
        data = _png_bytes(img)
        self._page_png_cache.set(index, data)
        return data

    def page_json(self, index: int, *, refresh: bool = False) -> Dict:
        if not refresh:
            cached = self._page_json_cache.get(index)
            if cached is not None:
                return cached

        img = self.extractor.get_page(index)

        # Set up debug context if enabled
        debug_ctx = None
        if self.config.debug:
            self._clear_debug_dir(index)  # Clear old debug images
            page_debug_dir = self._get_debug_dir(index)
            debug_ctx = DebugContext(enabled=True, output_dir=page_debug_dir)

        # Determine which detector to use
        if self.config.use_ml and self.ml_detector:
            # Force ML detection
            result = self.ml_detector.detect(img)
            source = DetectionSource.YOLO
        else:
            # CV detection first
            result = self.cv_detector.detect(img, debug=debug_ctx)
            source = DetectionSource.CV

            # Check for ML fallback
            if self.ml_fallback and self.ml_detector and result.confidence < self.config.confidence_threshold:
                print(f"  Page {index}: low CV confidence ({result.confidence:.2f}), trying ML...")
                ml_result = self.ml_detector.detect(img)

                # Use ML result if it found panels
                if ml_result.panels:
                    result = ml_result
                    source = DetectionSource.YOLO

        # Print debug info and save HTML
        if debug_ctx and debug_ctx.enabled:
            print(f"\n[DEBUG] Page {index}:")
            for i, step in enumerate(debug_ctx.steps):
                print(f"  {i + 1}. {step.name}: {step.panel_count} panels ({step.elapsed_ms:.1f}ms)")
            print(f"  Total: {debug_ctx.total_time_ms():.1f}ms, {len(result.panels)} final panels")
            html_path = debug_ctx.save_html()
            if html_path:
                print(f"  Debug HTML: {html_path}")

        bboxes = [p.bbox for p in result.panels]
        ordered_indices = order_panels(bboxes, self.config.reading_direction)

        page = {
            "index": index,
            "size": [img.width, img.height],
            "panels": [p.model_dump() for p in result.panels],
            "order": [result.panels[i].id for i in ordered_indices],
            "order_confidence": 0.9,
            "source": source.value,
            "user_override": False,
            "cv_confidence": float(result.confidence),
            "gutters": result.gutters,
            "processing_time": result.processing_time,
            "skip_scoring": self.cv_settings.skip_scoring,
        }

        self._page_json_cache.set(index, page)
        return page


def _static_bytes(path: str) -> Tuple[bytes, str]:
    if path in {"/", "/index.html"}:
        rel = "index.html"
    else:
        rel = path.lstrip("/")

    allowed = {
        "index.html",
        "styles.css",
        "app.js",
    }
    if rel not in allowed:
        raise FileNotFoundError(rel)

    data = resources.files("panelizer.preview").joinpath("static", rel).read_bytes()
    content_type, _ = mimetypes.guess_type(rel)
    return data, content_type or "application/octet-stream"


def dispatch_request(app: PreviewApp, path: str, qs: Dict[str, list]) -> Tuple[int, bytes, str, str]:
    """
    Pure routing logic for the viewer server.

    Returns (status_code, body, content_type, cache_control).
    """
    try:
        if path.startswith("/api/"):
            if path == "/api/book":
                return (
                    HTTPStatus.OK,
                    _json_bytes(app.book_info()),
                    "application/json; charset=utf-8",
                    "no-store",
                )

            if path.startswith("/api/page/") and path.endswith(".png"):
                idx = int(path.removeprefix("/api/page/").removesuffix(".png"))
                refresh = qs.get("refresh", ["0"])[0] == "1"
                return (
                    HTTPStatus.OK,
                    app.page_png(idx, refresh=refresh),
                    "image/png",
                    "no-store",
                )

            # Debug toggle: /api/debug?set=1 or /api/debug?set=0
            if path == "/api/debug":
                if "set" in qs:
                    app.set_debug(qs["set"][0] == "1")
                return (
                    HTTPStatus.OK,
                    _json_bytes({"debug": app.config.debug}),
                    "application/json; charset=utf-8",
                    "no-store",
                )

            # CV settings: GET returns current, query params update
            if path == "/api/settings":
                # Parse updates from query string (all supported keys)
                updates = {}
                for key in [
                    "min_panel_ratio",
                    "min_segment_ratio",
                    "panel_expansion",
                    "small_panel_grouping",
                    "big_panel_grouping",
                    "panel_splitting",
                    "use_denoising",
                    "use_canny",
                    "use_morphological_close",
                    "max_segments",
                    "prefer_axis_aligned",
                    "use_lsd_nfa",
                    "skip_scoring",
                    "max_dimension",
                ]:
                    if key in qs:
                        updates[key] = qs[key][0]
                if updates:
                    app.update_settings(updates)
                return (
                    HTTPStatus.OK,
                    _json_bytes(app.get_settings()),
                    "application/json; charset=utf-8",
                    "no-store",
                )

            # Debug steps list: /api/page/{idx}/debug.json (must come before generic .json)
            if path.startswith("/api/page/") and path.endswith("/debug.json"):
                idx = int(path.removeprefix("/api/page/").removesuffix("/debug.json"))
                return (
                    HTTPStatus.OK,
                    _json_bytes({"steps": app.debug_steps(idx)}),
                    "application/json; charset=utf-8",
                    "no-store",
                )

            # Debug image: /api/page/{idx}/debug/{filename}
            if "/debug/" in path and path.startswith("/api/page/"):
                # Parse /api/page/{idx}/debug/{filename}
                rest = path.removeprefix("/api/page/")
                idx_str, _, filename = rest.partition("/debug/")
                idx = int(idx_str)
                data = app.debug_image(idx, filename)
                if data is None:
                    return (HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8", "no-store")
                return (HTTPStatus.OK, data, "image/jpeg", "no-store")

            # Generic page JSON (after debug routes)
            if path.startswith("/api/page/") and path.endswith(".json"):
                idx = int(path.removeprefix("/api/page/").removesuffix(".json"))
                refresh = qs.get("refresh", ["0"])[0] == "1"
                return (
                    HTTPStatus.OK,
                    _json_bytes(app.page_json(idx, refresh=refresh)),
                    "application/json; charset=utf-8",
                    "no-store",
                )

            return (HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8", "no-store")

        data, content_type = _static_bytes(path)
        return (HTTPStatus.OK, data, content_type, "no-cache")
    except FileNotFoundError:
        return (HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8", "no-store")
    except ValueError:
        return (HTTPStatus.BAD_REQUEST, b"Invalid request", "text/plain; charset=utf-8", "no-store")
    except IndexError:
        return (HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8", "no-store")
    except Exception as e:
        msg = f"Error: {type(e).__name__}: {e}".encode("utf-8")
        return (HTTPStatus.INTERNAL_SERVER_ERROR, msg, "text/plain; charset=utf-8", "no-store")


def _make_handler(app: PreviewApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "panelizer-preview"

        def log_message(self, fmt: str, *args) -> None:
            # Avoid noisy default logs (and avoid printing local file paths).
            return

        def _send(self, status: int, body: bytes, content_type: str, *, cache: str = "no-store") -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The client (browser) navigated away / aborted the request while we were responding.
                return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, body, content_type, cache = dispatch_request(app, parsed.path, parse_qs(parsed.query or ""))
            return self._send(status, body, content_type, cache=cache)

    return Handler


def create_preview_server(config: PreviewConfig) -> Tuple[ThreadingHTTPServer, str]:
    app = PreviewApp(config)
    handler = _make_handler(app)

    httpd = ThreadingHTTPServer((config.host, config.port), handler)
    host, port = httpd.server_address[:2]
    url = f"http://{host}:{port}/"
    return httpd, url
