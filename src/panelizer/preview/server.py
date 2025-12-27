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


@dataclass(frozen=True)
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
        self.cv_detector = CVDetector()
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
        }

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
            debug_dir = self.config.debug_dir or Path(f"{self.config.file_path}.debug")
            page_debug_dir = debug_dir / f"page-{index:04d}"
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
