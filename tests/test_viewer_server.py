import tempfile
import types
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from panelizer.preview.server import PreviewApp, PreviewConfig, _make_handler, _static_bytes, dispatch_request
from panelizer.schema import ReadingDirection


class TestStaticAssets:
    @pytest.mark.parametrize(
        "path",
        ["/", "/index.html", "/styles.css", "/app.js"],
    )
    def test_static_bytes(self, path: str) -> None:
        data, content_type = _static_bytes(path)
        assert isinstance(data, (bytes, bytearray))
        assert len(data) > 0
        assert isinstance(content_type, str)
        assert content_type

    def test_static_bytes_rejects_unknown(self) -> None:
        with pytest.raises(FileNotFoundError):
            _static_bytes("/nope.js")

    def test_app_js_supports_overlay_in_panel_view(self) -> None:
        data, _content_type = _static_bytes("/app.js")
        text = data.decode("utf-8")
        assert 'state.mode !== "page"' not in text
        assert 'state.mode === "panel"' in text

    def test_app_js_wraps_panels_across_pages(self) -> None:
        data, _content_type = _static_bytes("/app.js")
        text = data.decode("utf-8")
        assert "async function nextPanel()" in text
        assert "async function prevPanel()" in text
        assert "await nextPage()" in text
        assert "await prevPage()" in text

    def test_app_js_includes_panel_conf_in_status(self) -> None:
        data, _content_type = _static_bytes("/app.js")
        text = data.decode("utf-8")
        assert "Panel conf" in text


class TestViewerApp:
    def test_book_info_and_page_endpoints_single_image(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            img = Image.new("RGB", (320, 240), color=(255, 255, 255))
            img.save(img_path, format="PNG")

            config = PreviewConfig(file_path=img_path, reading_direction=ReadingDirection.LTR)
            app = PreviewApp(config)

            info = app.book_info()
            assert info["file_name"] == img_path.name
            assert info["page_count"] == 1
            assert info["reading_direction"] == "ltr"
            assert info["book_hash"].startswith("sha256:")

            png0 = app.page_png(0)
            assert isinstance(png0, (bytes, bytearray))
            assert len(png0) > 10

            page0 = app.page_json(0)
            assert page0["index"] == 0
            assert page0["size"] == [320, 240]
            assert page0["source"] == "cv"
            assert 0.0 <= page0["cv_confidence"] <= 1.0
            assert isinstance(page0["panels"], list)
        finally:
            img_path.unlink()

    def test_page_png_cache(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            img = Image.new("RGB", (64, 48), color=(255, 255, 255))
            img.save(img_path, format="PNG")

            app = PreviewApp(PreviewConfig(file_path=img_path, reading_direction=ReadingDirection.LTR))
            a = app.page_png(0)
            b = app.page_png(0)
            assert a == b

            c = app.page_png(0, refresh=True)
            assert isinstance(c, (bytes, bytearray))
        finally:
            img_path.unlink()


class TestDispatchRequest:
    def test_dispatch_routes(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            Image.new("RGB", (80, 60), color=(255, 255, 255)).save(img_path, format="PNG")
            app = PreviewApp(PreviewConfig(file_path=img_path, reading_direction=ReadingDirection.LTR))

            status, body, ct, _cache = dispatch_request(app, "/api/book", {})
            assert status == 200
            assert ct.startswith("application/json")
            assert b"page_count" in body

            status, body, ct, _cache = dispatch_request(app, "/api/page/0.json", {})
            assert status == 200
            assert ct.startswith("application/json")
            assert b"cv_confidence" in body

            status, body, ct, _cache = dispatch_request(app, "/api/page/0.png", {})
            assert status == 200
            assert ct == "image/png"
            assert len(body) > 10

            status, _body, _ct, _cache = dispatch_request(app, "/api/page/nope.json", {})
            assert status == 400

            status, _body, _ct, _cache = dispatch_request(app, "/nope", {})
            assert status == 404

            status, body, ct, _cache = dispatch_request(app, "/styles.css", {})
            assert status == 200
            assert isinstance(body, (bytes, bytearray))
            assert ct
        finally:
            img_path.unlink()


class TestHandlerSend:
    @pytest.mark.parametrize("exc_type", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
    def test_send_ignores_client_disconnect(self, exc_type: type[BaseException]) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img_path = Path(f.name)

        try:
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(img_path, format="PNG")
            app = PreviewApp(PreviewConfig(file_path=img_path, reading_direction=ReadingDirection.LTR))
            handler_cls = _make_handler(app)

            class FakeWFile:
                def write(self, _body: bytes) -> None:
                    raise exc_type()

            fake = types.SimpleNamespace(
                send_response=lambda _status: None,
                send_header=lambda *_args, **_kwargs: None,
                end_headers=lambda: None,
                wfile=FakeWFile(),
            )

            handler_cls._send(cast(Any, fake), 200, b"hello", "text/plain; charset=utf-8")
        finally:
            img_path.unlink()
