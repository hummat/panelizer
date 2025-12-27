import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from PIL import Image

from panelizer.__main__ import cli, parse_pages_specs


class TestCLI:
    def test_cli_group_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Panelizer" in result.output

    def test_process_command_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--direction" in result.output

    def test_process_cbz(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CBZ
            cbz_path = Path(tmpdir) / "test.cbz"
            output_path = Path(tmpdir) / "test.panels.json"

            with zipfile.ZipFile(cbz_path, "w") as z:
                # Create simple test images
                for i in range(2):
                    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    z.writestr(f"page{i:03d}.png", buf.getvalue())

            result = runner.invoke(cli, ["process", str(cbz_path), "-o", str(output_path)])

            assert result.exit_code == 0
            assert "Processing" in result.output
            assert "Done!" in result.output
            assert output_path.exists()

            # Validate JSON output
            with open(output_path) as f:
                data = json.load(f)
            assert "book_hash" in data
            assert "pages" in data
            assert len(data["pages"]) == 2

    def test_process_pages_subset(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            cbz_path = Path(tmpdir) / "test.cbz"
            output_path = Path(tmpdir) / "test.panels.json"

            with zipfile.ZipFile(cbz_path, "w") as z:
                for i in range(10):
                    img = Image.new("RGB", (200, 150), color=(255, 255, 255))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    z.writestr(f"page{i:03d}.png", buf.getvalue())

            result = runner.invoke(cli, ["process", str(cbz_path), "-o", str(output_path), "--pages", "1-5"])
            assert result.exit_code == 0

            with open(output_path) as f:
                data = json.load(f)
            assert len(data["pages"]) == 5
            assert [p["index"] for p in data["pages"]] == [0, 1, 2, 3, 4]

    def test_process_rtl_direction(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            cbz_path = Path(tmpdir) / "manga.cbz"
            output_path = Path(tmpdir) / "manga.panels.json"

            with zipfile.ZipFile(cbz_path, "w") as z:
                img = Image.new("RGB", (400, 300), color=(255, 255, 255))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr("page001.png", buf.getvalue())

            result = runner.invoke(cli, ["process", str(cbz_path), "-o", str(output_path), "-d", "rtl"])

            assert result.exit_code == 0

            with open(output_path) as f:
                data = json.load(f)
            assert data["metadata"]["reading_direction"] == "rtl"

    def test_process_default_output_path(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            cbz_path = Path(tmpdir) / "comic.cbz"
            expected_output = Path(tmpdir) / "comic.panels.json"

            with zipfile.ZipFile(cbz_path, "w") as z:
                img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr("page.png", buf.getvalue())

            result = runner.invoke(cli, ["process", str(cbz_path)])

            assert result.exit_code == 0
            assert expected_output.exists()

    def test_visualize_command(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CBZ
            cbz_path = Path(tmpdir) / "test.cbz"
            json_path = Path(tmpdir) / "test.panels.json"
            viz_dir = Path(tmpdir) / "viz"

            # Create CBZ with one page
            with zipfile.ZipFile(cbz_path, "w") as z:
                img = Image.new("RGB", (400, 300), color=(255, 255, 255))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                z.writestr("page000.png", buf.getvalue())

            # Create matching JSON
            json_data = {
                "version": 1,
                "book_hash": "sha256:abc123",
                "pages": [
                    {
                        "index": 0,
                        "size": [400, 300],
                        "panels": [
                            {"id": "p-0", "bbox": [10, 10, 100, 100], "confidence": 0.9},
                            {"id": "p-1", "bbox": [200, 10, 100, 100], "confidence": 0.9},
                        ],
                        "order": ["p-0", "p-1"],
                        "order_confidence": 0.9,
                        "source": "cv",
                        "user_override": False,
                    }
                ],
                "overrides": {},
                "metadata": {
                    "reading_direction": "ltr",
                    "created": "2024-12-25T00:00:00",
                    "tool_version": "0.1.0",
                },
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            # Mock subprocess.run to avoid opening actual viewer
            with patch("panelizer.__main__.subprocess.run") as mock_run:
                result = runner.invoke(cli, ["visualize", str(cbz_path), str(json_path), "-o", str(viz_dir)])

                assert result.exit_code == 0
                assert "Rendering pages" in result.output
                assert "Rendered 1 pages" in result.output

                # Check output file was created
                viz_file = viz_dir / "page_0000.png"
                assert viz_file.exists()

                # Check subprocess was called to open viewer
                mock_run.assert_called_once()

    def test_visualize_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "visualize" in result.output.lower()

    def test_visualize_pages_subset(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            cbz_path = Path(tmpdir) / "test.cbz"
            json_path = Path(tmpdir) / "test.panels.json"
            viz_dir = Path(tmpdir) / "viz"

            with zipfile.ZipFile(cbz_path, "w") as z:
                for i in range(3):
                    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    z.writestr(f"page{i:03d}.png", buf.getvalue())

            json_data = {
                "version": 1,
                "book_hash": "sha256:abc123",
                "pages": [
                    {
                        "index": i,
                        "size": [400, 300],
                        "panels": [{"id": "p-0", "bbox": [10, 10, 100, 100], "confidence": 0.9}],
                        "order": ["p-0"],
                        "order_confidence": 0.9,
                        "source": "cv",
                        "user_override": False,
                    }
                    for i in range(3)
                ],
                "overrides": {},
                "metadata": {
                    "reading_direction": "ltr",
                    "created": "2024-12-25T00:00:00",
                    "tool_version": "0.1.0",
                },
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            with patch("panelizer.__main__.subprocess.run") as mock_run:
                result = runner.invoke(
                    cli, ["visualize", str(cbz_path), str(json_path), "-o", str(viz_dir), "--pages", "2-3"]
                )

                assert result.exit_code == 0
                assert (viz_dir / "page_0001.png").exists()
                assert (viz_dir / "page_0002.png").exists()
                assert not (viz_dir / "page_0000.png").exists()
                mock_run.assert_called_once()

    def test_preview_command_starts_and_stops(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "page.png"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path, format="PNG")

            class DummyServer:
                def __init__(self) -> None:
                    self.closed = False

                def serve_forever(self) -> None:
                    raise KeyboardInterrupt()

                def server_close(self) -> None:
                    self.closed = True

            dummy = DummyServer()

            with patch("panelizer.__main__.create_preview_server", return_value=(dummy, "http://127.0.0.1:12345/")):
                with patch("panelizer.__main__.webbrowser.open") as mock_open:
                    result = runner.invoke(cli, ["preview", str(img_path), "--no-open"])

            assert result.exit_code == 0
            assert "Preview running at" in result.output
            assert "http://127.0.0.1:12345/" in result.output
            assert dummy.closed is True
            mock_open.assert_not_called()


class TestParsePagesSpecs:
    def test_empty_is_none(self) -> None:
        assert parse_pages_specs(()) is None

    def test_single_page(self) -> None:
        assert parse_pages_specs(("3",)) == {2}

    def test_range(self) -> None:
        assert parse_pages_specs(("1-3",)) == {0, 1, 2}

    def test_multiple_ranges(self) -> None:
        assert parse_pages_specs(("1-2,5-6",)) == {0, 1, 4, 5}
