import io
import json
import tempfile
import zipfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from panel_flow.__main__ import cli


class TestCLI:
    def test_cli_group_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Panel Flow" in result.output

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

            result = runner.invoke(
                cli, ["process", str(cbz_path), "-o", str(output_path), "-d", "rtl"]
            )

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
