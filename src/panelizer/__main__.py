import json
import subprocess
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional, Set, Tuple

import click
from PIL import ImageDraw

from .cv.debug import DebugContext
from .cv.detector import CVDetector
from .extraction.extractor import Extractor
from .extraction.utils import calculate_book_hash
from .ordering import order_panels
from .preview.server import PreviewConfig, create_preview_server
from .schema import BookData, BookMetadata, DetectionSource, Page, ReadingDirection


def parse_pages_specs(specs: Tuple[str, ...]) -> Optional[Set[int]]:
    """
    Parse 1-based, inclusive page selectors into 0-based page indices.

    Examples:
      --pages 1-5
      --pages 17-23
      --pages 1-5,17-23
      --pages 3
    """
    if not specs:
        return None

    indices: Set[int] = set()
    for spec in specs:
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue

            if "-" in token:
                parts = [p.strip() for p in token.split("-", 1)]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise click.BadParameter(f"Invalid pages spec: {token!r}")
                try:
                    start_1b = int(parts[0])
                    end_1b = int(parts[1])
                except ValueError as e:
                    raise click.BadParameter(f"Invalid pages spec: {token!r}") from e
            else:
                try:
                    start_1b = int(token)
                except ValueError as e:
                    raise click.BadParameter(f"Invalid pages spec: {token!r}") from e
                end_1b = start_1b

            if start_1b <= 0 or end_1b <= 0:
                raise click.BadParameter(f"Pages must be >= 1 (got {token!r})")
            if end_1b < start_1b:
                raise click.BadParameter(f"Invalid range {token!r} (end < start)")

            for page_1b in range(start_1b, end_1b + 1):
                indices.add(page_1b - 1)

    return indices


@click.group()
def cli():
    """Panelizer: Pragmatic comic panel detection."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--direction", "-d", type=click.Choice(["ltr", "rtl"]), default="ltr", help="Reading direction")
@click.option(
    "--pages",
    multiple=True,
    help="Pages to process (1-based, inclusive). Examples: --pages 1-5 or --pages 17-23 or --pages 1-5,17-23",
)
@click.option("--debug", is_flag=True, help="Output debug images showing each detection step")
@click.option(
    "--debug-dir", type=click.Path(path_type=Path), help="Directory for debug output (default: <file>.debug/)"
)
def process(
    file: Path,
    output: Optional[Path],
    direction: str,
    pages: Tuple[str, ...],
    debug: bool,
    debug_dir: Optional[Path],
):
    """
    Process a comic file and generate panel metadata.

    Supported formats: CBZ, ZIP, PDF, JPG, JPEG, PNG, WEBP, BMP
    """
    if not output:
        output = file.with_suffix(".panels.json")

    # Set up debug context
    if debug:
        if not debug_dir:
            debug_dir = file.with_suffix(".debug")
        debug_ctx = DebugContext(enabled=True, output_dir=debug_dir)
        click.echo(f"Debug output will be saved to {debug_dir}/")
    else:
        debug_ctx = None

    click.echo(f"Processing {file}...")

    extractor = Extractor(file)
    detector = CVDetector()
    reading_dir = ReadingDirection(direction)
    selected_pages = parse_pages_specs(pages)

    pages_data = []
    matched = 0

    # Process pages
    for i, img in extractor.iter_pages():
        if selected_pages is not None and i not in selected_pages:
            continue

        matched += 1
        click.echo(f"  Page {i}...", nl=False)

        # Create per-page debug context if debugging
        page_debug = None
        if debug_ctx and debug_dir:
            page_debug = DebugContext(enabled=True, output_dir=debug_dir / f"page-{i:04d}")

        result = detector.detect(img, debug=page_debug)

        # Save debug HTML for this page
        if page_debug:
            html_path = page_debug.save_html()
            if html_path:
                click.echo(f" (debug: {html_path})", nl=False)

        # Order panels
        bboxes = [p.bbox for p in result.panels]
        ordered_indices = order_panels(bboxes, reading_dir)

        # Build page model
        page = Page(
            index=i,
            size=(img.width, img.height),
            panels=result.panels,
            order=[result.panels[idx].id for idx in ordered_indices],
            order_confidence=0.9,  # Heuristic confidence
            source=DetectionSource.CV,
            gutters=result.gutters,
            processing_time=result.processing_time,
        )
        pages_data.append(page)
        click.echo(f" found {len(result.panels)} panels (conf: {result.confidence:.2f})")

    if selected_pages is not None and matched == 0:
        raise click.ClickException(f"No pages matched --pages={','.join(pages)!r}")

    # Create book model
    book = BookData(
        book_hash=calculate_book_hash(file),
        pages=pages_data,
        metadata=BookMetadata(reading_direction=reading_dir, tool_version="0.1.0"),
    )

    with open(output, "w") as f:
        f.write(book.model_dump_json(indent=2))

    click.echo(f"\nDone! Saved to {output}")


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory (default: temp)")
@click.option(
    "--pages",
    multiple=True,
    help="Pages to visualize (1-based, inclusive). Examples: --pages 1-5 or --pages 17-23 or --pages 1-5,17-23",
)
def visualize(file: Path, json_file: Path, output_dir: Optional[Path], pages: Tuple[str, ...]):
    """
    Visualize detected panels overlaid on comic pages.

    Opens the first page in your default image viewer. Use arrow keys to navigate.
    """
    # Load panel data
    with open(json_file) as f:
        data = json.load(f)

    # Create output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="panelizer-viz-"))

    click.echo(f"Rendering pages to {output_dir}...")

    extractor = Extractor(file)
    output_files = []
    selected_pages = parse_pages_specs(pages)
    matched = 0

    # Process each page
    for page_idx, img in extractor.iter_pages():
        if selected_pages is not None and page_idx not in selected_pages:
            continue

        matched += 1
        # Find corresponding page data
        page_data = next((p for p in data["pages"] if p["index"] == page_idx), None)
        if not page_data:
            click.echo(f"  Page {page_idx}: No panel data found, skipping")
            continue

        # Draw bboxes
        draw = ImageDraw.Draw(img)
        border_w = max(4, min(12, round(min(img.width, img.height) * 0.006)))
        order = page_data.get("order") or []
        id_to_num = {pid: i + 1 for i, pid in enumerate(order)}
        for i, panel in enumerate(page_data["panels"]):
            x, y, w, h = panel["bbox"]
            # Draw bbox in red
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=border_w)

            panel_id = panel.get("id", "?")
            panel_num = id_to_num.get(panel_id, i + 1)
            conf = panel.get("confidence")
            conf_text = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "?"

            label = f"#{panel_num} cv:{conf_text}"
            draw.text(
                (x + border_w + 2, y + border_w + 2),
                label,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        # Save
        output_file = output_dir / f"page_{page_idx:04d}.png"
        img.save(output_file)
        output_files.append(output_file)
        click.echo(f"  Page {page_idx}: {len(page_data['panels'])} panels")

    if not output_files:
        if selected_pages is not None and matched == 0:
            raise click.ClickException(f"No pages matched --pages={','.join(pages)!r}")
        click.echo("No pages rendered!")
        return

    click.echo(f"\nRendered {len(output_files)} pages")

    # Open first image in default viewer
    first_image = output_files[0]
    click.echo(f"Opening {first_image} in default viewer...")

    try:
        subprocess.run(["xdg-open", str(first_image)], check=True)
    except FileNotFoundError:
        click.echo("Could not open viewer (xdg-open not found)")
        click.echo(f"View images manually in: {output_dir}")
    except subprocess.CalledProcessError:
        click.echo(f"Could not open viewer. View images manually in: {output_dir}")


def main():
    cli()


def _run_preview(
    file: Path,
    direction: str,
    host: str,
    port: int,
    open_browser: bool,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> None:
    config = PreviewConfig(
        file_path=file,
        reading_direction=ReadingDirection(direction),
        host=host,
        port=port,
        debug=debug,
        debug_dir=debug_dir,
    )
    httpd, url = create_preview_server(config)

    click.echo(f"Preview running at {url}")
    click.echo("Press Ctrl+C to stop.")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


@cli.command(name="preview")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--direction", "-d", type=click.Choice(["ltr", "rtl"]), default="ltr", help="Reading direction")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address (use 127.0.0.1 for local only)")
@click.option("--port", default=0, show_default=True, type=int, help="Port (0 chooses a free port)")
@click.option("--open/--no-open", "open_browser", default=True, show_default=True, help="Open browser automatically")
@click.option("--debug", is_flag=True, help="Output debug images and info for each page")
@click.option(
    "--debug-dir", type=click.Path(path_type=Path), help="Directory for debug output (default: <file>.debug/)"
)
def preview(
    file: Path, direction: str, host: str, port: int, open_browser: bool, debug: bool, debug_dir: Optional[Path]
) -> None:
    """Run a local web preview tool to inspect CV detection results."""
    if debug:
        actual_debug_dir = debug_dir or Path(f"{file}.debug")
        click.echo(f"Debug mode enabled. Output will be saved to {actual_debug_dir}/")
    _run_preview(
        file=file,
        direction=direction,
        host=host,
        port=port,
        open_browser=open_browser,
        debug=debug,
        debug_dir=debug_dir,
    )


@cli.command(name="viewer", hidden=True)
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--direction", "-d", type=click.Choice(["ltr", "rtl"]), default="ltr", help="Reading direction")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address (use 127.0.0.1 for local only)")
@click.option("--port", default=0, show_default=True, type=int, help="Port (0 chooses a free port)")
@click.option("--open/--no-open", "open_browser", default=True, show_default=True, help="Open browser automatically")
def viewer_alias(file: Path, direction: str, host: str, port: int, open_browser: bool) -> None:
    """Deprecated alias for `preview`."""
    click.echo("Deprecated: use `panelizer preview`.")
    _run_preview(file=file, direction=direction, host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    main()
