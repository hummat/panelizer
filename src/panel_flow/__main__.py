import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Set, Tuple

import click
from PIL import ImageDraw

from .cv.detector import CVDetector
from .extraction.extractor import Extractor
from .extraction.utils import calculate_book_hash
from .ordering import order_panels
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
    """Panel Flow: Pragmatic comic panel detection."""
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
def process(file: Path, output: Optional[Path], direction: str, pages: Tuple[str, ...]):
    """
    Process a comic file and generate panel metadata.

    Supported formats: CBZ, ZIP, PDF, JPG, JPEG, PNG, WEBP, BMP
    """
    if not output:
        output = file.with_suffix(".panels.json")

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
        panels, confidence = detector.detect(img)

        # Order panels
        bboxes = [p.bbox for p in panels]
        ordered_indices = order_panels(bboxes, reading_dir)

        # Build page model
        page = Page(
            index=i,
            size=(img.width, img.height),
            panels=panels,
            order=[panels[idx].id for idx in ordered_indices],
            order_confidence=0.9,  # Heuristic confidence
            source=DetectionSource.CV,
        )
        pages_data.append(page)
        click.echo(f" found {len(panels)} panels (conf: {confidence:.2f})")

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
        output_dir = Path(tempfile.mkdtemp(prefix="panel-flow-viz-"))

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
        for panel in page_data["panels"]:
            x, y, w, h = panel["bbox"]
            # Draw bbox in red
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
            # Draw panel ID
            draw.text((x + 5, y + 5), panel["id"], fill=(255, 0, 0))

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


if __name__ == "__main__":
    main()
