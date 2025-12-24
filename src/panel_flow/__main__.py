from pathlib import Path

import click

from .cv.detector import CVDetector
from .extraction.extractor import Extractor
from .extraction.utils import calculate_book_hash
from .ordering import order_panels
from .schema import BookData, BookMetadata, DetectionSource, Page, ReadingDirection


@click.group()
def cli():
    """Panel Flow: Pragmatic comic panel detection."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
@click.option(
    "--direction", "-d", type=click.Choice(["ltr", "rtl"]), default="ltr", help="Reading direction"
)
def process(file: Path, output: Path, direction: str):
    """Process a comic book (CBZ/PDF) and generate panel metadata."""
    if not output:
        output = file.with_suffix(".panels.json")

    click.echo(f"Processing {file}...")

    extractor = Extractor(file)
    detector = CVDetector()
    reading_dir = ReadingDirection(direction)

    pages_data = []

    # Process pages
    for i, img in extractor.iter_pages():
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

    # Create book model
    book = BookData(
        book_hash=calculate_book_hash(file),
        pages=pages_data,
        metadata=BookMetadata(reading_direction=reading_dir, tool_version="0.1.0"),
    )

    with open(output, "w") as f:
        f.write(book.model_dump_json(indent=2))

    click.echo(f"\nDone! Saved to {output}")


def main():
    cli()


if __name__ == "__main__":
    main()
