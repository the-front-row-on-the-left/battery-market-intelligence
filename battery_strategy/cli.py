from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from battery_strategy.agents.writer import WriterAgent
from battery_strategy.pipeline import PipelineFactory
from battery_strategy.rag.index_store import build_index, load_index
from battery_strategy.utils.settings import load_manifest, load_runtime_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def embed(config: str = typer.Option(..., help="Path to runtime YAML config.")) -> None:
    """Build the hybrid RAG index (separate step)."""
    runtime_config = load_runtime_config(config)
    manifest = load_manifest(runtime_config.manifest_path)
    index_dir = build_index(manifest.sources, runtime_config)
    bundle = load_index(index_dir)

    table = Table(title="Index Build Complete")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Index Directory", str(index_dir))
    table.add_row("Chunk Count", str(len(bundle.chunks)))
    table.add_row("Source Count", str(len(manifest.sources)))
    console.print(table)


@app.command()
def run(
    config: str = typer.Option(..., help="Path to runtime YAML config."),
    goal: str = typer.Option(..., help="Analysis goal or research question."),
) -> None:
    """Run the Supervisor-based multi-agent analysis pipeline."""
    factory = PipelineFactory.from_config(config)
    final_state = factory.run(goal)

    table = Table(title="Pipeline Finished")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Status", final_state["status"])
    table.add_row("Output Markdown", str(factory.config.output_dir / "final_report.md"))
    table.add_row("Output HTML", str(factory.config.output_dir / "final_report.html"))
    table.add_row("Output PDF", str(factory.config.output_dir / "final_report.pdf"))
    table.add_row("Output State", str(factory.config.output_dir / "final_state.json"))
    table.add_row("Reference File", str(factory.config.output_dir / "references.txt"))
    console.print(table)


@app.command()
def inspect_index(config: str = typer.Option(..., help="Path to runtime YAML config.")) -> None:
    """Inspect the already-built index."""
    runtime_config = load_runtime_config(config)
    bundle = load_index(runtime_config.index_dir)
    table = Table(title="Index Inspection")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Index Dir", str(runtime_config.index_dir))
    table.add_row("Chunks", str(len(bundle.chunks)))
    groups = sorted({chunk["source_group"] for chunk in bundle.chunks})
    table.add_row("Groups", ", ".join(groups))
    console.print(table)


@app.command("html-to-pdf")
def html_to_pdf(
    config: str = typer.Option(..., help="Path to runtime YAML config."),
    html: str | None = typer.Option(None, help="Path to source HTML report."),
    output: str | None = typer.Option(None, help="Path to output PDF."),
) -> None:
    """Render an existing HTML report to PDF, preserving embedded CSS."""
    runtime_config = load_runtime_config(config)
    output_dir = Path(runtime_config.output_dir)
    html_path = Path(html) if html else output_dir / "final_report.html"
    latest_pdf_path = Path(output) if output else output_dir / "final_report.pdf"
    archived_pdf_path = latest_pdf_path.with_name(
        f"{latest_pdf_path.stem}_{WriterAgent._report_file_timestamp()}{latest_pdf_path.suffix}"
    )

    WriterAgent.render_pdf_from_html(html_path, latest_pdf_path)
    if latest_pdf_path.exists():
        archived_pdf_path.write_bytes(latest_pdf_path.read_bytes())

    table = Table(title="HTML to PDF Complete")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Source HTML", str(html_path))
    table.add_row("Latest PDF", str(latest_pdf_path))
    table.add_row("Archived PDF", str(archived_pdf_path))
    console.print(table)


if __name__ == "__main__":
    app()
