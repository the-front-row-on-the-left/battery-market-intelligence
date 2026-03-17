from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from battery_strategy.index_store import build_index, load_index
from battery_strategy.pipeline import PipelineFactory
from battery_strategy.settings import load_manifest, load_runtime_config

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


if __name__ == "__main__":
    app()
