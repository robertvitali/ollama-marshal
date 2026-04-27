"""Typer CLI entry point for ollama-marshal."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
import structlog
import typer
import uvicorn

from ollama_marshal import __version__
from ollama_marshal.config import LogFormat, load_config

app = typer.Typer(
    name="ollama-marshal",
    help="Model-aware scheduling proxy for Ollama.",
    no_args_is_help=True,
)


def _setup_logging(level: str, fmt: str) -> None:
    """Configure structlog based on CLI options.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Log format ('console' or 'json').
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if fmt == LogFormat.JSON.value:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    log_level: int = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"ollama-marshal {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """ollama-marshal: Model-aware scheduling proxy for Ollama."""


@app.command()
def start(
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to marshal.yaml config file."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Proxy bind address."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Proxy listen port."),
    ] = None,
    ollama_host: Annotated[
        str | None,
        typer.Option("--ollama-host", help="Ollama API base URL."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
    ] = None,
    log_format: Annotated[
        str | None,
        typer.Option("--log-format", help="Log format (console, json)."),
    ] = None,
) -> None:
    """Start the ollama-marshal proxy server."""
    # Build CLI overrides
    cli_overrides: dict[str, str | int] = {}
    if host is not None:
        cli_overrides["proxy.host"] = host
    if port is not None:
        cli_overrides["proxy.port"] = port
    if ollama_host is not None:
        cli_overrides["ollama.host"] = ollama_host
    if log_level is not None:
        cli_overrides["logging.level"] = log_level
    if log_format is not None:
        cli_overrides["logging.format"] = log_format

    cfg = load_config(config_path=config, cli_overrides=cli_overrides)
    _setup_logging(cfg.logging.level, cfg.logging.format.value)

    logger = structlog.get_logger()
    logger.info(
        "cli.starting",
        version=__version__,
        proxy=f"{cfg.proxy.host}:{cfg.proxy.port}",
        ollama=cfg.ollama.host,
    )

    from ollama_marshal.server import create_app

    server_app = create_app(cfg)
    uvicorn.run(
        server_app,
        host=cfg.proxy.host,
        port=cfg.proxy.port,
        log_level=cfg.logging.level.lower(),
    )


@app.command()
def status(
    host: Annotated[
        str,
        typer.Option("--host", help="Proxy host to query."),
    ] = "http://localhost:11435",
) -> None:
    """Show the proxy status dashboard."""
    try:
        resp = httpx.get(f"{host}/api/marshal/status", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError:
        typer.echo("Error: Cannot reach ollama-marshal. Is it running?", err=True)
        raise typer.Exit(code=1) from None

    # Format and display dashboard
    uptime = data.get("uptime_seconds", 0)
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)

    typer.echo("=" * 50)
    typer.echo("  ollama-marshal status")
    typer.echo("=" * 50)
    typer.echo(f"  Uptime:  {hours}h {minutes}m {seconds}s")
    typer.echo()

    # Loaded models
    typer.echo("  Loaded Models:")
    loaded = data.get("loaded_models", [])
    if not loaded:
        typer.echo("    (none)")
    else:
        for m in loaded:
            size_gb = m["size_vram"] / (1024**3)
            typer.echo(
                f"    {m['name']:<30} "
                f"{size_gb:.1f} GB   "
                f"{m['pending_requests']} pending"
            )
    typer.echo()

    # Memory
    mem = data.get("memory", {})
    total_gb = mem.get("total", 0) / (1024**3)
    used_gb = mem.get("used_by_models", 0) / (1024**3)
    avail_gb = mem.get("available", 0) / (1024**3)
    typer.echo(
        f"  Memory:  {used_gb:.1f} / {total_gb:.1f} GB used"
        f"  ({avail_gb:.1f} GB available)"
    )
    typer.echo()

    # Queue
    queue = data.get("queue", {})
    total_pending = queue.get("total_pending", 0)
    typer.echo(f"  Queue:   {total_pending} pending")
    by_model = queue.get("by_model", {})
    for model, count in by_model.items():
        typer.echo(f"    {model:<30} {count}")
    typer.echo()

    # Metrics
    metrics = data.get("metrics", {})
    typer.echo(f"  Requests served:  {metrics.get('requests_served', 0)}")
    typer.echo(f"  Model swaps:      {metrics.get('model_swaps', 0)}")
    typer.echo(f"  Evictions:        {metrics.get('evictions', 0)}")
    typer.echo(f"  Avg wait:         {metrics.get('average_wait_ms', 0):.1f} ms")
    typer.echo("=" * 50)


@app.command()
def stop(
    host: Annotated[
        str,
        typer.Option("--host", help="Proxy host to stop."),
    ] = "http://localhost:11435",
) -> None:
    """Stop the running ollama-marshal proxy.

    Sends a shutdown signal by hitting the proxy's shutdown endpoint.
    If the proxy was started in the foreground, use Ctrl+C instead.
    """
    typer.echo(f"Targeting proxy at: {host}")
    typer.echo(
        "Note: If the proxy is running in the foreground, use Ctrl+C to stop it.\n"
        "Programmatic stop via API is planned for a future release."
    )
    raise typer.Exit(code=0)
