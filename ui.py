"""
ui.py

Terminal UI for the pipeline: a live, multi-step progress display built on
``rich``. Core modules stay UI-agnostic -- they call a small set of reporter
methods and receive a no-op reporter by default. Importing this module does not
import ``rich``, so the pipeline still runs (silently) if ``rich`` is missing.

Reporter API used by the pipeline:

    with reporter:                       # start/stop the live display
        reporter.step(key, text, total)  # begin a step (total=None -> spinner)
        reporter.describe(key, text)     # change a running step's label
        reporter.set_total(key, n)       # set the total once it becomes known
        reporter.advance(key, n=1)       # advance a determinate step
        reporter.done(key)               # mark a step complete (shows a check)
    reporter.header(title, subtitle)     # framed banner
    reporter.note / warn / success(text) # one-off messages
    reporter.summary(title, items)       # closing table of label -> value
"""

from __future__ import annotations


class NullReporter:
    """No-op reporter. The safe default so core code can report unconditionally."""

    # Lifecycle -------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # Framing / messages ----------------------------------------------------
    def header(self, title, subtitle=None):
        pass

    def note(self, text):
        pass

    def warn(self, text):
        pass

    def success(self, text):
        pass

    def summary(self, title, items):
        pass

    # Steps -----------------------------------------------------------------
    def step(self, key, description, total=None):
        pass

    def describe(self, key, description):
        pass

    def set_total(self, key, total):
        pass

    def advance(self, key, n=1):
        pass

    def done(self, key):
        pass


NULL_REPORTER = NullReporter()


class _PlainReporter(NullReporter):
    """Fallback when ``rich`` isn't installed: plain, non-interactive prints."""

    def header(self, title, subtitle=None):
        print("\n" + "=" * 60)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * 60 + "\n")

    def note(self, text):
        print(text)

    def warn(self, text):
        print(f"! {text}")

    def success(self, text):
        print(f"[done] {text}")

    def summary(self, title, items):
        print("\n" + "=" * 60)
        print(title)
        for label, value in items:
            print(f"  {label}: {value}")
        print("=" * 60)

    def step(self, key, description, total=None):
        print(f"-> {description}")

    def describe(self, key, description):
        print(f"   {description}")


class _RichReporter(NullReporter):
    """Live, multi-step progress display backed by ``rich``."""

    def __init__(self, console, progress):
        self._console = console
        self._progress = progress
        self._tasks: dict[str, int] = {}

    def __enter__(self):
        self._progress.start()
        return self

    def __exit__(self, *exc):
        # Never leave a step mid-spin if an exception unwound the pipeline.
        for key in list(self._tasks):
            self.done(key)
        self._progress.stop()
        return False

    def header(self, title, subtitle=None):
        from rich.panel import Panel
        from rich.text import Text

        body = Text(title, style="bold cyan")
        if subtitle:
            body.append("\n")
            body.append(subtitle, style="dim")
        self._console.print(Panel(body, border_style="cyan", expand=False, padding=(0, 2)))

    def note(self, text):
        self._console.print(f"[dim]{text}[/dim]")

    def warn(self, text):
        self._console.print(f"[yellow]![/yellow] {text}")

    def success(self, text):
        self._console.print(f"[green]✓[/green] {text}")

    def summary(self, title, items):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        for label, value in items:
            table.add_row(label, str(value))
        self._console.print(Panel(table, title=title, border_style="green", expand=False, padding=(0, 2)))

    def step(self, key, description, total=None):
        if key in self._tasks:
            self._progress.update(self._tasks[key], description=description, total=total)
        else:
            self._tasks[key] = self._progress.add_task(description, total=total)

    def describe(self, key, description):
        if key in self._tasks:
            self._progress.update(self._tasks[key], description=description)

    def set_total(self, key, total):
        if key in self._tasks:
            self._progress.update(self._tasks[key], total=total)

    def advance(self, key, n=1):
        if key in self._tasks:
            self._progress.advance(self._tasks[key], n)

    def done(self, key):
        tid = self._tasks.get(key)
        if tid is None:
            return
        task = next((t for t in self._progress.tasks if t.id == tid), None)
        total = task.total if (task and task.total) else None
        if total is None:
            # Indeterminate step: give it a unit total so it renders as finished.
            self._progress.update(tid, total=1, completed=1)
        else:
            self._progress.update(tid, completed=total)


def create_reporter(force_plain: bool = False):
    """Return a live reporter, degrading gracefully if ``rich`` is unavailable."""
    if force_plain:
        return _PlainReporter()
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:
        return _PlainReporter()

    console = Console()
    progress = Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    return _RichReporter(console, progress)
