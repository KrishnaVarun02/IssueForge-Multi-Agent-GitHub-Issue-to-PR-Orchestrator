"""Open a local SQLite checkpointer for durable LangGraph state."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def open_sqlite_checkpointer(
    database_path: str | Path,
) -> Iterator[SqliteSaver]:
    """Create the parent directory, open SQLite, and close it afterward."""
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer
