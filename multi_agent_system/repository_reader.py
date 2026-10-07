"""Read a bounded amount of source text from a local repository."""

from pathlib import Path

from multi_agent_system.langgraph_workflow import AgentState

ALLOWED_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "node_modules",
}
IGNORED_FILENAMES = {"tempCodeRunnerFile.py"}
MAX_INDEXED_FILES = 200
MAX_SELECTED_FILES = 8
MAX_CHARS_PER_FILE = 3_000
MAX_TOTAL_CHARS = 20_000


def bounded_file_snippet(content: str, character_limit: int) -> tuple[str, bool]:
    """Truncate at a complete line so the model never sees invented fragments."""
    if len(content) <= character_limit:
        return content, False

    prefix = content[:character_limit]
    last_newline = prefix.rfind("\n")
    if last_newline == -1:
        return "", True
    return prefix[: last_newline + 1], True


def should_read_file(path: Path, root: Path) -> bool:
    """Return True only for supported, non-secret repository files."""
    relative_path = path.relative_to(root)

    if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
        return False

    if path.name in IGNORED_FILENAMES:
        return False

    if path.name == ".env" or path.name.startswith(".env."):
        return False

    return path.suffix.lower() in ALLOWED_SUFFIXES


def _repository_candidates(root: Path) -> list[Path]:
    """Return a sorted, bounded list of allowed repository files."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and should_read_file(path, root)
    )[:MAX_INDEXED_FILES]


def index_repository(state: AgentState) -> AgentState:
    """List safe repository paths without reading their contents."""
    root = Path(state.get("repo_path", ".")).expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")

    repository_files = [
        str(path.relative_to(root)) for path in _repository_candidates(root)
    ]
    return {
        "repo_path": str(root),
        "repository_files": repository_files,
        "execution_log": ["repository_indexer"],
    }


def read_selected_files(
    state: AgentState, selected_files: list[str]
) -> AgentState:
    """Read only selected paths that appeared in the safe repository index."""
    root = Path(state["repo_path"]).resolve()
    allowed_paths = {
        str(path.relative_to(root)): path for path in _repository_candidates(root)
    }

    approved_files: list[str] = []
    selected_file_contents: dict[str, str] = {}
    context_sections: list[str] = []
    remaining_characters = MAX_TOTAL_CHARS

    for relative_path in selected_files[:MAX_SELECTED_FILES]:
        if remaining_characters <= 0:
            break

        path = allowed_paths.get(relative_path)
        if path is None:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        character_limit = min(MAX_CHARS_PER_FILE, remaining_characters)
        snippet, truncated = bounded_file_snippet(content, character_limit)
        section_title = f"### {relative_path}"
        if truncated:
            section_title += " (truncated at a complete line)"

        approved_files.append(relative_path)
        selected_file_contents[relative_path] = snippet
        context_sections.append(f"{section_title}\n{snippet}")
        remaining_characters -= len(snippet)

    code_context = "\n\n".join(context_sections)
    if not code_context:
        code_context = "No supported source files were found."

    return {
        "selected_files": approved_files,
        "selected_file_contents": selected_file_contents,
        "code_context": code_context,
    }


def read_repository(state: AgentState) -> AgentState:
    """Index a repository and read an initial bounded set of safe files."""
    index_updates = index_repository(state)
    indexed_state: AgentState = {**state, **index_updates}
    read_updates = read_selected_files(
        indexed_state, index_updates["repository_files"]
    )
    return {
        **index_updates,
        **read_updates,
        "execution_log": ["code_reader"],
    }
