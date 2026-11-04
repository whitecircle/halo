"""What runs in a sandbox: each language's source name, build and launch commands, and comment syntax, and the
text every backend hands a program."""

from dataclasses import dataclass, field

INTERPRETER_PLACEHOLDER = "$INTERPRETER"


@dataclass(frozen=True)
class LanguageSpec:
    """``compile_argv`` is ``None`` for interpreted languages. :data:`INTERPRETER_PLACEHOLDER` in ``run_argv`` is
    replaced with the backend's interpreter."""

    name: str
    source_name: str
    run_argv: tuple[str, ...]
    compile_argv: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    line_comment: str = "#"
    block_comment: tuple[str, str] | None = None

    @property
    def is_compiled(self) -> bool:
        return self.compile_argv is not None


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        source_name="main.py",
        # -s -E (not -I) drops user site-packages / PYTHON* env but keeps the script's dir on sys.path
        # so a session can import workspace modules written on earlier turns.
        run_argv=(INTERPRETER_PLACEHOLDER, "-s", "-E", "main.py"),
        aliases=("py", "python3"),
    ),
    "bash": LanguageSpec(
        name="bash",
        source_name="main.sh",
        # Run by name off the child's PATH: unlike Python there is no toolkit interpreter to pin to.
        run_argv=("bash", "main.sh"),
        aliases=("sh", "shell"),
    ),
    "cpp": LanguageSpec(
        name="cpp",
        source_name="main.cpp",
        compile_argv=("g++", "-O2", "-pipe", "-std=c++17", "-o", "main", "main.cpp"),
        run_argv=("./main",),
        aliases=("c++", "cxx", "cc"),
        line_comment="//",
        block_comment=("/*", "*/"),
    ),
    "c": LanguageSpec(
        name="c",
        source_name="main.c",
        compile_argv=("gcc", "-O2", "-pipe", "-std=c11", "-o", "main", "main.c"),
        run_argv=("./main",),
        aliases=(),
        line_comment="//",
        block_comment=("/*", "*/"),
    ),
}


def find_language(language: str) -> LanguageSpec | None:
    """By name or alias, case-insensitive."""
    key = (language or "").strip().lower()
    for spec in LANGUAGES.values():
        if key == spec.name or key in spec.aliases:
            return spec
    return None


def require_language(language: str) -> LanguageSpec:
    spec = find_language(language)
    if spec is None:
        raise ValueError(f"unsupported language {language!r}; supported: {', '.join(LANGUAGES)}")
    return spec


def utf8_encodable(text: str) -> str:
    """Replaces what UTF-8 cannot carry (a lone surrogate in model-written code or stdin) with ``?``, so the host never
    fails to hand its input over (a failure the grader books as infra, voiding the episode)."""
    return text.encode("utf-8", errors="replace").decode("utf-8")
