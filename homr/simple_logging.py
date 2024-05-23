import sys
from typing import Any


def eprint(*args: Any, **kwargs: Any) -> None:
    """
    A logger with differnt log levels felt overkill for this project.
    So we just have one logger that logs to stderr.
    """
    print(*args, file=sys.stderr, **kwargs)  # noqa: T201
