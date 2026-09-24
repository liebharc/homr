import os
from homr.simple_logging import eprint

def get_run_id() -> str:
    git_count = os.popen("git rev-list --count HEAD").read().strip()  # noqa: S605, S607
    git_head = os.popen("git rev-parse HEAD").read().strip()  # noqa: S605, S607
    return f"{git_count}-{git_head}"

if __name__ == "__main__":
    eprint(get_run_id())