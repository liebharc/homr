import os


def get_run_id() -> str:
    """
    Build a reproducible run identifier from the current git commit.

    Returns:
        String containing the commit count and full HEAD hash.
    """
    git_count = os.popen("git rev-list --count HEAD").read().strip()  # noqa: S605, S607
    git_head = os.popen("git rev-parse HEAD").read().strip()  # noqa: S605, S607
    return f"{git_count}-{git_head}"
