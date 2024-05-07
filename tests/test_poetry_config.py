import pathlib
import re
from platform import python_version

CURRENT_MINOR_VERSION = ".".join(python_version().split(".")[:2])

TEST_FILE_PATH = pathlib.Path(__file__).parent.resolve()
PYPROJECT_TOML_PATH = list(TEST_FILE_PATH.glob("../pyproject.toml"))
MAKEFILE_PATH = list(TEST_FILE_PATH.glob("../Makefile"))
PRECOMMIT_HOOKS_PATH = list(TEST_FILE_PATH.glob("../.git/hooks"))

PYPROJECT_TOML_VERSION_REGEX = r"\n((?:py(?:thon)?)(?:[_-]version)?)\s=\D+(\d+.\d+)"


def test_file_uniqueness() -> None:
    # File uniqueness
    if len(PYPROJECT_TOML_PATH) != 1:
        raise ValueError(
            "Found more than one 'pyproject.toml':"
            f" {', '.join(str(p) for p in PYPROJECT_TOML_PATH) }"
        )

    if len(MAKEFILE_PATH) != 1:
        raise ValueError(
            f"Found more than one 'Makefile': {', '.join(str(p) for p in MAKEFILE_PATH) }"
        )


def test_consistent_versioning() -> None:
    with open(PYPROJECT_TOML_PATH[0], encoding="utf-8") as f:
        pyproject_toml = f.read()

    # TOML configurations
    toml_versions = re.findall(PYPROJECT_TOML_VERSION_REGEX, pyproject_toml)
    for var_name, var_version in toml_versions:
        if var_version != CURRENT_MINOR_VERSION:
            raise ValueError(
                f'"{var_name}" on file pyproject.toml is not set to {CURRENT_MINOR_VERSION}'
            )


def test_isset_precommit_hooks() -> None:
    if len(PRECOMMIT_HOOKS_PATH) == 0:
        raise ValueError("Pre-commit hooks are not set, run `make pre-commit` in `bash`")
