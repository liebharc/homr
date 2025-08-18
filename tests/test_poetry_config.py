import pathlib

TEST_FILE_PATH = pathlib.Path(__file__).parent.resolve()
PYPROJECT_TOML_PATH = list(TEST_FILE_PATH.glob("../pyproject.toml"))
MAKEFILE_PATH = list(TEST_FILE_PATH.glob("../Makefile"))
PRECOMMIT_HOOKS_PATH = list(TEST_FILE_PATH.glob("../.git/hooks"))


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


def test_isset_precommit_hooks() -> None:
    if len(PRECOMMIT_HOOKS_PATH) == 0:
        raise ValueError("Pre-commit hooks are not set, run `make pre-commit` in `bash`")
