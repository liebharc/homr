"""
Update and check docs/PROJECT_TRACKER.yaml for the adversarial-homr project.

This script is local repo automation. It normally does not need to be uploaded to
future agents. Future agents should usually receive only docs/PROJECT_TRACKER.yaml.

What it does:
    - Reads docs/PROJECT_TRACKER.yaml.
    - Extracts expected files, ignored paths, and benchmark scan paths.
    - Traverses the repository while avoiding heavy/generated directories.
    - Writes docs/repo_inventory.json with scan-derived repo facts.
    - Writes docs/FOLDER_INDEX.md with folder/file documentation status.
    - Replaces the auto-generated scan summary block inside PROJECT_TRACKER.yaml.
    - Checks expected files for existence, emptiness, documentation, and forbidden
      benchmark imports/calls.

This script deliberately uses only the Python standard library. It does not require
PyYAML. It parses only the stable subset of YAML used by PROJECT_TRACKER.yaml and
updates only the managed scan block between:

    # BEGIN AUTO-GENERATED SCAN SUMMARY
    # END AUTO-GENERATED SCAN SUMMARY

Human/agent decisions such as "status: done" remain manually maintained in the
tracker and must not be inferred merely from file existence.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
TRACKER_PATH = REPO_ROOT / "docs" / "PROJECT_TRACKER.yaml"
INVENTORY_PATH = REPO_ROOT / "docs" / "repo_inventory.json"
FOLDER_INDEX_PATH = REPO_ROOT / "docs" / "FOLDER_INDEX.md"

AUTO_BEGIN = "# BEGIN AUTO-GENERATED SCAN SUMMARY"
AUTO_END = "# END AUTO-GENERATED SCAN SUMMARY"

COMMON_ALWAYS_IGNORE = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    ".ipynb_checkpoints",
}

FORBIDDEN_IMPORT_MODULES = {
    "homr.segmentation.inference_segnet",
    "homr.transformer.staff2score",
    "homr.transformer.encoder_inference",
    "homr.transformer.decoder_inference",
    "homr.staff_parsing_tromr",
    "torch",
    "tensorflow",
}

FORBIDDEN_CALL_NAMES = {
    "Staff2Score",
    "get_decoder",
    "Encoder",
    "parse_staff_tromr",
}

SAFE_DETERMINISTIC_HOMR_IMPORT_PREFIXES = {
    "homr.autocrop",
    "homr.bar_line_detection",
    "homr.bounding_boxes",
    "homr.brace_dot_detection",
    "homr.color_adjust",
    "homr.constants",
    "homr.debug",
    "homr.find_peaks",
    "homr.image_utils",
    "homr.model",
    "homr.music_xml_generator",
    "homr.noise_filtering",
    "homr.note_detection",
    "homr.resize",
    "homr.simple_logging",
    "homr.staff_detection",
    "homr.staff_dewarping",
    "homr.staff_parsing",
    "homr.staff_regions",
    "homr.type_definitions",
    "homr.transformer.configs",
    "homr.transformer.vocabulary",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".txt",
    ".rst",
    ".ipynb",
}

SOURCE_LIKE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".rst",
    ".txt",
}


def now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def normalize_rel_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = value.strip("/")
    if value == ".":
        return "."
    return value


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def load_tracker_text() -> str:
    if not TRACKER_PATH.exists():
        raise FileNotFoundError(
            f"Tracker not found: {TRACKER_PATH}. Create docs/PROJECT_TRACKER.yaml first."
        )
    return read_text(TRACKER_PATH)


def strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if value in {"", "null", "None"}:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def extract_nested_list(text: str, key: str) -> list[str]:
    """
    Extract list values under a line like:

        ignored_deep_scan:
          - path

    This intentionally supports only the stable tracker format.
    """
    lines = text.splitlines()
    result: list[str] = []

    key_pattern = re.compile(rf"^\s*{re.escape(key)}:\s*$")
    start_index: int | None = None
    key_indent = 0

    for i, line in enumerate(lines):
        if key_pattern.match(line):
            start_index = i
            key_indent = len(line) - len(line.lstrip(" "))
            break

    if start_index is None:
        return result

    for line in lines[start_index + 1 :]:
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))

        if indent <= key_indent:
            break

        stripped = line.strip()
        if stripped.startswith("- "):
            result.append(normalize_rel_path(strip_yaml_scalar(stripped[2:])))

    return result


def parse_expected_files(text: str) -> dict[str, dict[str, Any]]:
    """
    Parse expected_files from the stable tracker YAML subset.
    """
    lines = text.splitlines()

    section_start: int | None = None

    for i, line in enumerate(lines):
        if line.strip() == "expected_files:":
            section_start = i
            break

    if section_start is None:
        return {}

    expected: dict[str, dict[str, Any]] = {}
    current_path: str | None = None
    in_checks = False
    in_notes = False

    for line in lines[section_start + 1 :]:
        if line and not line.startswith(" "):
            break

        if not line.strip():
            continue

        entry_match = re.match(r"^  ([^\s].*?):\s*$", line)
        if entry_match:
            current_path = normalize_rel_path(entry_match.group(1))
            expected[current_path] = {"checks": [], "notes": []}
            in_checks = False
            in_notes = False
            continue

        if current_path is None:
            continue

        stripped = line.strip()

        if stripped == "checks:":
            in_checks = True
            in_notes = False
            continue

        if stripped == "notes:":
            in_notes = True
            in_checks = False
            continue

        if in_checks and stripped.startswith("- "):
            expected[current_path]["checks"].append(strip_yaml_scalar(stripped[2:]))
            continue

        if in_notes and stripped.startswith("- "):
            expected[current_path]["notes"].append(strip_yaml_scalar(stripped[2:]))
            continue

        field_match = re.match(r"^    ([A-Za-z0-9_\-]+):\s*(.*)$", line)
        if field_match:
            field = field_match.group(1)
            value = strip_yaml_scalar(field_match.group(2))
            expected[current_path][field] = value
            in_checks = False
            in_notes = False

    return expected


def parse_task_queue(text: str) -> list[dict[str, Any]]:
    """
    Parse the simple list of task_queue entries from PROJECT_TRACKER.yaml.
    """
    lines = text.splitlines()
    start: int | None = None

    for i, line in enumerate(lines):
        if line.strip() == "task_queue:":
            start = i
            break

    if start is None:
        return []

    tasks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("- id:"):
            if current is not None:
                tasks.append(current)
            current = {"id": strip_yaml_scalar(stripped.split(":", 1)[1])}
            continue

        if current is None:
            continue

        field_match = re.match(r"^([A-Za-z0-9_\-]+):\s*(.*)$", stripped)
        if field_match and not stripped.startswith("- "):
            current[field_match.group(1)] = strip_yaml_scalar(field_match.group(2))

    if current is not None:
        tasks.append(current)

    return tasks


def parse_folder_manifest(text: str) -> dict[str, dict[str, Any]]:
    lines = text.splitlines()
    start: int | None = None

    for i, line in enumerate(lines):
        if line.strip() == "folder_manifest:":
            start = i
            break

    if start is None:
        return {}

    folders: dict[str, dict[str, Any]] = {}
    current_path: str | None = None

    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break

        if not line.strip():
            continue

        entry_match = re.match(r"^  ([^\s].*?):\s*$", line)
        if entry_match:
            current_path = normalize_rel_path(entry_match.group(1))
            folders[current_path] = {}
            continue

        if current_path is None:
            continue

        field_match = re.match(r"^    ([A-Za-z0-9_\-]+):\s*(.*)$", line)
        if field_match:
            folders[current_path][field_match.group(1)] = strip_yaml_scalar(
                field_match.group(2)
            )

    return folders


def path_is_under(path: str, parent: str) -> bool:
    path = normalize_rel_path(path)
    parent = normalize_rel_path(parent)
    if parent == ".":
        return True
    return path == parent or path.startswith(parent + "/")


def is_ignored_path(path: str, ignored: list[str]) -> bool:
    norm = normalize_rel_path(path)
    parts = norm.split("/")

    if any(part in COMMON_ALWAYS_IGNORE for part in parts):
        return True

    for ignored_path in ignored:
        ignored_norm = normalize_rel_path(ignored_path)
        if path_is_under(norm, ignored_norm):
            return True

    return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_read_small_text(path: Path, max_bytes: int = 1024 * 1024) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def summarize_ignored_path(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": relpath(path) if path.exists() else normalize_rel_path(str(path)),
        "exists": path.exists(),
        "kind": "ignored_path",
        "file_count": 0,
        "dir_count": 0,
        "total_size_bytes": 0,
        "sample_files": [],
    }

    if not path.exists():
        return item

    if path.is_file():
        item["file_count"] = 1
        item["total_size_bytes"] = path.stat().st_size
        item["sample_files"] = [path.name]
        return item

    sample_files: list[str] = []

    for root, dirs, files in os.walk(path):
        item["dir_count"] += len(dirs)
        item["file_count"] += len(files)

        for filename in files:
            file_path = Path(root) / filename
            try:
                item["total_size_bytes"] += file_path.stat().st_size
            except OSError:
                pass

            if len(sample_files) < 10:
                try:
                    sample_files.append(relpath(file_path))
                except ValueError:
                    sample_files.append(str(file_path))

    item["sample_files"] = sample_files
    return item


def parse_python_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module_docstring": None,
        "has_module_docstring": False,
        "classes": [],
        "functions": [],
        "imports": [],
        "parse_error": None,
    }

    text = safe_read_small_text(path)

    if text is None:
        info["parse_error"] = "not-text-or-too-large"
        return info

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        info["parse_error"] = f"SyntaxError: {exc}"
        return info

    docstring = ast.get_docstring(tree)
    info["module_docstring"] = docstring
    info["has_module_docstring"] = bool(docstring and docstring.strip())

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["functions"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module
            info["imports"].append(module)

    return info


def parse_notebook_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cell_count": None,
        "first_markdown_heading": None,
        "has_intro_markdown": False,
        "parse_error": None,
    }

    try:
        data = json.loads(read_text(path))
    except Exception as exc:
        info["parse_error"] = f"{type(exc).__name__}: {exc}"
        return info

    cells = data.get("cells", [])
    info["cell_count"] = len(cells)

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        source = source.strip()

        if source:
            info["has_intro_markdown"] = True
            for line in source.splitlines():
                if line.strip().startswith("#"):
                    info["first_markdown_heading"] = line.strip()
                    return info
            info["first_markdown_heading"] = source.splitlines()[0][:120]
            return info

    return info


def parse_markdown_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "first_heading": None,
        "has_first_heading": False,
        "first_paragraph": None,
    }

    text = safe_read_small_text(path)

    if text is None:
        return info

    paragraph_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#") and info["first_heading"] is None:
            info["first_heading"] = stripped
            info["has_first_heading"] = True
            continue

        if stripped and not stripped.startswith("#") and info["first_paragraph"] is None:
            paragraph_lines.append(stripped)

        if paragraph_lines and not stripped:
            info["first_paragraph"] = " ".join(paragraph_lines)[:300]
            break

    if paragraph_lines and info["first_paragraph"] is None:
        info["first_paragraph"] = " ".join(paragraph_lines)[:300]

    return info


def parse_mapping_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "top_level_keys": [],
        "parse_error": None,
    }

    suffix = path.suffix.lower()
    text = safe_read_small_text(path)

    if text is None:
        info["parse_error"] = "not-text-or-too-large"
        return info

    if suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                info["top_level_keys"] = list(data.keys())[:50]
            return info
        except Exception as exc:
            info["parse_error"] = f"{type(exc).__name__}: {exc}"
            return info

    keys: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if not line.startswith((" ", "\t")):
            match = re.match(r"^([A-Za-z0-9_.\-\"']+)\s*[:=]", line)
            if match:
                keys.append(match.group(1).strip("\"'"))

    info["top_level_keys"] = keys[:50]
    return info


def describe_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    suffix = path.suffix.lower()

    item: dict[str, Any] = {
        "path": relpath(path),
        "name": path.name,
        "suffix": suffix,
        "exists": True,
        "is_file": True,
        "size_bytes": stat.st_size,
        "empty": stat.st_size == 0,
        "sha256": None,
        "description_source": None,
        "description": None,
        "documentation": {},
    }

    if suffix in SOURCE_LIKE_EXTENSIONS and stat.st_size <= 20 * 1024 * 1024:
        try:
            item["sha256"] = file_sha256(path)
        except OSError:
            item["sha256"] = None

    if suffix == ".py":
        py_info = parse_python_file(path)
        item["documentation"]["python"] = py_info
        if py_info.get("has_module_docstring"):
            first_line = str(py_info["module_docstring"]).strip().splitlines()[0]
            item["description_source"] = "module_docstring"
            item["description"] = first_line

    elif suffix == ".ipynb":
        nb_info = parse_notebook_file(path)
        item["documentation"]["notebook"] = nb_info
        if nb_info.get("first_markdown_heading"):
            item["description_source"] = "first_markdown"
            item["description"] = nb_info["first_markdown_heading"]

    elif suffix in {".md", ".rst"}:
        md_info = parse_markdown_file(path)
        item["documentation"]["markdown"] = md_info
        if md_info.get("first_heading"):
            item["description_source"] = "first_heading"
            item["description"] = md_info["first_heading"]

    elif suffix in {".yaml", ".yml", ".toml", ".json"}:
        map_info = parse_mapping_file(path)
        item["documentation"]["mapping"] = map_info
        keys = map_info.get("top_level_keys") or []
        if keys:
            item["description_source"] = "top_level_keys"
            item["description"] = "Top-level keys: " + ", ".join(keys[:10])

    return item


def folder_has_doc_file(path: Path) -> bool:
    return (path / "README.md").exists() or (path / ".repo_meta.json").exists()


def describe_folder(path: Path, folder_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rel = "." if path == REPO_ROOT else relpath(path)
    manifest = folder_manifest.get(rel, {})

    item = {
        "path": rel,
        "name": path.name if rel != "." else ".",
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "has_doc_file": folder_has_doc_file(path),
        "doc_files": [
            name
            for name in ("README.md", ".repo_meta.json")
            if (path / name).exists()
        ],
        "manifest_purpose": manifest.get("purpose"),
        "manifest_status": manifest.get("status"),
        "manifest_documentation_file": manifest.get("documentation_file"),
    }

    return item


def node_full_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = node_full_name(node.value)
        if base:
            return base + "." + node.attr
        return node.attr
    return ""


def scan_for_forbidden_python_usage(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    text = safe_read_small_text(path)
    if text is None:
        return findings

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        findings.append(
            {
                "path": relpath(path),
                "kind": "parse_error",
                "detail": f"SyntaxError while checking forbidden usage: {exc}",
            }
        )
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module in FORBIDDEN_IMPORT_MODULES or any(
                    module.startswith(forbidden + ".")
                    for forbidden in FORBIDDEN_IMPORT_MODULES
                ):
                    findings.append(
                        {
                            "path": relpath(path),
                            "kind": "forbidden_import",
                            "line": getattr(node, "lineno", None),
                            "module": module,
                        }
                    )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module

            if module in FORBIDDEN_IMPORT_MODULES or any(
                module.startswith(forbidden + ".")
                for forbidden in FORBIDDEN_IMPORT_MODULES
            ):
                findings.append(
                    {
                        "path": relpath(path),
                        "kind": "forbidden_import",
                        "line": getattr(node, "lineno", None),
                        "module": module,
                    }
                )

            for alias in node.names:
                if alias.name in FORBIDDEN_CALL_NAMES:
                    findings.append(
                        {
                            "path": relpath(path),
                            "kind": "forbidden_imported_symbol",
                            "line": getattr(node, "lineno", None),
                            "symbol": alias.name,
                            "module": module,
                        }
                    )

        elif isinstance(node, ast.Call):
            full_name = node_full_name(node.func)
            leaf_name = full_name.split(".")[-1] if full_name else ""

            if leaf_name in FORBIDDEN_CALL_NAMES:
                findings.append(
                    {
                        "path": relpath(path),
                        "kind": "forbidden_call",
                        "line": getattr(node, "lineno", None),
                        "call": full_name,
                    }
                )

    return findings


def should_scan_forbidden_usage(path_rel: str, benchmark_paths: list[str]) -> bool:
    for scan_path in benchmark_paths:
        scan_path = normalize_rel_path(scan_path)
        if path_is_under(path_rel, scan_path):
            return True
    return False


def scan_repository(
    ignored_paths: list[str],
    benchmark_scan_paths: list[str],
    folder_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "generated_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "ignored_deep_scan": ignored_paths,
        "benchmark_scan_paths": benchmark_scan_paths,
        "files": {},
        "folders": {},
        "omitted_paths": {},
        "forbidden_findings": [],
        "documentation_findings": [],
    }

    for ignored in ignored_paths:
        path = REPO_ROOT / ignored
        if path.exists():
            inventory["omitted_paths"][normalize_rel_path(ignored)] = summarize_ignored_path(path)

    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        root_rel = "." if root_path == REPO_ROOT else relpath(root_path)

        dirs[:] = [
            dirname
            for dirname in dirs
            if not is_ignored_path(
                normalize_rel_path((root_path / dirname).resolve().relative_to(REPO_ROOT.resolve()).as_posix()),
                ignored_paths,
            )
        ]

        if is_ignored_path(root_rel, ignored_paths):
            continue

        folder_info = describe_folder(root_path, folder_manifest)
        inventory["folders"][root_rel] = folder_info

        if (
            root_rel != "."
            and not folder_info["has_doc_file"]
            and folder_info.get("manifest_documentation_file") not in {"none-required-generated", "existing-or-central-index"}
        ):
            inventory["documentation_findings"].append(
                {
                    "path": root_rel,
                    "kind": "folder_missing_doc_file",
                    "detail": "Folder has no README.md or .repo_meta.json.",
                }
            )

        for filename in files:
            file_path = root_path / filename
            file_rel = relpath(file_path)

            if is_ignored_path(file_rel, ignored_paths):
                continue

            try:
                info = describe_file(file_path)
            except OSError as exc:
                info = {
                    "path": file_rel,
                    "exists": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            inventory["files"][file_rel] = info

            suffix = file_path.suffix.lower()

            if suffix == ".py":
                py_doc = info.get("documentation", {}).get("python", {})
                if (
                    filename != "__init__.py"
                    and not py_doc.get("has_module_docstring")
                ):
                    inventory["documentation_findings"].append(
                        {
                            "path": file_rel,
                            "kind": "python_missing_module_docstring",
                            "detail": "Python file has no module docstring.",
                        }
                    )

                if should_scan_forbidden_usage(file_rel, benchmark_scan_paths):
                    inventory["forbidden_findings"].extend(
                        scan_for_forbidden_python_usage(file_path)
                    )

            elif suffix == ".ipynb":
                nb_doc = info.get("documentation", {}).get("notebook", {})
                if not nb_doc.get("has_intro_markdown"):
                    inventory["documentation_findings"].append(
                        {
                            "path": file_rel,
                            "kind": "notebook_missing_intro_markdown",
                            "detail": "Notebook has no introductory markdown cell.",
                        }
                    )

            elif suffix in {".md", ".rst"}:
                md_doc = info.get("documentation", {}).get("markdown", {})
                if not md_doc.get("has_first_heading"):
                    inventory["documentation_findings"].append(
                        {
                            "path": file_rel,
                            "kind": "markdown_missing_heading",
                            "detail": "Markdown/RST file has no heading.",
                        }
                    )

    return inventory


def expected_file_report(
    expected_files: dict[str, dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    scanned_files = inventory["files"]

    for path, manifest in expected_files.items():
        norm = normalize_rel_path(path)
        actual = scanned_files.get(norm)
        path_obj = REPO_ROOT / norm

        item: dict[str, Any] = {
            "path": norm,
            "manifest_status": manifest.get("status", ""),
            "purpose": manifest.get("purpose", ""),
            "exists": path_obj.exists(),
            "scanned": actual is not None,
            "empty": None,
            "size_bytes": None,
            "documentation_ok": None,
            "detected_status": None,
        }

        if path_obj.exists() and path_obj.is_file():
            try:
                size = path_obj.stat().st_size
            except OSError:
                size = None

            item["size_bytes"] = size
            item["empty"] = size == 0 if size is not None else None

            if actual is not None:
                suffix = actual.get("suffix", "")
                if suffix == ".py":
                    py_doc = actual.get("documentation", {}).get("python", {})
                    item["documentation_ok"] = bool(
                        path_obj.name == "__init__.py" or py_doc.get("has_module_docstring")
                    )
                elif suffix == ".ipynb":
                    nb_doc = actual.get("documentation", {}).get("notebook", {})
                    item["documentation_ok"] = bool(nb_doc.get("has_intro_markdown"))
                elif suffix in {".md", ".rst"}:
                    md_doc = actual.get("documentation", {}).get("markdown", {})
                    item["documentation_ok"] = bool(md_doc.get("has_first_heading"))
                else:
                    item["documentation_ok"] = True

            if item["empty"]:
                item["detected_status"] = "empty-placeholder"
            elif not item["documentation_ok"]:
                item["detected_status"] = "needs-documentation"
            else:
                item["detected_status"] = "present"

        elif path_obj.exists() and path_obj.is_dir():
            item["detected_status"] = "present-directory"
            item["documentation_ok"] = folder_has_doc_file(path_obj)
        else:
            item["detected_status"] = "missing"

        report[norm] = item

    return report


def inventory_summary(
    inventory: dict[str, Any],
    expected_report: dict[str, Any],
) -> dict[str, Any]:
    missing_expected = [
        path
        for path, item in expected_report.items()
        if item.get("detected_status") == "missing"
    ]

    empty_expected = [
        path
        for path, item in expected_report.items()
        if item.get("detected_status") == "empty-placeholder"
    ]

    needs_doc_expected = [
        path
        for path, item in expected_report.items()
        if item.get("detected_status") == "needs-documentation"
    ]

    folder_doc_findings = [
        item
        for item in inventory["documentation_findings"]
        if item["kind"] == "folder_missing_doc_file"
    ]

    return {
        "generated_at": inventory["generated_at"],
        "repo_root": inventory["repo_root"],
        "totals": {
            "files_scanned": len(inventory["files"]),
            "folders_scanned": len(inventory["folders"]),
            "omitted_paths": len(inventory["omitted_paths"]),
            "expected_files": len(expected_report),
            "expected_files_missing": len(missing_expected),
            "expected_files_empty": len(empty_expected),
            "expected_files_needing_documentation": len(needs_doc_expected),
            "forbidden_findings": len(inventory["forbidden_findings"]),
            "documentation_findings": len(inventory["documentation_findings"]),
            "folder_doc_findings": len(folder_doc_findings),
        },
        "expected_files_missing": missing_expected,
        "expected_files_empty": empty_expected,
        "expected_files_needing_documentation": needs_doc_expected,
        "forbidden_findings": inventory["forbidden_findings"],
        "documentation_findings_sample": inventory["documentation_findings"][:100],
        "omitted_path_summaries": inventory["omitted_paths"],
        "expected_file_report": expected_report,
    }


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def to_yaml(value: Any, indent: int = 0) -> str:
    """
    Minimal YAML serializer for simple JSON-like structures.

    It emits valid YAML using quoted strings and plain booleans/null/numbers.
    """
    space = " " * indent

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        return yaml_quote(value)

    if isinstance(value, list):
        if not value:
            return "[]"

        lines: list[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}- {to_yaml(item, indent + 2).lstrip()}")
            else:
                lines.append(f"{space}- {to_yaml(item, indent + 2)}")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "{}"

        lines = []
        for key, item in value.items():
            key_str = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{space}{key_str}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{space}{key_str}: {to_yaml(item, indent + 2)}")
        return "\n".join(lines)

    return yaml_quote(str(value))


def update_tracker_scan_block(tracker_text: str, scan_summary: dict[str, Any]) -> str:
    generated_block = (
        f"{AUTO_BEGIN}\n"
        "scan_generated:\n"
        + to_yaml(scan_summary, indent=2)
        + "\n"
        f"{AUTO_END}"
    )

    pattern = re.compile(
        re.escape(AUTO_BEGIN) + r".*?" + re.escape(AUTO_END),
        flags=re.DOTALL,
    )

    if pattern.search(tracker_text):
        return pattern.sub(generated_block, tracker_text)

    if not tracker_text.endswith("\n"):
        tracker_text += "\n"

    return tracker_text + "\n" + generated_block + "\n"


def write_inventory(inventory: dict[str, Any]) -> None:
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )


def write_folder_index(inventory: dict[str, Any]) -> None:
    lines: list[str] = []

    lines.append("# Folder Index")
    lines.append("")
    lines.append("Generated by `python scripts/update_project_tracker.py --write`.")
    lines.append("")
    lines.append("This file is generated for local browsing. The normal agent handoff file is `docs/PROJECT_TRACKER.yaml`.")
    lines.append("")

    lines.append("## Folders")
    lines.append("")

    for path in sorted(inventory["folders"]):
        folder = inventory["folders"][path]
        lines.append(f"### `{path}`")
        lines.append("")
        purpose = folder.get("manifest_purpose") or "No manifest purpose recorded."
        lines.append(f"- Purpose: {purpose}")
        lines.append(f"- Documentation file present: {folder.get('has_doc_file')}")
        doc_files = folder.get("doc_files") or []
        if doc_files:
            lines.append(f"- Documentation files: {', '.join(doc_files)}")
        status = folder.get("manifest_status")
        if status:
            lines.append(f"- Manifest status: {status}")
        lines.append("")

    lines.append("## Source-like files")
    lines.append("")

    for path in sorted(inventory["files"]):
        file_info = inventory["files"][path]
        suffix = file_info.get("suffix", "")

        if suffix not in SOURCE_LIKE_EXTENSIONS:
            continue

        lines.append(f"### `{path}`")
        lines.append("")
        description = file_info.get("description") or "No description detected."
        lines.append(f"- Description: {description}")
        lines.append(f"- Size bytes: {file_info.get('size_bytes')}")
        lines.append(f"- Empty: {file_info.get('empty')}")

        if suffix == ".py":
            py_doc = file_info.get("documentation", {}).get("python", {})
            lines.append(f"- Module docstring: {py_doc.get('has_module_docstring')}")
            classes = py_doc.get("classes") or []
            functions = py_doc.get("functions") or []
            if classes:
                lines.append(f"- Classes: {', '.join(classes[:20])}")
            if functions:
                lines.append(f"- Functions: {', '.join(functions[:20])}")

        elif suffix == ".ipynb":
            nb_doc = file_info.get("documentation", {}).get("notebook", {})
            lines.append(f"- Intro markdown: {nb_doc.get('has_intro_markdown')}")
            lines.append(f"- Cell count: {nb_doc.get('cell_count')}")

        lines.append("")

    write_text(FOLDER_INDEX_PATH, "\n".join(lines) + "\n")


def run_scan() -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    tracker_text = load_tracker_text()
    ignored_paths = extract_nested_list(tracker_text, "ignored_deep_scan")
    benchmark_scan_paths = extract_nested_list(tracker_text, "benchmark_scan_paths")
    expected_files = parse_expected_files(tracker_text)
    folder_manifest = parse_folder_manifest(tracker_text)

    inventory = scan_repository(
        ignored_paths=ignored_paths,
        benchmark_scan_paths=benchmark_scan_paths,
        folder_manifest=folder_manifest,
    )

    expected_report = expected_file_report(expected_files, inventory)
    summary = inventory_summary(inventory, expected_report)

    return tracker_text, inventory, expected_report, summary


def print_check_summary(summary: dict[str, Any]) -> None:
    totals = summary["totals"]

    print("Tracker check summary")
    print("=====================")
    print(f"Generated at: {summary['generated_at']}")
    print(f"Files scanned: {totals['files_scanned']}")
    print(f"Folders scanned: {totals['folders_scanned']}")
    print(f"Omitted paths: {totals['omitted_paths']}")
    print(f"Expected files: {totals['expected_files']}")
    print(f"Missing expected files: {totals['expected_files_missing']}")
    print(f"Empty expected files: {totals['expected_files_empty']}")
    print(f"Expected files needing documentation: {totals['expected_files_needing_documentation']}")
    print(f"Forbidden findings: {totals['forbidden_findings']}")
    print(f"Documentation findings: {totals['documentation_findings']}")
    print("")

    if summary["expected_files_missing"]:
        print("Missing expected files:")
        for path in summary["expected_files_missing"]:
            print(f"  - {path}")
        print("")

    if summary["expected_files_empty"]:
        print("Empty expected files:")
        for path in summary["expected_files_empty"]:
            print(f"  - {path}")
        print("")

    if summary["expected_files_needing_documentation"]:
        print("Expected files needing documentation:")
        for path in summary["expected_files_needing_documentation"]:
            print(f"  - {path}")
        print("")

    if summary["forbidden_findings"]:
        print("Forbidden benchmark usage findings:")
        for finding in summary["forbidden_findings"]:
            line = finding.get("line")
            line_text = f":{line}" if line is not None else ""
            detail = finding.get("module") or finding.get("symbol") or finding.get("call") or finding.get("detail")
            print(f"  - {finding['path']}{line_text} [{finding['kind']}] {detail}")
        print("")


def print_next_actions(tracker_text: str, summary: dict[str, Any]) -> None:
    tasks = parse_task_queue(tracker_text)

    actionable_statuses = {
        "next",
        "needs-create",
        "needs-edit",
        "needs-review",
        "blocked",
        "blocked-needs-file",
        "pending",
    }

    priority_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    actionable = [
        task
        for task in tasks
        if task.get("status") in actionable_statuses
    ]

    actionable.sort(
        key=lambda task: (
            priority_rank.get(str(task.get("priority", "medium")), 1),
            str(task.get("id", "")),
        )
    )

    print("Next actions from task_queue")
    print("============================")
    if not actionable:
        print("No actionable tasks found.")
    else:
        for task in actionable[:20]:
            print(f"- {task.get('id')} [{task.get('priority', 'medium')}] {task.get('status')}: {task.get('title')}")
            if task.get("success_command"):
                print(f"  check: {task.get('success_command')}")

    print("")
    print("Scan-derived urgent items")
    print("=========================")

    if summary["forbidden_findings"]:
        print("- Resolve forbidden benchmark usage findings first.")
    if summary["expected_files_missing"]:
        print("- Missing expected files:")
        for path in summary["expected_files_missing"][:20]:
            print(f"  - {path}")
    if summary["expected_files_empty"]:
        print("- Empty expected files:")
        for path in summary["expected_files_empty"][:20]:
            print(f"  - {path}")

    if (
        not summary["forbidden_findings"]
        and not summary["expected_files_missing"]
        and not summary["expected_files_empty"]
    ):
        print("- No missing/empty/forbidden urgent items detected by scan.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update and check docs/PROJECT_TRACKER.yaml."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Scan repo and print summary. Does not write files.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write inventory, folder index, and update tracker scan section.",
    )

    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Write docs/repo_inventory.json only.",
    )

    parser.add_argument(
        "--folder-index",
        action="store_true",
        help="Write docs/FOLDER_INDEX.md only.",
    )

    parser.add_argument(
        "--next",
        action="store_true",
        help="Print next tasks and urgent scan-derived items.",
    )

    parser.add_argument(
        "--fail-on-docs",
        action="store_true",
        help="Make --check exit nonzero if documentation findings exist.",
    )

    args = parser.parse_args()

    if not any([args.check, args.write, args.inventory, args.folder_index, args.next]):
        args.check = True

    try:
        tracker_text, inventory, _expected_report, summary = run_scan()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    if args.inventory or args.write:
        write_inventory(inventory)
        print(f"Wrote {relpath(INVENTORY_PATH)}")

    if args.folder_index or args.write:
        write_folder_index(inventory)
        print(f"Wrote {relpath(FOLDER_INDEX_PATH)}")

    if args.write:
        updated_tracker = update_tracker_scan_block(tracker_text, summary)
        write_text(TRACKER_PATH, updated_tracker)
        print(f"Updated {relpath(TRACKER_PATH)}")

    if args.check:
        print_check_summary(summary)

    if args.next:
        print_next_actions(tracker_text, summary)

    should_fail = False

    if summary["forbidden_findings"]:
        should_fail = True

    if summary["expected_files_missing"]:
        should_fail = True

    if summary["expected_files_empty"]:
        should_fail = True

    if args.fail_on_docs and summary["documentation_findings_sample"]:
        should_fail = True

    if args.check and should_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()