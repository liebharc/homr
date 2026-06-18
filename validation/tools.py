"""
OMR tools: callables with signature (kern_text: str, image: Path | None) -> str.

Each tool represents one OMR system or conversion pipeline under test:
  - kern_text: ground-truth kern; used directly by format converters (music21, hum2xml).
  - image:     score image for real OMR tools (homr, transcoda); ignored by converters.
  - return:    recognised score as raw text - MusicXML or **kern.

The benchmark calls to_xml() to normalise the output before computing NED, so tools
may return either format.  The raw output is stored as-is in the benchmark database.

To add a new tool: implement a callable with the signature above and add it to TOOLS.
"""

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ToolFn = Callable[[str, "Path | None"], str]

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
_HUM2XML = _REPO_ROOT / "datasets" / "hum2xml"
_HOMR_MAIN = _REPO_ROOT / "homr" / "main.py"
_TRANSCODA_ROOT = _REPO_ROOT.parent / "transcoda"
_TRANSCODA_SCRIPT = _TRANSCODA_ROOT / "scripts" / "inference.py"
_TRANSCODA_WEIGHTS = _TRANSCODA_ROOT / "weights" / "transcoda-59M-zeroshot-v1.ckpt"
_SMT_ROOT = _REPO_ROOT.parent / "SMT"
_SMT_WEIGHTS = _SMT_ROOT / "weights" / "model.safetensors"

# Batch inference script run inside SMT's uv environment.
# Reads a JSON list of image paths from sys.argv[1] (a temp file), loads the model
# once, processes all images, and prints a JSON list of {ok, kern/error} objects.
# Images are resized to fit within the model's trained 2100x2970 bounds before inference.
_SMT_BATCH_INFERENCE = "\n".join(
    [
        "import sys, json, torch, cv2, numpy as np",
        "from data_augmentation.data_augmentation import convert_img_to_tensor",
        "from smt_model import SMTModelForCausalLM",
        "paths = json.loads(open(sys.argv[1]).read())",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "model = SMTModelForCausalLM.from_pretrained(sys.argv[2]).to(device)",
        "model.eval()",
        "MAXW, MAXH = 2100, 2970",
        "results = []",
        "for path in paths:",
        "    try:",
        "        img = cv2.imread(path)",
        "        h, w = img.shape[:2]",
        "        scale = min(MAXW / w, MAXH / h, 1.0)",
        "        if scale < 1.0:",
        "            img = cv2.resize(img, (int(w * scale), int(h * scale)))",
        "        with torch.no_grad():",
        "            t = convert_img_to_tensor(img).unsqueeze(0).to(device)",
        "            preds, _ = model.predict(t, convert_to_str=True)",
        "        kern = ''.join(preds)",
        "        kern = kern.replace('<b>', '\\n').replace('<s>', ' ').replace('<t>', '\\t')",
        "        results.append({'ok': True, 'kern': kern})",
        "    except Exception as e:",
        "        results.append({'ok': False, 'error': str(e)})",
        "print(json.dumps(results))",
    ]
)

_CLEF_WITHOUT_LINE = re.compile(r"(\*clef[A-Z])(?!\d)")
_CLEF_DEFAULT_LINE = {"F": "4", "G": "2", "C": "3"}
_PURE_NUMBER = re.compile(r"^\d+$")


def _normalize_kern(kern: str) -> str:
    """Fix non-standard kern notation that hum2xml cannot parse."""

    def _fix_clef(m: re.Match) -> str:
        letter = m.group(1)[-1]
        return m.group(1) + _CLEF_DEFAULT_LINE.get(letter, "1")

    kern = _CLEF_WITHOUT_LINE.sub(_fix_clef, kern)

    fixed: list[str] = []
    for raw_line in kern.split("\n"):
        line = raw_line
        if "\t" in line:
            tokens = line.split("\t")
            if any(t.startswith("=") for t in tokens):
                tokens = ["=" if _PURE_NUMBER.match(t) else t for t in tokens]
                line = "\t".join(tokens)
        fixed.append(line)
    return "\n".join(fixed)


def is_kern(text: str) -> bool:
    """Return True if text looks like **kern, False if it looks like MusicXML."""
    stripped = text.lstrip()
    return stripped.startswith("**") or "\t**kern" in stripped[:200]


# ---------------------------------------------------------------------------
# Format converters  (use kern_text, ignore image)
# ---------------------------------------------------------------------------


def hum2xml(kern_text: str, image: Path | None) -> str:
    """Convert **kern text to MusicXML via the external hum2xml binary."""
    if not _HUM2XML.exists():
        raise FileNotFoundError(f"hum2xml not found at {_HUM2XML}.")
    kern_text = _normalize_kern(kern_text)
    if not kern_text.endswith("\n"):
        kern_text += "\n"
    result = subprocess.run(  # noqa: S603
        [str(_HUM2XML)],
        input=kern_text.encode("utf-8"),
        capture_output=True,
        cwd=str(_HUM2XML.parent),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hum2xml exited with code {result.returncode}\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')}\n"
            f"stdout: {result.stdout.decode('utf-8', errors='replace')[:500]}"
        )
    return result.stdout.decode("utf-8")


def _opus_to_score(opus: Any) -> Any:
    """Concatenate all Opus scores into one Score by joining same-indexed parts."""
    import music21 as m21  # noqa: PLC0415
    import music21.stream  # noqa: PLC0415

    scores = list(getattr(opus, "scores", []))
    n_parts = max((len(list(getattr(sc, "parts", []))) for sc in scores), default=0)
    new_parts = [m21.stream.Part() for _ in range(n_parts)]
    measure_offset = 0
    for sc in scores:
        sc_parts = list(getattr(sc, "parts", []))
        n_meas = max((len(list(p.getElementsByClass("Measure"))) for p in sc_parts), default=0)
        for p_idx in range(n_parts):
            if p_idx >= len(sc_parts):
                continue
            for m_rel, measure in enumerate(sc_parts[p_idx].getElementsByClass("Measure")):
                new_m = m21.stream.Measure(number=measure_offset + m_rel + 1)
                for el in measure:
                    new_m.insert(el.offset, copy.deepcopy(el))
                new_parts[p_idx].append(new_m)
        measure_offset += n_meas
    merged = m21.stream.Score()
    for p in new_parts:
        merged.append(p)
    return merged


def music21(kern_text: str, image: Path | None) -> str:
    """Convert **kern text to MusicXML via music21."""
    import music21 as m21  # noqa: PLC0415
    import music21.musicxml.m21ToXml  # noqa: PLC0415
    import music21.stream  # noqa: PLC0415

    parsed = m21.converter.parse(kern_text, format="humdrum")
    if isinstance(parsed, m21.stream.Opus):
        parsed = _opus_to_score(parsed)
    return m21.musicxml.m21ToXml.GeneralObjectExporter(parsed).parse().decode("utf-8")


# ---------------------------------------------------------------------------
# OMR tools  (use image, ignore kern_text)
# ---------------------------------------------------------------------------


def _run_homr_on_dir(image_dir: Path) -> None:
    """Run homr on every image in image_dir, writing .musicxml alongside each."""
    result = subprocess.run(  # noqa: S603
        ["poetry", "run", "python", str(_HOMR_MAIN), str(image_dir)],  # noqa: S607
        cwd=str(_REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"homr exited with code {result.returncode}\n{stderr[:1000]}")


class HomrTool:
    """
    OMR tool wrapping homr.  Outputs MusicXML.

    Supports single-image mode (one subprocess call per image) and batch mode
    via batch_run(), which passes all images to homr as a directory in a single
    subprocess call - the neural network weights are loaded only once, making it
    significantly faster when processing many samples.
    """

    def __call__(self, kern_text: str, image: Path | None) -> str:
        if image is None:
            raise ValueError("homr requires an image; the dataset may not provide one.")
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / ("img" + image.suffix)
            shutil.copy(image, dst)
            _run_homr_on_dir(Path(tmpdir))
            xml_path = dst.with_suffix(".musicxml")
            if not xml_path.exists():
                raise RuntimeError(f"homr produced no .musicxml for {image.name}")
            return xml_path.read_text(encoding="utf-8")

    def batch_run(
        self,
        samples: list[tuple[str, str, Path | None]],
    ) -> list[tuple[str | None, str | None]]:
        """
        Process all samples in one homr invocation.

        Returns a list of (output_xml, error_message) parallel to samples.
        output_xml is None (and error_message set) when a sample failed.
        """
        results: list[tuple[str | None, str | None]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stem_by_index: list[str | None] = []

            for sample_id, _kern, image in samples:
                if image is None:
                    stem_by_index.append(None)
                    continue
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", sample_id)
                dst = tmp / (safe + image.suffix)
                # Avoid name collisions by appending a counter when needed.
                if dst.exists():
                    dst = tmp / (safe + f"_{os.urandom(4).hex()}" + image.suffix)
                shutil.copy(image, dst)
                stem_by_index.append(dst.stem)

            try:
                _run_homr_on_dir(tmp)
            except RuntimeError as e:
                return [(None, str(e))] * len(samples)

            for stem in stem_by_index:
                if stem is None:
                    results.append((None, "no image provided for this sample"))
                    continue
                xml_path = tmp / (stem + ".musicxml")
                if xml_path.exists():
                    results.append((xml_path.read_text(encoding="utf-8"), None))
                else:
                    results.append((None, f"homr produced no .musicxml for {stem}"))

        return results


def transcoda(kern_text: str, image: Path | None) -> str:
    """
    OMR tool wrapping transcoda.  Expects the transcoda repository next to this repo.
    Outputs **kern text (the benchmark converts it to MusicXML automatically).
    """
    if image is None:
        raise ValueError("transcoda requires an image; the dataset may not provide one.")
    if not _TRANSCODA_SCRIPT.exists():
        raise FileNotFoundError(
            f"transcoda not found at {_TRANSCODA_ROOT}. "
            "Expected the transcoda repo to be a sibling of the homr repo."
        )
    if not _TRANSCODA_WEIGHTS.exists():
        raise FileNotFoundError(f"transcoda weights not found at {_TRANSCODA_WEIGHTS}.")
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "scripts/inference.py",
            "--weights",
            str(_TRANSCODA_WEIGHTS),
            "--image",
            str(image),
        ],
        cwd=str(_TRANSCODA_ROOT),
        check=True,
    )
    pred_path = image.with_name(image.stem + "_pred.krn")
    if not pred_path.exists():
        raise RuntimeError(f"transcoda produced no _pred.krn for {image.name}")
    kern = pred_path.read_text(encoding="utf-8")
    pred_path.unlink()
    return kern


def _ensure_kern_header(kern: str) -> str:
    """Prepend **kern spine headers if the output omits them (as SMT does)."""
    for line in kern.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if not stripped.startswith("**"):
            n_spines = len(line.split("\t"))
            return "\t".join(["**kern"] * n_spines) + "\n" + kern
        break
    return kern


class SmtTool:
    """
    OMR tool wrapping SMT (Sheet Music Transformer).
    Weights are loaded from SMT/weights/ next to the homr repo.
    Outputs **kern text (the benchmark converts it to MusicXML automatically).

    Supports single-image mode and batch mode via batch_run(), which loads the
    model once for all images - much faster than reloading per sample.
    """

    def _check_paths(self) -> None:
        if not _SMT_ROOT.exists():
            raise FileNotFoundError(
                f"SMT not found at {_SMT_ROOT}. "
                "Expected the SMT repo to be a sibling of the homr repo."
            )
        if not _SMT_WEIGHTS.exists():
            raise FileNotFoundError(f"SMT weights not found at {_SMT_WEIGHTS}.")

    def __call__(self, kern_text: str, image: Path | None) -> str:
        if image is None:
            raise ValueError("SMT requires an image; the dataset may not provide one.")
        results = self.batch_run([("_", "", image)])
        output, error = results[0]
        if error:
            raise RuntimeError(error)
        return output  # type: ignore[return-value]

    def batch_run(
        self,
        samples: list[tuple[str, str, Path | None]],
    ) -> list[tuple[str | None, str | None]]:
        """
        Process all samples in one SMT invocation (model loaded once).

        Returns a list of (output_kern, error_message) parallel to samples.
        output_kern is None (and error_message set) when a sample failed.
        """
        self._check_paths()

        results: list[tuple[str | None, str | None]] = [
            (None, "no image provided for this sample") for _ in samples
        ]
        valid: list[tuple[int, Path]] = [
            (i, image) for i, (_, _, image) in enumerate(samples) if image is not None
        ]
        if not valid:
            return results

        with tempfile.TemporaryDirectory() as tmpdir:
            paths_file = Path(tmpdir) / "paths.json"
            paths_file.write_text(json.dumps([str(img) for _, img in valid]))

            proc = subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "uv",
                    "run",
                    "python",
                    "-c",
                    _SMT_BATCH_INFERENCE,
                    str(paths_file),
                    str(_SMT_ROOT / "weights"),
                ],
                cwd=str(_SMT_ROOT),
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace")
                error = f"SMT exited with code {proc.returncode}\n{stderr[:1000]}"
                for orig_idx, _ in valid:
                    results[orig_idx] = (None, error)
                return results

            batch_results: list[dict] = json.loads(proc.stdout.decode("utf-8"))
            for (orig_idx, _), br in zip(valid, batch_results, strict=True):
                if br["ok"]:
                    results[orig_idx] = (_ensure_kern_header(br["kern"]), None)
                else:
                    results[orig_idx] = (None, br["error"])

        return results


def oemer(kern_text: str, image: Path | None) -> str:
    """OMR tool wrapping oemer. Outputs MusicXML to the current working directory."""
    if image is None:
        raise ValueError("oemer requires an image; the dataset may not provide one.")
    result = subprocess.run(  # noqa: S603
        ["oemer", str(image)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"oemer exited with code {result.returncode}\n{stderr[:1000]}")
    # oemer writes <stem>.musicxml in cwd, not next to the input image
    xml_path = Path(image.stem + ".musicxml")
    if not xml_path.exists():
        raise RuntimeError(f"oemer produced no .musicxml for {image.name}")
    xml = xml_path.read_text(encoding="utf-8")
    xml_path.unlink()
    return xml


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_homr_tool = HomrTool()
_smt_tool = SmtTool()

TOOLS: dict[str, ToolFn] = {
    "music21": music21,
    "hum2xml": hum2xml,
    "homr": _homr_tool,
    "oemer": oemer,
    "smt": _smt_tool,
    "transcoda": transcoda,
}
