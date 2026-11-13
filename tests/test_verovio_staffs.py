# ruff: noqa: S101, S314
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.verovio_staffs import (
    VEROVIO_SHARE_GRAND_STAFF,
    VEROVIO_SHARE_SINGLE_STAFF,
    _misplaced_clefs,
    choose_parts,
    clef_changes_within_measures,
    fill_voices,
    render,
)


def _note(step: str, duration: int, voice: int, staff: int = 1, octave: int = 4) -> str:
    note_type = {4: "half", 2: "quarter"}[duration]
    return f"""<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>
      <duration>{duration}</duration><voice>{voice}</voice><type>{note_type}</type>
      <staff>{staff}</staff></note>"""


def _score(*measures: str, staves: int = 1) -> str:
    clefs = '<clef number="1"><sign>G</sign><line>2</line></clef>'
    if staves == 2:  # noqa: PLR2004
        clefs += '<clef number="2"><sign>F</sign><line>4</line></clef>'
    first = f"""<attributes><divisions>2</divisions><key><fifths>0</fifths></key>
      <time><beats>4</beats><beat-type>4</beat-type></time><staves>{staves}</staves>
      {clefs}</attributes>"""
    body = "".join(
        f'<measure number="{i + 1}">{first if i == 0 else ""}{m}</measure>'
        for i, m in enumerate(measures)
    )
    return f"""<score-partwise version="4.0"><part-list>
      <score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
      <part id="P1">{body}</part></score-partwise>"""


def test_choose_parts_is_stable_and_has_the_shares() -> None:
    assert choose_parts("lc123", [1, 2]) == choose_parts("lc123", [1, 2])
    names = [f"piece{i}" for i in range(4000)]
    single = sum(0 in choose_parts(n, [1]) for n in names) / len(names)
    grand = sum(0 in choose_parts(n, [2]) for n in names) / len(names)
    assert abs(single - VEROVIO_SHARE_SINGLE_STAFF) < 0.03  # noqa: PLR2004
    assert abs(grand - VEROVIO_SHARE_GRAND_STAFF) < 0.03  # noqa: PLR2004


def test_fill_voices_pads_a_short_voice() -> None:
    measure = _note("C", 4, 1) + _note(
        "D", 4, 1
    ) + "<backup><duration>8</duration></backup>" "" + _note("E", 2, 2)
    root = ET.fromstring(_score(measure))
    fill_voices(root)
    forwards = root.findall("part/measure/forward")
    assert len(forwards) == 1
    assert forwards[0].findtext("duration") == "6"
    assert forwards[0].findtext("voice") == "2"
    assert forwards[0].findtext("staff") == "1"


def test_fill_voices_leaves_cross_staff_voices() -> None:
    measure = (
        _note("C", 4, 1)
        + _note("D", 2, 1)
        + _note("E", 2, 1, staff=2, octave=3)
        + "<backup><duration>8</duration></backup>"
        + _note("C", 4, 5, staff=2, octave=3)
        + _note("D", 4, 5, staff=2, octave=3)
    )
    root = ET.fromstring(_score(measure, staves=2))
    fill_voices(root)
    assert root.findall("part/measure/forward") == []


def test_clef_changes_within_measures() -> None:
    clef = '<attributes><clef number="1"><sign>F</sign><line>4</line></clef></attributes>'
    root = ET.fromstring(
        _score(
            _note("C", 4, 1) + clef + _note("D", 4, 1, octave=3),
            clef + _note("C", 4, 1, octave=3) + _note("D", 4, 1, octave=3),
        )
    )
    assert clef_changes_within_measures(root) == {(0, 0): 1}


def test_misplaced_clefs() -> None:
    svg = """<g xmlns="http://www.w3.org/2000/svg" class="staff">
      <g class="layer">
        <g class="note"><use transform="translate(100, 0)"/></g>
        <g class="note"><use transform="translate(300, 0)"/></g>
      </g>
      <g class="layer">
        <g class="note"><use transform="translate(120, 0)"/></g>
        <g class="clef"><use transform="translate(150, 0)"/></g>
      </g>
      <g class="layer"><g class="clef"><use transform="translate(400, 0)"/></g></g>
    </g>"""
    assert _misplaced_clefs(ET.fromstring(svg)) == 1


def test_render_finds_bars_and_staves(tmp_path: Path) -> None:
    measure = _note("C", 4, 1) + _note("E", 4, 1)
    lower = _note("C", 4, 5, staff=2, octave=3) + _note("G", 4, 5, staff=2, octave=2)
    full = measure + "<backup><duration>8</duration></backup>" + lower
    file = tmp_path / "piece.musicxml"
    file.write_text(_score(full, full, full, staves=2))
    score = render(str(file), "piece")
    assert sum(s.bars for s in score.systems) == 3  # noqa: PLR2004
    for system in score.systems:
        assert len(system.staffs) == 2  # noqa: PLR2004
        assert not system.misplaced_clefs
        upper, lower_staff = system.staffs
        assert upper.line_bottom < lower_staff.line_top
        assert upper.top <= upper.line_top
        assert lower_staff.bottom > lower_staff.line_bottom  # the G2 is below the lines
        merged = upper.merge(lower_staff)
        assert merged.staff_space == upper.staff_space
        assert abs(upper.staff_space * 4 - (upper.line_bottom - upper.line_top)) < 1e-6
