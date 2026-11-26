"""
A typed model of MusicXML, shared by generator and parser. The generator's builder
creates it from tokens, its passes complete it and writer.py writes it. reader.py
reads MusicXML into it for the parser. Notes know when they start, the writer adds
the backups and forwards. Fields marked "read" are only set by the reader.
"""

from dataclasses import dataclass, field


@dataclass
class Pitch:
    step: str
    octave: str
    alter: int | float | None = None  # a float is a microtone

    def key(self) -> str:
        return f"{self.step}|{'' if self.alter is None else self.alter}|{self.octave}"


@dataclass
class Slur:
    type: str
    number: int | None = None


@dataclass
class Note:
    staff: int
    voice: int
    pitch: Pitch | None = None  # None for a rest
    is_unpitched: bool = False  # read: neither pitch nor rest, e.g. percussion
    is_hidden: bool = False  # read: not printed, or without notehead
    has_notations: bool = False  # read: has a printed <notations>
    is_chord: bool = False
    is_grace: bool = False
    is_measure_rest: bool = False
    duration: int | None = None
    type: str | None = None
    dots: int = 0
    accidental: str | None = None
    time_modification: tuple[int, int] | None = None
    # Position of the model, e.g. upper2. None for rests which fill a measure.
    position: str | None = None
    beam_state: str | None = None  # the primary beam as the model predicted it
    beams: list[tuple[int, str]] = field(default_factory=list)
    stem: str | None = None
    # The token doesn't tell single from two-note tremolos, scores mostly have single ones
    has_tremolo: bool = False
    # <notations>, in this order
    ties: list[str] = field(default_factory=list)
    tied: list[str] = field(default_factory=list)  # read: <tied>, the writer draws ties
    direct: list[str] = field(default_factory=list)  # e.g. fermata
    tuplets: list[str] = field(default_factory=list)
    articulations: list[str] = field(default_factory=list)
    ornaments: list[tuple[str, dict[str, str], str | None]] = field(default_factory=list)
    slurs: list[Slur] = field(default_factory=list)
    image_position: tuple[int, int] | None = None
    # Decoder confidence per branch: probability and margin to the runner-up
    confidence: dict[str, tuple[float, float]] | None = None
    start: int = 0  # in divisions from the start of the measure

    @property
    def end(self) -> int:
        return self.start + (self.duration or 0)

    @property
    def is_rest(self) -> bool:
        return self.pitch is None

    @property
    def is_second_voice(self) -> bool:
        return (self.position or "").endswith("2")


@dataclass
class Clef:
    staff: int
    sign: str
    line: str
    octave_change: int
    time: int = 0  # in divisions, like Note.start
    after_barline: bool = False  # printed before the barline, applies to the next measure


@dataclass
class Key:
    fifths: int
    staff: int | None = None  # None for all staves
    time: int = 0


@dataclass
class Time:
    beats: str
    beat_type: str
    time: int = 0


@dataclass
class MultipleRest:
    measures: int
    time: int = 0


Attribute = Clef | Key | Time | MultipleRest


@dataclass
class Dynamics:
    staff: int
    marks: list[tuple[str, str | None]]  # tag and text
    time: int = 0


@dataclass
class OctaveShift:
    staff: int
    type: str
    placement: str
    time: int = 0
    size: int = 8


@dataclass
class Metronome:
    per_minute: int
    tempo: int
    time: int = 0


@dataclass
class NewSystem:
    time: int = 0


@dataclass
class NewPage:
    time: int = 0


@dataclass
class BarStyle:
    style: str


@dataclass
class Ending:
    type: str
    number: int


@dataclass
class Repeat:
    direction: str


@dataclass
class Barline:
    location: str
    children: list[BarStyle | Ending | Repeat] = field(default_factory=list)
    time: int = 0


Item = Note | Attribute | Dynamics | OctaveShift | Metronome | NewSystem | NewPage | Barline


@dataclass
class Measure:
    number: int
    items: list[Item] = field(default_factory=list)
    # Added for a multi-measure rest, its voices aren't rebalanced
    is_rest_filler: bool = False
    divisions: int | None = None  # read: a change of divisions in this measure

    def notes(self) -> list[Note]:
        return [item for item in self.items if isinstance(item, Note)]

    def has_notes(self) -> bool:
        return any(isinstance(item, Note) for item in self.items)

    def chords(self) -> list[list[Note]]:
        """The notes, a chord as one list."""
        result: list[list[Note]] = []
        for note in self.notes():
            if note.is_chord and result:
                result[-1].append(note)
            else:
                result.append([note])
        return result

    def end(self) -> int:
        """Where the last note ends."""
        return max((note.end for note in self.notes()), default=0)

    def barline(self, location: str) -> Barline:
        for item in self.items:
            if isinstance(item, Barline) and item.location == location:
                return item
        barline = Barline(location)
        self.items.append(barline)
        return barline


@dataclass
class Part:
    id: str
    has_two_staves: bool
    measures: list[Measure] = field(default_factory=list)
    divisions: int = 0  # per quarter note
