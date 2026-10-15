import unittest

from homr.transformer.vocabulary import EncodedSymbol
from homr.triplet_restoration import restore_unmarked_triplets
from training.transformer.training_vocabulary import read_token_lines

HEADER = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . ."""

BARLINE = "barline . . . . ."

# Two beats: three eighths in the right hand against one quarter in the left, twice.
THREES_AGAINST_QUARTERS = """note_8 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_8 D5 _ _ _ upper
note_8 E5 _ _ _ upper
note_8 F5 _ _ _ upper&note_4 G2 _ _ _ lower
note_8 E5 _ _ _ upper
note_8 D5 _ _ _ upper
barline . . . . ."""


# Two beats that add up as they are: two eighths in the right hand against each quarter in the left.
PAIRS_AGAINST_QUARTERS = """note_8 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_8 D5 _ _ _ upper
note_8 E5 _ _ _ upper&note_4 G2 _ _ _ lower
note_8 F5 _ _ _ upper
barline . . . . ."""

# As THREES_AGAINST_QUARTERS, with the hands swapped.
THREES_IN_THE_LEFT_HAND = """note_4 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_8 E3 _ _ _ lower
note_8 G3 _ _ _ lower
note_4 D5 _ _ _ upper&note_8 B2 _ _ _ lower
note_8 D3 _ _ _ lower
note_8 G3 _ _ _ lower
barline . . . . ."""


# As THREES_AGAINST_QUARTERS, with a second voice in the right hand moving along in thirds.
THREES_IN_TWO_VOICES = """note_8 E5 _ _ _ upper&note_8 C5 _ _ _ upper2&note_4 C3 _ _ _ lower
note_8 F5 _ _ _ upper&note_8 D5 _ _ _ upper2
note_8 G5 _ _ _ upper&note_8 E5 _ _ _ upper2
note_8 A5 _ _ _ upper&note_8 F5 _ _ _ upper2&note_4 G2 _ _ _ lower
note_8 G5 _ _ _ upper&note_8 E5 _ _ _ upper2
note_8 F5 _ _ _ upper&note_8 D5 _ _ _ upper2
barline . . . . ."""


# Six eighths in the right hand against one half note in the left: one triplet alone would not
# make the hands meet, two do.
SIX_AGAINST_A_HALF = """note_8 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_8 D5 _ _ _ upper
note_8 E5 _ _ _ upper
note_8 F5 _ _ _ upper
note_8 E5 _ _ _ upper
note_8 D5 _ _ _ upper
barline . . . . ."""


# The right hand on its own: nothing in the measure says how long its six eighths take.
SIX_ALONE = """note_8 C5 _ _ _ upper
note_8 D5 _ _ _ upper
note_8 E5 _ _ _ upper
note_8 F5 _ _ _ upper
note_8 E5 _ _ _ upper
note_8 D5 _ _ _ upper
barline . . . . ."""


# Both hands play eighths together: the measure makes sense as it is, and it would as triplets too.
EIGHTHS_IN_BOTH_HANDS = """note_8 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_8 D5 _ _ _ upper&note_8 D3 _ _ _ lower
note_8 E5 _ _ _ upper&note_8 E3 _ _ _ lower
note_8 F5 _ _ _ upper&note_8 F3 _ _ _ lower
note_8 E5 _ _ _ upper&note_8 E3 _ _ _ lower
note_8 D5 _ _ _ upper&note_8 D3 _ _ _ lower
barline . . . . ."""


# As THREES_AGAINST_QUARTERS, but the right hand is silent.
THREE_RESTS_AGAINST_QUARTERS = """rest_8 _ _ _ _ upper&note_4 C3 _ _ _ lower
rest_8 _ _ _ _ upper
rest_8 _ _ _ _ upper
rest_8 _ _ _ _ upper&note_4 G2 _ _ _ lower
rest_8 _ _ _ _ upper
rest_8 _ _ _ _ upper
barline . . . . ."""


def _voice(*parts: str) -> list[EncodedSymbol]:
    return read_token_lines("\n".join(parts).splitlines())


def _rhythms(symbols: list[EncodedSymbol], position: str) -> list[str]:
    return [
        s.rhythm
        for s in symbols
        if s.position == position and s.rhythm.startswith(("note", "rest"))
    ]


class TestTripletRestoration(unittest.TestCase):
    def test_three_equal_notes_against_one_beat_become_a_triplet(self) -> None:
        voice = _voice(HEADER, *[THREES_AGAINST_QUARTERS] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 18)
        self.assertEqual(_rhythms(result, "lower"), ["note_4"] * 6)

    def test_notes_that_fill_their_beats_are_left_alone(self) -> None:
        voice = _voice(HEADER, *[PAIRS_AGAINST_QUARTERS] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_8"] * 12)
        self.assertEqual(_rhythms(result, "lower"), ["note_4"] * 6)

    def test_triplets_are_found_in_either_hand(self) -> None:
        voice = _voice(HEADER, *[THREES_IN_THE_LEFT_HAND] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_4"] * 6)
        self.assertEqual(_rhythms(result, "lower"), ["note_12"] * 18)

    def test_voices_which_move_together_become_triplets_together(self) -> None:
        voice = _voice(HEADER, *[THREES_IN_TWO_VOICES] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 18)
        self.assertEqual(_rhythms(result, "upper2"), ["note_12"] * 18)
        self.assertEqual(_rhythms(result, "lower"), ["note_4"] * 6)

    def test_threes_in_a_row_against_one_long_note_become_triplets_together(self) -> None:
        voice = _voice(HEADER, *[SIX_AGAINST_A_HALF] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 18)
        self.assertEqual(_rhythms(result, "lower"), ["note_2"] * 3)

    def test_a_page_with_triplets_in_only_two_places_is_left_alone(self) -> None:
        # One or two odd measures are more likely misread in some other way
        voice = _voice(
            HEADER,
            THREES_AGAINST_QUARTERS,
            PAIRS_AGAINST_QUARTERS,
            THREES_AGAINST_QUARTERS,
            PAIRS_AGAINST_QUARTERS,
        )

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_8"] * 20)

    def test_on_a_page_in_triplets_a_measure_which_makes_sense_is_left_alone(self) -> None:
        # Nothing in the measure says how long the six eighths take, so they stay as read
        voice = _voice(HEADER, *[THREES_AGAINST_QUARTERS] * 6, SIX_ALONE)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 36 + ["note_8"] * 6)

    def test_on_a_page_in_triplets_hands_moving_together_are_left_alone(self) -> None:
        voice = _voice(HEADER, *[THREES_AGAINST_QUARTERS] * 6, EIGHTHS_IN_BOTH_HANDS)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 36 + ["note_8"] * 6)
        self.assertEqual(_rhythms(result, "lower"), ["note_4"] * 12 + ["note_8"] * 6)

    def test_on_a_page_in_triplets_notes_that_fill_their_beats_are_still_left_alone(self) -> None:
        voice = _voice(HEADER, *[THREES_AGAINST_QUARTERS] * 6, PAIRS_AGAINST_QUARTERS)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_12"] * 36 + ["note_8"] * 4)

    def test_music_without_barlines_is_left_alone(self) -> None:
        # Without barlines there are no measures to make sense of
        without_barline = THREES_AGAINST_QUARTERS.removesuffix("\n" + BARLINE)
        voice = _voice(HEADER, *[without_barline] * 6, BARLINE)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["note_8"] * 36)

    def test_three_rests_are_not_a_triplet(self) -> None:
        voice = _voice(HEADER, *[THREE_RESTS_AGAINST_QUARTERS] * 3)

        result = restore_unmarked_triplets(voice)

        self.assertEqual(_rhythms(result, "upper"), ["rest_8"] * 18)
