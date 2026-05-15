import unittest
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from homr.transformer.lift_conversion import to_agnostic_lift, from_agnostic_lift

class TestLiftConversion(unittest.TestCase):
    def test_semantic_to_agnostic_key_d(self):
        # Key of D (F# and C#)
        # Semantic: F#4, F#4, F4(natural), F4(natural), F#4
        # Note: Semantic natural is represented by 'empty' (_)
        
        symbols = [
            EncodedSymbol("keySignature_2"),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "F4", empty),
            EncodedSymbol("note_4", "F4", empty),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("barline"),
        ]
        
        agnostic = to_agnostic_lift(symbols)
        
        # 1. F#4 (#) -> lift is empty because it's in key signature
        # 2. F#4 (#) -> lift is empty because it's already F# in measure
        # 3. F4 (_) -> lift is natural because it's different from F# (key signature)
        # 4. F4 (_) -> lift is empty because it's already natural in measure
        # 5. F#4 (#) -> lift is sharp because it's different from natural (previous state)
        
        expected_lifts = [nonote, empty, empty, "natural", empty, "sharp", nonote]
        actual_lifts = [s.lift for s in agnostic]
        
        self.assertEqual(actual_lifts, expected_lifts)

    def test_round_trip_key_d(self):
        symbols = [
            EncodedSymbol("keySignature_2"),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "F4", empty),
            EncodedSymbol("note_4", "F4", empty),
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("barline"),
            # Next measure resets
            EncodedSymbol("note_4", "F4", "#"),
        ]
        
        agnostic = to_agnostic_lift(symbols)
        restored = from_agnostic_lift(agnostic)
        
        self.assertEqual([str(s) for s in restored], [str(s) for s in symbols])

    def test_agnostic_to_semantic_flats(self):
        # Key of F (Bb)
        # Semantic: Bb4, B4(natural), Bb4
        
        symbols = [
            EncodedSymbol("keySignature_-1"),
            EncodedSymbol("note_4", "B4", "b"),
            EncodedSymbol("note_4", "B4", empty),
            EncodedSymbol("note_4", "B4", "b"),
        ]
        
        agnostic = to_agnostic_lift(symbols)
        expected_lifts = [nonote, empty, "natural", "flat"]
        actual_lifts = [s.lift for s in agnostic]
        
        self.assertEqual(actual_lifts, expected_lifts)
        
        restored = from_agnostic_lift(agnostic)
        self.assertEqual([str(s) for s in restored], [str(s) for s in symbols])

if __name__ == "__main__":
    unittest.main()
