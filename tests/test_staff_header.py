import unittest

from homr.transformer.staff_header import HeaderGate, StaffContext, update_context
from homr.transformer.vocabulary import EncodedSymbol

SURE = 0.95
UNSURE = 0.5


class TestStaffHeader(unittest.TestCase):
    def test_update_context_keeps_last_clef_and_key(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("clef_F4", position="lower"),
            EncodedSymbol("keySignature_2"),
            EncodedSymbol("note_4", "C4", position="upper"),
            EncodedSymbol("clef_C3", position="upper"),
        ]
        context = update_context(StaffContext({"upper": "clef_G1"}, "keySignature_1"), symbols)
        self.assertEqual(context.clefs, {"upper": "clef_C3", "lower": "clef_F4"})
        self.assertEqual(context.key, "keySignature_2")

    def test_no_context_keeps_the_model(self) -> None:
        gate = HeaderGate(None)
        self.assertIsNone(gate.replacement("clef_F4", "upper", UNSURE))

    def test_unsure_clef_and_key_are_carried(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertEqual(gate.replacement("clef_F4", "upper", UNSURE), "clef_G2")
        self.assertEqual(gate.replacement("keySignature_0", "upper", UNSURE), "keySignature_-2")

    def test_key_is_carried_if_the_unsure_clef_matches(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertIsNone(gate.replacement("clef_G2", "upper", UNSURE))
        self.assertEqual(gate.replacement("keySignature_0", "upper", UNSURE), "keySignature_-2")

    def test_key_stays_after_a_sure_clef(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertIsNone(gate.replacement("clef_G2", "upper", SURE))
        self.assertIsNone(gate.replacement("keySignature_0", "upper", UNSURE))

    def test_key_is_carried_after_an_unsure_clef_even_if_the_model_is_sure(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertEqual(gate.replacement("clef_F4", "upper", UNSURE), "clef_G2")
        self.assertEqual(gate.replacement("keySignature_0", "upper", SURE), "keySignature_-2")

    def test_sure_symbols_stay(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertIsNone(gate.replacement("clef_F4", "upper", SURE))
        self.assertIsNone(gate.replacement("keySignature_0", "upper", SURE))

    def test_lower_clef_uses_the_lower_context(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2", "lower": "clef_F4"}))
        self.assertIsNone(gate.replacement("clef_G2", "upper", UNSURE))
        self.assertIsNone(gate.replacement("chord", "", SURE))
        self.assertEqual(gate.replacement("clef_G2", "lower", UNSURE), "clef_F4")

    def test_only_the_staff_start_is_changed(self) -> None:
        gate = HeaderGate(StaffContext({"upper": "clef_G2"}, "keySignature_-2"))
        self.assertIsNone(gate.replacement("note_4", "upper", SURE))
        self.assertIsNone(gate.replacement("clef_F4", "upper", UNSURE))
        self.assertIsNone(gate.replacement("keySignature_0", "upper", UNSURE))
