import unittest

import numpy as np

from homr.transformer.configs import Config
from homr.transformer.token_grammar import HEADS, TokenGrammar


class TestTokenGrammar(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.grammar = TokenGrammar(self.config)
        self.vocab = {
            "rhythm": self.config.rhythm_vocab,
            "pitch": self.config.pitch_vocab,
            "lift": self.config.lift_vocab,
            "position": self.config.position_vocab,
            "articulation": self.config.articulation_vocab,
            "slur": self.config.slur_vocab,
        }

    def log_probs(self, **preferred: str) -> dict[str, np.ndarray]:
        result = {h: np.full(len(self.vocab[h]), -10.0) for h in HEADS}
        for head, value in preferred.items():
            result[head][self.vocab[head][value]] = -0.1
        return result

    def decode(self, log_probs: dict[str, np.ndarray]) -> dict[str, str]:
        token = self.grammar.best(log_probs)
        inverse = {h: {v: k for k, v in self.vocab[h].items()} for h in HEADS}
        return {h: inverse[h][token[h]] for h in HEADS}

    def test_rest_gets_no_pitch(self) -> None:
        lp = self.log_probs(
            rhythm="rest_4", pitch="C4", lift="#", position="upper", articulation="_", slur="_"
        )
        lp["pitch"][self.vocab["pitch"]["_"]] = -1.0
        lp["lift"][self.vocab["lift"]["_"]] = -1.0
        token = self.decode(lp)
        self.assertEqual(token["rhythm"], "rest_4")
        self.assertEqual(token["pitch"], "_")
        self.assertEqual(token["lift"], "_")

    def test_note_gets_a_pitch(self) -> None:
        lp = self.log_probs(
            rhythm="note_4", pitch="_", lift="_", position="upper", articulation="_", slur="_"
        )
        lp["pitch"][self.vocab["pitch"]["E4"]] = -1.0
        self.assertEqual(self.decode(lp)["pitch"], "E4")

    def test_barline_has_no_attributes(self) -> None:
        lp = self.log_probs(
            rhythm="barline", pitch="C4", lift="_", position="upper", articulation="_", slur="_"
        )
        for head in HEADS[1:]:
            lp[head][self.vocab[head]["."]] = -2.0
        token = self.decode(lp)
        self.assertEqual(token["rhythm"], "barline")
        self.assertTrue(all(token[h] == "." for h in HEADS[1:]))

    def test_rhythm_decided_jointly(self) -> None:
        # barline is slightly more likely, but its other heads are very unlikely
        lp = self.log_probs(
            rhythm="barline", pitch="C4", lift="_", position="upper", articulation="_", slur="_"
        )
        lp["rhythm"][self.vocab["rhythm"]["note_4"]] = -0.2
        self.assertEqual(self.decode(lp)["rhythm"], "note_4")

    def test_special_tokens_are_never_chosen(self) -> None:
        lp = self.log_probs(rhythm="BOS")
        lp["rhythm"][self.vocab["rhythm"]["PAD"]] = -0.1
        lp["rhythm"][self.vocab["rhythm"]["EOS"]] = -0.1
        lp["rhythm"][self.vocab["rhythm"]["barline"]] = -5.0
        self.assertEqual(self.decode(lp)["rhythm"], "barline")

    def test_model_older_than_vocabulary(self) -> None:
        lp = self.log_probs(
            rhythm="note_4", pitch="C4", lift="_", position="upper", articulation="_", slur="_"
        )
        lp["articulation"] = lp["articulation"][:-3]
        self.assertEqual(self.decode(lp)["articulation"], "_")
