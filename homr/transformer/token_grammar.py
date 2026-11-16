"""
Which values of the decoder heads may go together with a rhythm token.

The heads predict independently, so their argmaxes can disagree, e.g. a rest with a
pitch. The grammar follows the training tokens:
- notes: a pitch, no "." in any head
- rests, clefs, dynamics, octave shifts: pitch and lift "_", no "." in any head,
  clefs and octave shifts also articulation "_", only rests can have a slur (no beam)
- everything else (bar lines, signatures, chord, ...): "." in every head
"""

import numpy as np

from homr.transformer.configs import Config
from homr.transformer.vocabulary import BEAMS, empty, nonote
from homr.type_definitions import NDArray

HEADS = ("rhythm", "pitch", "lift", "position", "articulation", "slur")
ATTRIBUTES = HEADS[1:]
KINDS = ("note", "rest", "clef", "dynamic", "octaveShift", "other")
SPECIAL = ("PAD", "BOS", "EOS")

Token = dict[str, int]
LogProbs = dict[str, NDArray]


def kind_of(rhythm: str) -> str:
    for kind in KINDS[:-1]:
        if rhythm.startswith(kind):
            return kind
    return "other"


def _allowed(kind: str, head: str, value: str) -> bool:
    if kind == "other":
        return value == nonote
    if value == nonote:
        return False
    if kind == "note":
        return head != "pitch" or value != empty
    if head in ("pitch", "lift"):
        return value == empty
    if head == "articulation" and kind in ("clef", "octaveShift"):
        return value == empty
    if head == "slur":
        if kind == "rest":
            return not any(beam in value for beam in BEAMS)
        return value == empty
    return True


class TokenGrammar:
    def __init__(self, config: Config):
        vocabs = {
            "pitch": config.pitch_vocab,
            "lift": config.lift_vocab,
            "position": config.position_vocab,
            "articulation": config.articulation_vocab,
            "slur": config.slur_vocab,
        }
        self.kind = np.array([KINDS.index(kind_of(r)) for r in config.rhythm_vocab])
        # mask[head][kind]: -inf for values which can't go with the kind, else 0
        self.mask = {
            head: np.array(
                [[0.0 if _allowed(kind, head, v) else -np.inf for v in vocab] for kind in KINDS]
            )
            for head, vocab in vocabs.items()
        }
        self.never = np.array([config.rhythm_vocab[s] for s in SPECIAL])

    def _mask(self, log_probs: LogProbs, head: str) -> NDArray:
        # A model can be older than the vocabulary, new tokens are appended last
        return self.mask[head][:, : len(log_probs[head])]

    def rhythm_scores(self, log_probs: LogProbs) -> NDArray:
        """
        Log-prob of the best valid token for every rhythm. PAD, BOS and EOS get -inf,
        the decoder decides on EOS from the rhythm head alone.
        """
        kind_best = sum(
            (log_probs[h][None, :] + self._mask(log_probs, h)).max(axis=1) for h in ATTRIBUTES
        )
        scores = log_probs["rhythm"] + kind_best[self.kind[: len(log_probs["rhythm"])]]
        scores[self.never] = -np.inf
        return scores

    def token(self, log_probs: LogProbs, rhythm: int) -> Token:
        """The best valid token with this rhythm."""
        kind = self.kind[rhythm]
        result = {"rhythm": rhythm}
        for h in ATTRIBUTES:
            result[h] = int((log_probs[h] + self._mask(log_probs, h)[kind]).argmax())
        return result

    def best(self, log_probs: LogProbs) -> Token:
        return self.token(log_probs, int(self.rhythm_scores(log_probs).argmax()))
