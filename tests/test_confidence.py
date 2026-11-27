# ruff: noqa: S101
import re
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np

from homr.main import main
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import ScoreDecoder, head_confidence
from training.transformer.training_vocabulary import read_token_lines

TOKENS = """clef_G2 . . . . upper
note_4 E4 _ _ _ upper
note_4 C4 _ _ _ upper
barline . . . . .""".splitlines()

CONFIDENCE_COMMENT = re.compile(
    r" conf: ([rplaso]:[01]\.\d{4}:[01]\.\d{4};)*[rplaso]:[01]\.\d{4}:[01]\.\d{4} "
)


def _confidence_comments(xml: ET.Element) -> list[list[str | None]]:
    return [
        [c.text for c in note if c.tag is ET.Comment and (c.text or "").startswith(" conf:")]  # type: ignore[comparison-overlap]
        for note in xml.iter("note")
    ]


class TestHeadConfidence(unittest.TestCase):
    def test_probability_and_margin(self) -> None:
        probability, margin = head_confidence(np.log(np.array([[0.6, 0.3, 0.1]])))
        self.assertAlmostEqual(probability, 0.6)
        self.assertAlmostEqual(margin, 0.3)

    def test_is_stable_for_large_logits_and_float16(self) -> None:
        probability, margin = head_confidence(np.array([[1000.0, 0.0]], dtype=np.float16))
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(margin, 1.0)

    def test_single_token(self) -> None:
        self.assertEqual(head_confidence(np.array([[5.0]])), (1.0, 1.0))


class TestConfidenceInMusicXml(unittest.TestCase):
    def test_no_comment_without_confidence(self) -> None:
        xml = generate_xml(XmlGeneratorArguments(), [read_token_lines(TOKENS)], "")
        self.assertNotIn("conf:", ET.tostring(xml, encoding="unicode"))

    def test_comment_with_confidence(self) -> None:
        tokens = read_token_lines(TOKENS)
        tokens[1].confidence = {
            "r": (0.99, 0.98),
            "p": (0.5, 0.25),
            "l": (1.0, 1.0),
            "a": (1.0, 1.0),
            "s": (1.0, 1.0),
            "o": (1.0, 1.0),
        }
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        comments = _confidence_comments(xml)
        self.assertEqual([len(c) for c in comments], [1, 0])
        text = comments[0][0] or ""
        self.assertRegex(text, CONFIDENCE_COMMENT)
        parsed = {
            head: (float(p), float(margin))
            for head, p, margin in (
                entry.split(":") for entry in text.removeprefix(" conf: ").strip().split(";")
            )
        }
        self.assertEqual(list(parsed), ["r", "p", "l", "a", "s", "o"])
        self.assertEqual(parsed["p"], (0.5, 0.25))
        for probability, margin in parsed.values():
            self.assertTrue(0.0 <= probability <= 1.0)
            self.assertTrue(0.0 <= margin <= probability)


def _fake_decoder(output_confidence: bool) -> tuple[ScoreDecoder, int]:
    """A decoder which emits one note and then the end token, without model weights."""
    config = Config()
    config.output_confidence = output_confidence
    session = mock.MagicMock()
    cache = [mock.MagicMock() for _ in range(config.decoder_depth * 4)]

    def logits(size: int, token: int, peak: float, runner_up: int | None = None) -> mock.MagicMock:
        values = np.zeros((1, 1, size), dtype=np.float32)
        values[0, 0, token] = peak
        if runner_up is not None:
            values[0, 0, runner_up] = peak - 1
        output = mock.MagicMock()
        output.numpy.return_value = values
        return output

    def step(rhythm: int) -> list[mock.MagicMock]:
        return [
            logits(config.num_rhythm_tokens, rhythm, 4.0, runner_up=rhythm + 1),
            logits(config.num_pitch_tokens, 3, 4.0),
            logits(config.num_lift_tokens, 3, 4.0),
            logits(config.num_position_tokens, 3, 4.0),
            logits(config.num_articulation_tokens, 3, 4.0),
            logits(config.num_slur_tokens, 3, 4.0),
            mock.MagicMock(numpy=mock.MagicMock(return_value=np.zeros((1, 2)))),
            *cache,
        ]

    session.io_binding.return_value.get_outputs.side_effect = [step(5), step(config.eos_token)]
    return ScoreDecoder(session, fp16=False, use_gpu=False, config=config), 5


class TestDecoderConfidence(unittest.TestCase):
    def generate(self, output_confidence: bool) -> list:
        decoder, _ = _fake_decoder(output_confidence)
        start = np.array([[1]], dtype=np.int64)
        nonote = np.array([[0]], dtype=np.int64)
        context = np.zeros((1, 4, 512), dtype=np.float32)
        return decoder.generate(start, nonote, context=context)

    def test_is_off_by_default(self) -> None:
        self.assertFalse(Config().output_confidence)
        symbols = self.generate(False)
        self.assertEqual(len(symbols), 1)
        self.assertIsNone(symbols[0].confidence)

    def test_values_are_probabilities_and_the_choice_stays_the_same(self) -> None:
        without = self.generate(False)
        with_confidence = self.generate(True)
        self.assertEqual(
            [(s.rhythm, s.pitch, s.lift, s.articulation, s.slur, s.position) for s in without],
            [
                (s.rhythm, s.pitch, s.lift, s.articulation, s.slur, s.position)
                for s in with_confidence
            ],
        )
        confidence = with_confidence[0].confidence
        assert confidence is not None
        self.assertEqual(list(confidence), ["r", "p", "l", "a", "s", "o"])
        for probability, margin in confidence.values():
            self.assertTrue(0.0 <= probability <= 1.0)
            self.assertTrue(0.0 <= margin <= probability)
        # The rhythm logits have a runner-up one below the peak, the other heads don't
        self.assertLess(confidence["r"][1], confidence["p"][1])


class TestCommandLine(unittest.TestCase):
    def run_main(self, *flags: str) -> mock.MagicMock:
        with (
            mock.patch("sys.argv", ["homr", "--no-title", *flags, __file__]),
            mock.patch("homr.main.download_weights"),
            mock.patch("homr.main.run_homr") as run_homr,
        ):
            main()
        return run_homr

    def test_flag_is_opt_in(self) -> None:
        self.assertFalse(self.run_main().call_args.args[1].output_confidence)
        self.assertTrue(self.run_main("--output-confidence").call_args.args[1].output_confidence)
