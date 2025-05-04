import unittest

from homr.transformer.configs import default_config
from training.transformer.kern_tokens import split_symbol_into_token


class TestTransformerConfig(unittest.TestCase):
    """
    The definition of what is a rest or note must match between the
    transformer config and how prepare the expected tokens for a
    training run.
    """

    def test_note_and_rest_indices(self) -> None:
        config = default_config
        self.assertEqual(len(config.noteindexes), 70)

    def test_split_symbol_into_token(self) -> None:
        param_list = [
            ("**kern", "nonote"),
            ("*clefF4", "nonote"),
            ("*k[]", "nonote"),
            ("*M2/4", "nonote"),
            ("4c", "note"),
            ("4r", "note"),
            ("qc", "note"),
            ("4ddn", "note"),
            ("2e-", "note"),
            ("4dd-", "note"),
            ("=7", "nonote"),
            ("==", "nonote"),
            (".", "nonote"),
        ]
        for token, note in param_list:
            with self.subTest(token):
                tokens = split_symbol_into_token(token)
                self.assertEqual(tokens[0], note)
