import unittest

from training.transformer.data_set_filters import (
    contains_supported_clef,
    contains_supported_number_of_kerns,
)
from training.transformer.mix_datasets import mix_training_sets


class TestMixDataSets(unittest.TestCase):

    def test_mix_training_sets(self):
        dataset1 = ["a", "b", "c", "d", "e", "f", "g"]
        dataset2 = ["1", "2", "3", "5", "6"]
        dataset3 = ["x", "y", "z"]
        actual = mix_training_sets([dataset1, dataset2, dataset3], [0.5, 1.0, 1.0], 10)
        self.assertEqual(len(actual), 10)
        for element in dataset2:
            self.assertTrue(element in actual)
        for element in dataset3:
            self.assertTrue(element in actual)

    def test_contains_supported_clef(self):
        self.assertTrue(contains_supported_clef("32BB-JJk	32F 32B- 32d-JJk"))
        self.assertTrue(contains_supported_clef(""))
        self.assertTrue(contains_supported_clef("*clefF4	*clefG2"))
        self.assertTrue(contains_supported_clef("*clefF4	*clefF4"))
        self.assertTrue(contains_supported_clef("*clefG2	*clefG2"))
        self.assertTrue(contains_supported_clef("*clefF4	*"))
        self.assertTrue(contains_supported_clef("*clefF4"))
        self.assertTrue(contains_supported_clef("*clefG2"))
        self.assertFalse(contains_supported_clef("*clefC2"))
        self.assertFalse(contains_supported_clef("*clefG2	*clefC2"))

    def test_contains_supported_number_of_kerns(self):
        self.assertTrue(contains_supported_number_of_kerns("32BB-JJk	32F 32B- 32d-JJk"))
        self.assertTrue(contains_supported_number_of_kerns(""))
        self.assertTrue(contains_supported_number_of_kerns("**kern"))
        self.assertTrue(contains_supported_number_of_kerns("**kern	**kern"))
        self.assertFalse(contains_supported_number_of_kerns("**kern	**kern   **kern"))
