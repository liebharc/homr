import unittest

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
