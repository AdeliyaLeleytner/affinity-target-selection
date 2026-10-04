import unittest

import numpy as np

from study.features import (AMINO_ACIDS, DescriptorRBF, MolecularKernel,
                            grouped_folds, molecule_features,
                            nested_group_subsets, protein_features,
                            tanimoto_kernel)


class Features(unittest.TestCase):
    def test_binary_tanimoto_rectangular_and_empty(self):
        x = np.array([[0, 0, 0], [1, 1, 0]])
        y = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]])
        np.testing.assert_allclose(tanimoto_kernel(x, y), [[1, 0, 0], [0, 1/3, 1]])
        with self.assertRaises(ValueError):
            tanimoto_kernel([[2, 0]])

    def test_scaling_is_frozen_at_training_fit(self):
        train = np.array([[0., 5.], [2., 5.]])
        fitted = DescriptorRBF.fit(train)
        np.testing.assert_array_equal(fitted.mean_, [1, 5])
        np.testing.assert_array_equal(fitted.scale_, [1, 1])
        np.testing.assert_array_equal(fitted.transform([[101., 7.]]), [[100, 2]])
        np.testing.assert_allclose(fitted.kernel(train), [[1, np.exp(-2)], [np.exp(-2), 1]])
        fitted.kernel([[1001., -999.]], train)
        np.testing.assert_array_equal(fitted.mean_, [1, 5])
        np.testing.assert_array_equal(fitted.scale_, [1, 1])
        np.testing.assert_array_equal(fitted.transform(train), [[-1, 0], [1, 0]])

    def test_combined_kernel_and_owned_training_rows(self):
        fps = np.array([[1, 0], [0, 1]])
        descriptors = np.array([[0.], [2.]])
        kernel = MolecularKernel.fit(fps, descriptors)
        expected = [[1, 0.5 * np.exp(-4)], [0.5 * np.exp(-4), 1]]
        np.testing.assert_allclose(kernel.against_train(fps, descriptors), expected)
        fps[:] = 0
        descriptors[:] = 1000
        np.testing.assert_array_equal(kernel.train_fingerprints, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(kernel.train_descriptors, [[0], [2]])

    def test_acyclic_grouping_identity_and_chirality(self):
        features = molecule_features(["CCO", "OCC", "CCC", "Cc1ccccc1", "Oc1ccccc1",
                                      "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"])
        self.assertEqual(features.fingerprints.shape, (7, 2048))
        self.assertEqual(features.descriptors.shape, (7, 6))
        self.assertEqual(features.identities[0], features.identities[1])
        self.assertEqual(features.scaffold_groups[0], features.scaffold_groups[1])
        self.assertNotEqual(features.scaffold_groups[0], features.scaffold_groups[2])
        self.assertEqual(features.scaffolds[0], "")
        self.assertEqual(features.scaffolds[3], "c1ccccc1")
        self.assertEqual(features.scaffold_groups[3], features.scaffold_groups[4])
        self.assertNotEqual(features.identities[5], features.identities[6])
        self.assertFalse(np.array_equal(features.fingerprints[5], features.fingerprints[6]))
        np.testing.assert_array_equal(features.fingerprints[0], features.fingerprints[1])
        self.assertAlmostEqual(features.descriptors[0, 0], 46.069, places=3)
        np.testing.assert_array_equal(features.descriptors[0, 3:5], [1, 1])

    def test_bad_smiles_raise(self):
        with self.assertRaises(ValueError):
            molecule_features("CCO")
        for smiles in ["not_a_molecule", "", " ", None]:
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                molecule_features([smiles])

    def test_protein_normalization_and_direction(self):
        got = protein_features(["A C A", "a"])
        a, c = AMINO_ACIDS.index("A"), AMINO_ACIDS.index("C")
        self.assertAlmostEqual(got[0, a], 2 / 3)
        self.assertAlmostEqual(got[0, c], 1 / 3)
        self.assertEqual(got[0, 20 + 20 * a + c], 0.5)
        self.assertEqual(got[0, 20 + 20 * c + a], 0.5)
        np.testing.assert_allclose(got[:, :20].sum(axis=1), [1, 1])
        np.testing.assert_allclose(got[:, 20:].sum(axis=1), [1, 0])
        for invalid in ["", "AX", "A*"]:
            with self.assertRaises(ValueError):
                protein_features([invalid])

    def test_folds_preserve_groups_and_input_permutations(self):
        groups = np.array(["a", "a", "b", "c", "c", "d", "e", "f", "g"])
        folds = grouped_folds(groups, seed=13)
        self.assertEqual(set(folds), set(range(5)))
        for group in set(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)
        permutation = np.array([7, 2, 0, 8, 1, 5, 6, 4, 3])
        permuted = grouped_folds(groups[permutation], seed=13)
        np.testing.assert_array_equal(permuted, folds[permutation])
        for fold in range(5):
            self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))
        with self.assertRaises(ValueError):
            grouped_folds(["a", "a", "b"])

    def test_nested_training_group_budgets(self):
        groups = np.array(["heldout", "a", "a", "b", "c", "c", "d", "heldout"])
        train = np.flatnonzero(groups != "heldout")
        subsets = nested_group_subsets(groups[train], [0, 1, 2, 4], seed=13)
        self.assertEqual(len(subsets[0]), 0)
        for budget, subset in zip([0, 1, 2, 4], subsets):
            selected = train[subset]
            self.assertEqual(len(set(groups[selected])), budget)
            self.assertNotIn("heldout", groups[selected])
            for group in set(groups[selected]):
                self.assertEqual(set(selected[groups[selected] == group]),
                                 set(np.flatnonzero(groups == group)))
        for smaller, larger in zip(subsets, subsets[1:]):
            self.assertTrue(set(smaller).issubset(larger))
        np.testing.assert_array_equal(subsets[-1], np.arange(len(train)))
        repeat = nested_group_subsets(groups[train], [0, 1, 2, 4], seed=13)
        for left, right in zip(subsets, repeat):
            np.testing.assert_array_equal(left, right)
        permutation = np.array([5, 2, 0, 4, 1, 3])
        reordered = nested_group_subsets(groups[train][permutation], [0, 1, 2, 4], seed=13)
        for subset, shuffled in zip(subsets, reordered):
            self.assertEqual(set(groups[train][subset]),
                             set(groups[train][permutation][shuffled]))
        with self.assertRaises(ValueError):
            nested_group_subsets(groups[train], [3, 2])


if __name__ == "__main__":
    unittest.main()
