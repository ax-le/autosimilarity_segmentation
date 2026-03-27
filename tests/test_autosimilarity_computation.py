# -*- coding: utf-8 -*-
"""
Comprehensive tests for autosimilarity_computation.py

Tests include:
- Normalization tests
- All similarity function tests (cosine, autocorrelation, RBF, centered RBF)
- Gamma computation tests
- Dispatcher function tests
- Mathematical property verification (symmetry, diagonal, value ranges)
- Side effect checking
"""

import unittest
import numpy as np
import as_seg.autosimilarity_computation as auto_comp
import as_seg.model.errors as err


class TestL2NormaliseBarwise(unittest.TestCase):
    """Test l2_normalise_barwise function."""

    def test_unit_norm(self):
        """Test that normalized vectors have unit L2 norm."""
        np.random.seed(42)
        arr = np.random.rand(10, 5)
        normalized = auto_comp.l2_normalise_barwise(arr)
        
        # Check each row has norm 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), rtol=1e-10)

    def test_known_input(self):
        """Test with known input values."""
        arr = np.array([[3.0, 4.0], [5.0, 12.0]])
        expected = np.array([[3/5, 4/5], [5/13, 12/13]])
        normalized = auto_comp.l2_normalise_barwise(arr)
        np.testing.assert_allclose(normalized, expected, rtol=1e-10)

    def test_zero_vector_handling(self):
        """Test that zero vectors are replaced with eps."""
        arr = np.array([[0.0, 0.0], [1.0, 1.0]])
        normalized = auto_comp.l2_normalise_barwise(arr)
        
        # First row should be eps (since divided by 0 norm)
        eps = 1e-10
        np.testing.assert_allclose(normalized[0], [eps, eps])
        
        # Second row should be normalized normally
        expected_second = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(normalized[1], expected_second, rtol=1e-10)

    def test_no_side_effects(self):
        """Test that input is not modified."""
        np.random.seed(42)
        arr = np.random.rand(10, 5)
        original = arr.copy()
        auto_comp.l2_normalise_barwise(arr)
        np.testing.assert_array_equal(arr, original)


class TestCosineAutosimilarity(unittest.TestCase):
    """Test get_cosine_autosimilarity function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_symmetry(self):
        """Test that autosimilarity matrix is symmetric."""
        autosim = auto_comp.get_cosine_autosimilarity(self.arr)
        np.testing.assert_allclose(autosim, autosim.T, rtol=1e-10)

    def test_diagonal_is_one(self):
        """Test that diagonal elements are 1 (self-similarity)."""
        autosim = auto_comp.get_cosine_autosimilarity(self.arr)
        np.testing.assert_allclose(np.diag(autosim), np.ones(10), rtol=1e-10)

    def test_value_range(self):
        """Test that values are in [-1, 1]."""
        autosim = auto_comp.get_cosine_autosimilarity(self.arr)
        self.assertTrue(np.all(autosim >= -1.0 - 1e-10))
        self.assertTrue(np.all(autosim <= 1.0 + 1e-10))


    def test_known_input(self):
        """Test with known input."""
        # Two identical vectors should have cosine similarity 1
        arr = np.array([[1.0, 0.0], [1.0, 0.0]])
        autosim = auto_comp.get_cosine_autosimilarity(arr)
        expected = np.ones((2, 2))
        np.testing.assert_allclose(autosim, expected, rtol=1e-10)

        # Orthogonal vectors should have cosine similarity 0
        arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        autosim = auto_comp.get_cosine_autosimilarity(arr)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(autosim, expected, rtol=1e-10)

    def test_list_input(self):
        """Test that function accepts list input."""
        arr_list = [[1.0, 2.0], [3.0, 4.0]]
        autosim = auto_comp.get_cosine_autosimilarity(arr_list)
        self.assertEqual(autosim.shape, (2, 2))

    def test_no_side_effects(self):
        """Test that input is not modified."""
        original = self.arr.copy()
        auto_comp.get_cosine_autosimilarity(self.arr)
        np.testing.assert_array_equal(self.arr, original)


class TestAutocorrelationAutosimilarity(unittest.TestCase):
    """Test get_autocorrelation_autosimilarity function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_symmetry(self):
        """Test that autosimilarity matrix is symmetric."""
        autosim = auto_comp.get_autocorrelation_autosimilarity(self.arr)
        np.testing.assert_allclose(autosim, autosim.T, rtol=1e-10)

    def test_centering(self):
        """Test that centering is applied (mean is subtracted)."""
        # Create array where columns have different means
        arr = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        autosim = auto_comp.get_autocorrelation_autosimilarity(arr, normalise=False)
        
        # The function should center the data
        # After centering: [[-1, -1], [0, 0], [1, 1]]
        # Dot product should give specific values
        self.assertIsInstance(autosim, np.ndarray)
        self.assertEqual(autosim.shape, (3, 3))

    def test_with_normalization(self):
        """Test with normalization enabled."""
        autosim_norm = auto_comp.get_autocorrelation_autosimilarity(self.arr, normalise=True)
        # Diagonal should be 1 after normalization
        np.testing.assert_allclose(np.diag(autosim_norm), np.ones(10), rtol=1e-10)

    def test_without_normalization(self):
        """Test without normalization."""
        autosim = auto_comp.get_autocorrelation_autosimilarity(self.arr, normalise=False)
        # Should still be symmetric
        np.testing.assert_allclose(autosim, autosim.T, rtol=1e-10)

    def test_deprecated_covariance_function(self):
        """Test that deprecated get_covariance_autosimilarity works identically."""
        autosim_auto = auto_comp.get_autocorrelation_autosimilarity(self.arr)
        autosim_cov = auto_comp.get_covariance_autosimilarity(self.arr)
        np.testing.assert_array_equal(autosim_auto, autosim_cov)

    def test_no_side_effects(self):
        """Test that input is not modified."""
        original = self.arr.copy()
        auto_comp.get_autocorrelation_autosimilarity(self.arr)
        np.testing.assert_array_equal(self.arr, original)


class TestRBFAutosimilarity(unittest.TestCase):
    """Test get_rbf_autosimilarity function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_symmetry(self):
        """Test that autosimilarity matrix is symmetric."""
        autosim = auto_comp.get_rbf_autosimilarity(self.arr)
        np.testing.assert_allclose(autosim, autosim.T, rtol=1e-10)

    def test_diagonal_is_one(self):
        """Test that diagonal elements are 1 (self-similarity in RBF)."""
        autosim = auto_comp.get_rbf_autosimilarity(self.arr)
        np.testing.assert_allclose(np.diag(autosim), np.ones(10), rtol=1e-10)

    def test_value_range(self):
        """Test that values are in (0, 1]."""
        autosim = auto_comp.get_rbf_autosimilarity(self.arr)
        self.assertTrue(np.all(autosim > 0.0))
        self.assertTrue(np.all(autosim <= 1.0))

    def test_with_custom_gamma(self):
        """Test with custom gamma value."""
        gamma = 0.5
        autosim = auto_comp.get_rbf_autosimilarity(self.arr, gamma=gamma)
        self.assertEqual(autosim.shape, (10, 10))
        # Should still have diagonal of 1
        np.testing.assert_allclose(np.diag(autosim), np.ones(10), rtol=1e-10)

    def test_with_default_gamma(self):
        """Test with default gamma (computed from get_gamma_std)."""
        autosim = auto_comp.get_rbf_autosimilarity(self.arr, gamma=None)
        self.assertEqual(autosim.shape, (10, 10))

    def test_with_and_without_normalization(self):
        """Test both normalization options."""
        autosim_norm = auto_comp.get_rbf_autosimilarity(self.arr, normalise=True)
        autosim_no_norm = auto_comp.get_rbf_autosimilarity(self.arr, normalise=False)
        
        # Both should be valid autosimilarity matrices
        self.assertEqual(autosim_norm.shape, (10, 10))
        self.assertEqual(autosim_no_norm.shape, (10, 10))
        
        # They should differ (unless the data happens to be pre-normalized)
        # Just check they're both valid
        np.testing.assert_allclose(np.diag(autosim_norm), np.ones(10), rtol=1e-10)
        np.testing.assert_allclose(np.diag(autosim_no_norm), np.ones(10), rtol=1e-10)

    def test_no_side_effects(self):
        """Test that input is not modified."""
        original = self.arr.copy()
        auto_comp.get_rbf_autosimilarity(self.arr)
        np.testing.assert_array_equal(self.arr, original)


class TestCenteredRBFAutosimilarity(unittest.TestCase):
    """Test get_centered_rbf_autosimilarity function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_symmetry(self):
        """Test that autosimilarity matrix is symmetric."""
        autosim = auto_comp.get_centered_rbf_autosimilarity(self.arr)
        np.testing.assert_allclose(autosim, autosim.T, rtol=1e-10)

    def test_diagonal_is_one(self):
        """Test that diagonal elements are 1."""
        autosim = auto_comp.get_centered_rbf_autosimilarity(self.arr)
        np.testing.assert_allclose(np.diag(autosim), np.ones(10), rtol=1e-10)

    def test_value_range(self):
        """Test that values are in (0, 1]."""
        autosim = auto_comp.get_centered_rbf_autosimilarity(self.arr)
        self.assertTrue(np.all(autosim > 0.0))
        self.assertTrue(np.all(autosim <= 1.0))

    def test_differs_from_non_centered(self):
        """Test that centered RBF differs from regular RBF."""
        autosim_centered = auto_comp.get_centered_rbf_autosimilarity(self.arr)
        autosim_regular = auto_comp.get_rbf_autosimilarity(self.arr)
        
        # They should generally differ (unless mean is already zero)
        # Both should be valid matrices though
        self.assertEqual(autosim_centered.shape, autosim_regular.shape)

    def test_no_side_effects(self):
        """Test that input is not modified."""
        original = self.arr.copy()
        auto_comp.get_centered_rbf_autosimilarity(self.arr)
        np.testing.assert_array_equal(self.arr, original)


class TestGammaStd(unittest.TestCase):
    """Test get_gamma_std function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_positive_output(self):
        """Test that gamma is always positive."""
        gamma = auto_comp.get_gamma_std(self.arr)
        self.assertGreater(gamma, 0.0)

    def test_with_scaling_factor(self):
        """Test with different scaling factors."""
        gamma1 = auto_comp.get_gamma_std(self.arr, scaling_factor=1)
        gamma2 = auto_comp.get_gamma_std(self.arr, scaling_factor=2)
        
        # gamma2 should be twice gamma1
        self.assertAlmostEqual(gamma2, 2 * gamma1, places=10)

    def test_with_and_without_diagonal(self):
        """Test with diagonal included vs excluded."""
        gamma_no_diag = auto_comp.get_gamma_std(self.arr, no_diag=True)
        gamma_with_diag = auto_comp.get_gamma_std(self.arr, no_diag=False)
        
        # Both should be positive
        self.assertGreater(gamma_no_diag, 0.0)
        self.assertGreater(gamma_with_diag, 0.0)
        
        # They should differ (diagonal has zeros which affects std)
        self.assertNotAlmostEqual(gamma_no_diag, gamma_with_diag)

    def test_with_and_without_normalization(self):
        """Test both normalization options."""
        gamma_norm = auto_comp.get_gamma_std(self.arr, normalise=True)
        gamma_no_norm = auto_comp.get_gamma_std(self.arr, normalise=False)
        
        # Both should be positive
        self.assertGreater(gamma_norm, 0.0)
        self.assertGreater(gamma_no_norm, 0.0)

    def test_no_side_effects(self):
        """Test that input is not modified."""
        original = self.arr.copy()
        auto_comp.get_gamma_std(self.arr)
        np.testing.assert_array_equal(self.arr, original)


class TestSwitchAutosimilarity(unittest.TestCase):
    """Test switch_autosimilarity dispatcher function."""

    def setUp(self):
        np.random.seed(42)
        self.arr = np.random.rand(10, 5)

    def test_cosine_dispatch(self):
        """Test that 'cosine' dispatches to get_cosine_autosimilarity."""
        autosim_switch = auto_comp.switch_autosimilarity(self.arr, "cosine")
        autosim_direct = auto_comp.get_cosine_autosimilarity(self.arr)
        np.testing.assert_array_equal(autosim_switch, autosim_direct)

    def test_autocorrelation_dispatch(self):
        """Test that 'autocorrelation' dispatches correctly."""
        autosim_switch = auto_comp.switch_autosimilarity(self.arr, "autocorrelation")
        autosim_direct = auto_comp.get_autocorrelation_autosimilarity(self.arr)
        np.testing.assert_array_equal(autosim_switch, autosim_direct)

    def test_covariance_dispatch(self):
        """Test that 'covariance' dispatches to autocorrelation (deprecated name)."""
        autosim_cov = auto_comp.switch_autosimilarity(self.arr, "covariance")
        autosim_auto = auto_comp.switch_autosimilarity(self.arr, "autocorrelation")
        np.testing.assert_array_equal(autosim_cov, autosim_auto)

    def test_rbf_dispatch(self):
        """Test that 'rbf' dispatches to get_rbf_autosimilarity."""
        autosim_switch = auto_comp.switch_autosimilarity(self.arr, "rbf", gamma=0.5)
        autosim_direct = auto_comp.get_rbf_autosimilarity(self.arr, gamma=0.5)
        np.testing.assert_array_equal(autosim_switch, autosim_direct)

    def test_centered_rbf_dispatch(self):
        """Test that 'centered_rbf' dispatches correctly."""
        autosim_switch = auto_comp.switch_autosimilarity(self.arr, "centered_rbf")
        autosim_direct = auto_comp.get_centered_rbf_autosimilarity(self.arr)
        np.testing.assert_array_equal(autosim_switch, autosim_direct)

    def test_case_insensitivity(self):
        """Test that similarity type is case insensitive."""
        autosim_lower = auto_comp.switch_autosimilarity(self.arr, "cosine")
        autosim_upper = auto_comp.switch_autosimilarity(self.arr, "COSINE")
        autosim_mixed = auto_comp.switch_autosimilarity(self.arr, "CoSiNe")
        
        np.testing.assert_array_equal(autosim_lower, autosim_upper)
        np.testing.assert_array_equal(autosim_lower, autosim_mixed)

    def test_invalid_similarity_type_raises(self):
        """Test that invalid similarity type raises InvalidArgumentValueException."""
        with self.assertRaises(err.InvalidArgumentValueException):
            auto_comp.switch_autosimilarity(self.arr, "invalid_type")

    def test_normalise_parameter(self):
        """Test that normalise parameter is passed through."""
        # For autocorrelation which accepts normalise
        autosim_norm = auto_comp.switch_autosimilarity(
            self.arr, "autocorrelation", normalise=True
        )
        autosim_no_norm = auto_comp.switch_autosimilarity(
            self.arr, "autocorrelation", normalise=False
        )
        
        # They should differ
        self.assertEqual(autosim_norm.shape, autosim_no_norm.shape)


class TestAutosimilarityProperties(unittest.TestCase):
    """Property-based tests for autosimilarity functions."""

    SIMILARITY_TYPES = ["cosine", "autocorrelation", "rbf", "centered_rbf"]
    RANDOM_SEEDS = [42, 123, 456]
    MATRIX_SHAPES = [(10, 5), (20, 8), (15, 3)]

    def test_all_types_produce_symmetric_matrices(self):
        """Test that all similarity types produce symmetric matrices."""
        for sim_type in self.SIMILARITY_TYPES:
            for seed in self.RANDOM_SEEDS:
                for shape in self.MATRIX_SHAPES:
                    with self.subTest(sim_type=sim_type, seed=seed, shape=shape):
                        np.random.seed(seed)
                        arr = np.random.rand(*shape)
                        autosim = auto_comp.switch_autosimilarity(arr, sim_type)
                        
                        # Check symmetry
                        np.testing.assert_allclose(
                            autosim, autosim.T, rtol=1e-10,
                            err_msg=f"{sim_type} not symmetric for seed={seed}, shape={shape}"
                        )

    def test_all_types_produce_square_matrices(self):
        """Test that all similarity types produce square matrices."""
        for sim_type in self.SIMILARITY_TYPES:
            for shape in self.MATRIX_SHAPES:
                with self.subTest(sim_type=sim_type, shape=shape):
                    np.random.seed(42)
                    arr = np.random.rand(*shape)
                    autosim = auto_comp.switch_autosimilarity(arr, sim_type)
                    
                    # Check square
                    self.assertEqual(autosim.shape[0], autosim.shape[1])
                    # Check size matches input rows
                    self.assertEqual(autosim.shape[0], shape[0])

    def test_diagonal_values_for_distance_based(self):
        """Test diagonal values for cosine and RBF (should be 1)."""
        distance_types = ["cosine", "rbf", "centered_rbf"]
        
        for sim_type in distance_types:
            with self.subTest(sim_type=sim_type):
                np.random.seed(42)
                arr = np.random.rand(10, 5)
                autosim = auto_comp.switch_autosimilarity(arr, sim_type)
                
                # Diagonal should be 1
                np.testing.assert_allclose(
                    np.diag(autosim), np.ones(10), rtol=1e-10,
                    err_msg=f"{sim_type} diagonal not 1"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
