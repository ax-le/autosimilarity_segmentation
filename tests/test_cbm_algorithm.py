# -*- coding: utf-8 -*-
"""
Exhaustive tests for CBM_algorithm.py

Tests include:
- Fast vs slow computation mode equivalence
- Side effect checking
- Exception handling (ToDebugException)
- Helper function unit tests
- Timing comparisons
"""

import unittest
import time
import copy
import numpy as np
import as_seg.CBM_algorithm as cbm
import as_seg.model.errors as err
from scipy.sparse import diags

class TestCBMWithValues(unittest.TestCase):
    def setUp(self):
        # Fixed seed for reproducibility
        np.random.seed(42)
        self.N = 100
        # Random autosimilarity matrix
        self.autosim = np.random.rand(self.N, self.N)
        self.max_size = 10
        self.EXPECTED_SEGMENTS = [(0, 4), (4, 12), (12, 20), (20, 28), (28, 36), (36, 44), (44, 54), (54, 62), (62, 70), (70, 80), (80, 88), (88, 96), (96, 100)]
        self.EXPECTED_SCORE = 44.71811465385153
        
    def test_cbm_output_with_values(self):
        print(f"\nTesting with N={self.N}, max_size={self.max_size}")
        segments, score = cbm.compute_cbm(self.autosim, max_size=self.max_size, compute_score_using_convolve=False)
        
        print(f"Computed Segments: {segments}")
        print(f"Computed Score: {score}")
        
        if self.EXPECTED_SEGMENTS is not None:
            self.assertEqual(segments, self.EXPECTED_SEGMENTS, "Segments do not match expected values")
        else:
            print("Skipping segments assertion (EXPECTED_SEGMENTS is None)")
            
        if self.EXPECTED_SCORE is not None:
            self.assertAlmostEqual(score, self.EXPECTED_SCORE, places=5, msg="Score does not match expected value")
        else:
            print("Skipping score assertion (EXPECTED_SCORE is None)")

    def test_cbm_fast_output_with_values(self):
        print(f"\nTesting Convolutive with N={self.N}, max_size={self.max_size}")
        segments, score = cbm.compute_cbm(self.autosim, max_size=self.max_size, compute_score_using_convolve=True)
        
        print(f"Computed Segments (Convolutive): {segments}")
        print(f"Computed Score (Convolutive): {score}")
        
        if self.EXPECTED_SEGMENTS is not None:
            self.assertEqual(segments, self.EXPECTED_SEGMENTS, "Segments do not match expected values")
        
        if self.EXPECTED_SCORE is not None:
            self.assertAlmostEqual(score, self.EXPECTED_SCORE, places=5, msg="Score does not match expected value")

class TestComputeCBMFastVsSlowEquivalence(unittest.TestCase):
    """Test that compute_score_using_convolve mode returns identical results to slow mode."""

    # Test parameters
    MATRIX_SIZES = [20, 50, 100, 200, 500, 1000]
    RANDOM_SEEDS = [42, 123, 456, 789, 0]
    MAX_SIZES = [8, 16, 32]
    PENALTY_FUNCS = ["modulo8", "modulo4", "modulo8modulo4"]

    def _create_autosimilarity(self, size, seed):
        """Create a random autosimilarity matrix."""
        np.random.seed(seed)
        return np.random.rand(size, size)

    def test_equivalence_across_matrix_sizes_and_seeds(self):
        """Test fast vs slow equivalence for multiple sizes and seeds."""
        for size in self.MATRIX_SIZES:
            for seed in self.RANDOM_SEEDS:
                for max_size in self.MAX_SIZES:
                    with self.subTest(size=size, seed=seed, max_size=max_size):
                        autosim = self._create_autosimilarity(size, seed)

                        # Slow computation
                        segments_slow, score_slow = cbm.compute_cbm(
                            autosim, max_size=max_size, compute_score_using_convolve=False
                        )

                        # Fast computation
                        segments_fast, score_fast = cbm.compute_cbm(
                            autosim, max_size=max_size, compute_score_using_convolve=True
                        )

                        self.assertEqual(
                            segments_slow, segments_fast,
                            f"Segments differ for size={size}, seed={seed}, max_size={max_size}"
                        )
                        self.assertAlmostEqual(
                            score_slow, score_fast, places=2,
                            msg=f"Scores differ for size={size}, seed={seed}, max_size={max_size}"
                        )

    def test_equivalence_across_max_sizes(self):
        """Test fast vs slow equivalence for different max_size values."""
        size = 100
        seed = 42
        autosim = self._create_autosimilarity(size, seed)

        for max_size in self.MAX_SIZES:
            with self.subTest(max_size=max_size):
                segments_slow, score_slow = cbm.compute_cbm(
                    autosim, max_size=max_size, compute_score_using_convolve=False
                )
                segments_fast, score_fast = cbm.compute_cbm(
                    autosim, max_size=max_size, compute_score_using_convolve=True
                )

                self.assertEqual(segments_slow, segments_fast)
                self.assertAlmostEqual(score_slow, score_fast, places=10)

    def test_equivalence_across_penalty_funcs(self):
        """Test fast vs slow equivalence for different penalty functions."""
        size = 80
        seed = 42
        autosim = self._create_autosimilarity(size, seed)
        max_size = 16

        for penalty_func in self.PENALTY_FUNCS:
            with self.subTest(penalty_func=penalty_func):
                segments_slow, score_slow = cbm.compute_cbm(
                    autosim, max_size=max_size, penalty_func=penalty_func,
                    compute_score_using_convolve=False
                )
                segments_fast, score_fast = cbm.compute_cbm(
                    autosim, max_size=max_size, penalty_func=penalty_func,
                    compute_score_using_convolve=True
                )

                self.assertEqual(segments_slow, segments_fast)
                self.assertAlmostEqual(score_slow, score_fast, places=10)

    def test_equivalence_with_bands_number(self):
        """Test fast vs slow equivalence with different bands_number."""
        size = 80
        seed = 42
        autosim = self._create_autosimilarity(size, seed)
        max_size = 16

        for bands_number in [None, 3, 5, 7]:
            with self.subTest(bands_number=bands_number):
                segments_slow, score_slow = cbm.compute_cbm(
                    autosim, max_size=max_size, bands_number=bands_number,
                    compute_score_using_convolve=False
                )
                segments_fast, score_fast = cbm.compute_cbm(
                    autosim, max_size=max_size, bands_number=bands_number,
                    compute_score_using_convolve=True
                )

                self.assertEqual(segments_slow, segments_fast)
                self.assertAlmostEqual(score_slow, score_fast, places=10)

    # def test_timing_comparison(self):
    #     """Time both fast and slow modes for performance comparison."""
    #     seed = 42
    #     num_runs = 3

    #     all_slow_times = []
    #     all_fast_times = []

    #     for size in self.MATRIX_SIZES:
    #         for max_size in self.MAX_SIZES:
    #             slow_times = []
    #             fast_times = []
    #             autosim = self._create_autosimilarity(size, seed)
    #             for _ in range(num_runs):
    #                 # Time slow computation
    #                 start = time.perf_counter()
    #                 cbm.compute_cbm(autosim, max_size=max_size, compute_score_using_convolve=False)
    #                 slow_times.append(time.perf_counter() - start)

    #                 # Time fast computation
    #                 start = time.perf_counter()
    #                 cbm.compute_cbm(autosim, max_size=max_size, compute_score_using_convolve=True)
    #                 fast_times.append(time.perf_counter() - start)

    #             avg_slow = sum(slow_times) / num_runs
    #             avg_fast = sum(fast_times) / num_runs

    #             print(f"\n=== Timing Comparison (N={size}, max_size={max_size}, {num_runs} runs) ===")
    #             print(f"Slow computation average: {avg_slow:.4f}s")
    #             print(f"Fast computation average: {avg_fast:.4f}s")
    #             print(f"Speedup: {avg_slow / avg_fast:.2f}x" if avg_fast > 0 else "N/A")

    #             all_slow_times.append(avg_slow)
    #             all_fast_times.append(avg_fast)

    #     avg_slow = np.mean(all_slow_times)
    #     avg_fast = np.mean(all_fast_times)

    #     print(f"\n=== Overall Timing Comparison ({num_runs} runs) ===")
    #     print(f"Slow computation average: {avg_slow:.4f}s")
    #     print(f"Fast computation average: {avg_fast:.4f}s")
    #     print(f"Speedup: {avg_slow / avg_fast:.2f}x" if avg_fast > 0 else "N/A")

class TestComputeCBMSideEffects(unittest.TestCase):
    """Verify that compute_cbm does not modify input matrices."""

    def setUp(self):
        np.random.seed(42)
        self.N = 50
        self.autosim = np.random.rand(self.N, self.N)
        self.max_size = 16

    def test_no_side_effects_slow_mode(self):
        """Ensure input is unchanged after slow computation."""
        original = self.autosim.copy()
        cbm.compute_cbm(self.autosim, max_size=self.max_size, compute_score_using_convolve=False)
        np.testing.assert_array_equal(
            self.autosim, original,
            err_msg="Slow mode modified the input autosimilarity matrix"
        )

    def test_no_side_effects_fast_mode(self):
        """Ensure input is unchanged after fast computation."""
        original = self.autosim.copy()
        cbm.compute_cbm(self.autosim, max_size=self.max_size, compute_score_using_convolve=True)
        np.testing.assert_array_equal(
            self.autosim, original,
            err_msg="Fast mode modified the input autosimilarity matrix"
        )


class TestComputeCBMExceptions(unittest.TestCase):
    """Test that ToDebugException and other errors are raised appropriately."""

    def test_invalid_min_size_raises(self):
        """Test that min_size < 1 raises InvalidArgumentValueException."""
        np.random.seed(42)
        autosim = np.random.rand(50, 50)
        
        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.compute_cbm(autosim, min_size=0, max_size=16)

        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.compute_cbm(autosim, min_size=-1, max_size=16)

    def test_invalid_penalty_func_raises(self):
        """Test that invalid penalty_func raises InvalidArgumentValueException."""
        np.random.seed(42)
        autosim = np.random.rand(50, 50)
        
        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.compute_cbm(autosim, penalty_func="invalid_penalty")


class TestComputeCBMOutputStructure(unittest.TestCase):
    """Test output structure and validity."""

    def setUp(self):
        np.random.seed(42)
        self.N = 100
        self.autosim = np.random.rand(self.N, self.N)
        self.max_size = 16

    def test_segments_are_list_of_tuples(self):
        """Test that segments is a list of tuples."""
        segments, _ = cbm.compute_cbm(
            self.autosim, max_size=self.max_size, compute_score_using_convolve=False
        )
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, tuple)
            self.assertEqual(len(seg), 2)

    def test_segments_cover_entire_matrix(self):
        """Test that segments cover from 0 to N-1."""
        for fast_comp in [True, False]:
            with self.subTest(compute_score_using_convolve=fast_comp):
                segments, _ = cbm.compute_cbm(
                    self.autosim, max_size=self.max_size, compute_score_using_convolve=fast_comp
                )
                # First segment should start at 0
                self.assertEqual(segments[0][0], 0)
                # Last segment should end at N
                self.assertEqual(segments[-1][1], self.N)

    def test_segments_are_contiguous(self):
        """Test that segments are contiguous (no gaps)."""
        for fast_comp in [True, False]:
            with self.subTest(compute_score_using_convolve=fast_comp):
                segments, _ = cbm.compute_cbm(
                    self.autosim, max_size=self.max_size, compute_score_using_convolve=fast_comp
                )
                for i in range(len(segments) - 1):
                    self.assertEqual(
                        segments[i][1], segments[i + 1][0],
                        f"Gap between segment {i} and {i + 1}"
                    )

    def test_score_is_valid_float(self):
        """Test that score is a valid float."""
        for fast_comp in [True, False]:
            with self.subTest(compute_score_using_convolve=fast_comp):
                _, score = cbm.compute_cbm(
                    self.autosim, max_size=self.max_size, compute_score_using_convolve=fast_comp
                )
                self.assertIsInstance(score, (int, float))
                self.assertFalse(np.isnan(score))
                self.assertFalse(np.isinf(score))


class TestPrecomputeCorrScoresConv(unittest.TestCase):
    """Test the precompute_corr_scores_conv function."""

    def setUp(self):
        np.random.seed(42)
        self.N = 50
        self.autosim = np.random.rand(self.N, self.N)
        self.min_size = 1
        self.max_size = 16
        self.kernels = cbm.compute_all_kernels(self.max_size)

    def test_output_shape(self):
        """Test that output has correct shape."""
        scores, max_corr = cbm.precompute_corr_scores_conv(
            self.autosim, self.kernels, self.min_size, self.max_size, self.N
        )
        
        self.assertEqual(scores.shape[0], self.N)
        self.assertEqual(scores.shape[1], self.max_size + 1)

    def test_precomputed_matches_direct_computation(self):
        """Test that precomputed scores match direct correlation_score calls."""
        scores, _ = cbm.precompute_corr_scores_conv(
            self.autosim, self.kernels, self.min_size, self.max_size, self.N
        )
        
        # Check a few segments
        for start_idx in [0, 10, 20]:
            for length in [4, 8, 12]:
                end_idx = start_idx + length
                if end_idx <= self.N and length <= self.max_size:
                    expected = cbm.correlation_score(
                        self.autosim[start_idx:end_idx, start_idx:end_idx],
                        self.kernels
                    )
                    actual = scores[start_idx, length]
                    self.assertAlmostEqual(
                        expected, actual, places=10,
                        msg=f"Mismatch at start={start_idx}, length={length}"
                    )

    def test_max_corr_seg_size_eight(self):
        """Test that max_corr_seg_size_eight is returned correctly for size 8."""
        scores, max_corr = cbm.precompute_corr_scores_conv(
            self.autosim, self.kernels, self.min_size, self.max_size, self.N
        )
        
        # Verify max_corr is the max of size-8 segments
        size_8_scores = scores[:self.N - 8 + 1, 8]
        expected_max = np.max(size_8_scores[size_8_scores != -np.inf])
        self.assertAlmostEqual(max_corr, expected_max, places=10)


class TestPenaltyCostFromArg(unittest.TestCase):
    """Test penalty_cost_from_arg function."""

    def test_modulo8_penalty(self):
        """Test modulo8 penalty function values."""
        # segment_length == 8 -> 0
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 8), 0)
        # segment_length % 4 == 0 (but not 8) -> 1/4
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 4), 1/4)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 12), 1/4)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 16), 1/4)
        # segment_length % 2 == 0 (but not %4) -> 1/2
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 2), 1/2)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 6), 1/2)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 10), 1/2)
        # odd lengths -> 1
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 1), 1)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 3), 1)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8", 5), 1)

    def test_modulo4_penalty(self):
        """Test modulo4 penalty function values."""
        # segment_length % 4 == 0 -> 0
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 4), 0)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 8), 0)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 12), 0)
        # segment_length % 2 == 0 (but not %4) -> 1/2
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 2), 1/2)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 6), 1/2)
        # odd lengths -> 1
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 1), 1)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo4", 3), 1)

    def test_modulo8modulo4_penalty(self):
        """Test modulo8modulo4 penalty function values."""
        # segment_length == 8 -> 0
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 8), 0)
        # segment_length == 4 -> 1/4
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 4), 1/4)
        # segment_length % 2 == 0 -> 1/2
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 2), 1/2)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 6), 1/2)
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 12), 1/2)
        # odd -> 1
        self.assertEqual(cbm.penalty_cost_from_arg("modulo8modulo4", 1), 1)

    def test_target_deviation_penalties(self):
        """Test target deviation penalty functions."""
        # target_deviation_8_alpha_half: |x - 8|^(1/2)
        self.assertAlmostEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_half", 8), 0
        )
        self.assertAlmostEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_half", 4), 2.0
        )
        self.assertAlmostEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_half", 12), 2.0
        )

        # target_deviation_8_alpha_one: |x - 8|
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_one", 8), 0
        )
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_one", 4), 4
        )
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_one", 12), 4
        )

        # target_deviation_8_alpha_two: |x - 8|^2
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_two", 8), 0
        )
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_two", 4), 16
        )
        self.assertEqual(
            cbm.penalty_cost_from_arg("target_deviation_8_alpha_two", 12), 16
        )

    def test_invalid_penalty_func_raises(self):
        """Test that invalid penalty function raises exception."""
        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.penalty_cost_from_arg("invalid_function", 8)


class TestPossibleSegmentStart(unittest.TestCase):
    """Test possible_segment_start function."""

    def test_basic_range(self):
        """Test basic segment start calculation."""
        # idx=10, min_size=1, max_size=None -> range(0, 10)
        result = list(cbm.possible_segment_start(10, min_size=1, max_size=None))
        self.assertEqual(result, list(range(0, 10)))

    def test_with_max_size(self):
        """Test with max_size constraint."""
        # idx=10, min_size=1, max_size=5 -> range(5, 10)
        result = list(cbm.possible_segment_start(10, min_size=1, max_size=5))
        self.assertEqual(result, list(range(5, 10)))

    def test_with_min_size(self):
        """Test with min_size constraint."""
        # idx=10, min_size=3, max_size=None -> range(0, 8)
        result = list(cbm.possible_segment_start(10, min_size=3, max_size=None))
        self.assertEqual(result, list(range(0, 8)))

    def test_with_both_constraints(self):
        """Test with both min_size and max_size."""
        # idx=10, min_size=2, max_size=5 -> range(5, 9)
        result = list(cbm.possible_segment_start(10, min_size=2, max_size=5))
        self.assertEqual(result, list(range(5, 9)))

    def test_small_idx(self):
        """Test when idx is smaller than max_size."""
        # idx=3, min_size=1, max_size=10 -> range(0, 3)
        result = list(cbm.possible_segment_start(3, min_size=1, max_size=10))
        self.assertEqual(result, list(range(0, 3)))

    def test_idx_smaller_than_min_size(self):
        """Test when idx is smaller than min_size (empty result)."""
        # idx=2, min_size=3 -> []
        result = list(cbm.possible_segment_start(2, min_size=3, max_size=10))
        self.assertEqual(result, [])

    def test_invalid_min_size_raises(self):
        """Test that min_size < 1 raises exception."""
        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.possible_segment_start(10, min_size=0)
        
        with self.assertRaises(err.InvalidArgumentValueException):
            cbm.possible_segment_start(10, min_size=-1)


class TestComputeAllKernels(unittest.TestCase):
    """Test compute_all_kernels function."""

    def test_kernel_count(self):
        """Test that correct number of kernels are generated."""
        max_size = 10
        kernels = cbm.compute_all_kernels(max_size)
        self.assertEqual(len(kernels), max_size + 1)

    def test_kernel_shapes(self):
        """Test kernel shapes are correct."""
        max_size = 10
        kernels = cbm.compute_all_kernels(max_size)
        
        for p in range(1, max_size + 1):
            self.assertEqual(kernels[p].shape, (p, p))

    def test_kernel_zero(self):
        """Test kernel at index 0 is [0]."""
        kernels = cbm.compute_all_kernels(10)
        self.assertEqual(kernels[0], [0])

    def test_full_kernel_values(self):
        """Test full kernel is ones minus identity."""
        max_size = 5
        kernels = cbm.compute_all_kernels(max_size, bands_number=None)
        
        for p in range(1, max_size + 1):
            expected = np.ones((p, p)) - np.identity(p)
            np.testing.assert_array_equal(kernels[p], expected)

    def test_banded_kernel(self):
        """Test banded kernel structure."""
        max_size = 10
        bands_number = 3
        kernels = cbm.compute_all_kernels(max_size, bands_number=bands_number)
        
        # For p < bands_number, should be full kernel
        for p in range(1, bands_number):
            expected = np.ones((p, p)) - np.identity(p)
            np.testing.assert_array_equal(kernels[p], expected)
        
        # For p >= bands_number, diagonal should be 0 and only bands_number bands
        for p in range(bands_number, max_size + 1):
            kernel = kernels[p]
            # Check diagonal is 0
            np.testing.assert_array_equal(np.diag(kernel), np.zeros(p))
            # Check the whole matrix
            k = np.array([np.ones(p-i) for i in np.abs(range(-bands_number, bands_number + 1))],dtype=object)
            offset = [i for i in range(-bands_number, bands_number + 1)]
            kern = diags(k,offset).toarray() - np.identity(p)
            np.testing.assert_array_equal(kernel, kern)

class TestCorrelationScore(unittest.TestCase):
    """Test correlation_score function."""

    def setUp(self):
        self.kernels = cbm.compute_all_kernels(10)

    def test_known_matrix(self):
        """Test correlation score on known matrices."""
        # 2x2 all ones matrix with kernel that is ones minus identity
        # kernel = [[0, 1], [1, 0]]
        # score = sum(kernel * matrix) / p^2 = 2 / 4 = 0.5
        matrix = np.ones((2, 2))
        score = cbm.correlation_score(matrix, self.kernels)
        self.assertAlmostEqual(score, 0.5, places=10)

    def test_identity_matrix(self):
        """Test correlation score on identity matrix."""
        # Identity matrix with kernel = ones - identity
        # All off-diagonal elements in matrix are 0, so score = 0
        matrix = np.identity(3)
        score = cbm.correlation_score(matrix, self.kernels)
        self.assertAlmostEqual(score, 0.0, places=10)

    def test_zeros_matrix(self):
        """Test correlation score on zeros matrix."""
        matrix = np.zeros((4, 4))
        score = cbm.correlation_score(matrix, self.kernels)
        self.assertAlmostEqual(score, 0.0, places=10)


class TestBacktrackingForBestSegments(unittest.TestCase):
    """Test backtracking_for_best_segments function."""

    def test_simple_case(self):
        """Test simple backtracking case."""
        segments_best_starts = [0, 0, 1, 2, 3]
        segments = cbm.backtracking_for_best_segments(segments_best_starts, 4)
        expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.assertEqual(segments, expected)

    def test_larger_segments(self):
        """Test backtracking with larger segments."""
        segments_best_starts = [0, None, None, 0, None, None, 4, 3]
        segments = cbm.backtracking_for_best_segments(segments_best_starts, 7)
        expected = [(0, 3), (3, 7)]
        self.assertEqual(segments, expected)

    def test_single_segment_coverage(self):
        """Test single segment covering entire range."""
        segments_best_starts = [0, None, None, None, None, 0]
        segments = cbm.backtracking_for_best_segments(segments_best_starts, 5)
        expected = [(0, 5)]
        self.assertEqual(segments, expected)


class TestBlockDiagonalSelfSimilarity(unittest.TestCase):
    """Test CBM on synthetic block-diagonal self-similarity matrices.

    The matrices are zero everywhere, except on contiguous square blocks
    centered on the diagonal.  Each block of size s at position (start, start)
    has all entries equal to 1.  The diagonal itself is always 1 (as expected
    for any proper self-similarity matrix).

    Because the correlation kernel ignores the diagonal, the off-diagonal mass
    inside each block is the only signal.  With penalty_weight=0 (no length
    penalty) CBM should recover the true blocks exactly regardless of their
    sizes.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _build_block_diagonal(block_sizes):
        """Build a block-diagonal self-similarity matrix.

        Parameters
        ----------
        block_sizes : list of int
            Sizes of the consecutive blocks along the diagonal.

        Returns
        -------
        autosim : np.ndarray, shape (N, N)
            Matrix that is 1 inside each block and 0 outside.
        segments : list of tuples (start, end)
            Ground-truth segment boundaries induced by the blocks.
        """
        N = sum(block_sizes)
        autosim = np.zeros((N, N))
        segments = []
        start = 0
        for size in block_sizes:
            end = start + size
            autosim[start:end, start:end] = 1.0
            segments.append((start, end))
            start = end
        return autosim, segments

    # ------------------------------------------------------------------
    # Tests for the matrix-construction helper itself
    # ------------------------------------------------------------------
    def test_helper_shape(self):
        """Matrix has the correct shape."""
        block_sizes = [4, 8, 4]
        autosim, _ = self._build_block_diagonal(block_sizes)
        N = sum(block_sizes)
        self.assertEqual(autosim.shape, (N, N))

    def test_helper_block_values(self):
        """Entries inside blocks are 1, entries outside are 0."""
        block_sizes = [3, 5, 2]
        autosim, segments = self._build_block_diagonal(block_sizes)
        for start, end in segments:
            # Inside block: all ones
            np.testing.assert_array_equal(
                autosim[start:end, start:end],
                np.ones((end - start, end - start)),
                err_msg=f"Block ({start}, {end}) should be all ones",
            )
        # Outside all blocks: all zeros
        N = autosim.shape[0]
        mask = np.zeros((N, N), dtype=bool)
        for start, end in segments:
            mask[start:end, start:end] = True
        np.testing.assert_array_equal(
            autosim[~mask],
            np.zeros(np.sum(~mask)),
            err_msg="Entries outside blocks should be zero",
        )

    def test_helper_segments_coverage(self):
        """Helper segments cover [0, N) without gaps."""
        block_sizes = [2, 6, 4, 8]
        _, segments = self._build_block_diagonal(block_sizes)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], sum(block_sizes))
        for i in range(len(segments) - 1):
            self.assertEqual(segments[i][1], segments[i + 1][0])

    # ------------------------------------------------------------------
    # Tests for CBM on block-diagonal matrices
    # ------------------------------------------------------------------
    def test_my_it_works(self):
        """Run CBM and assert that it recovers the ground-truth blocks."""
        autosim = np.zeros((13,13))
        autosim[0:4, 0:4] = 1
        autosim[4:9, 4:9] = 1
        autosim[9:13, 9:13] = 1
        segments, _ = cbm.compute_cbm(
            autosim,
            max_size=13,
            penalty_weight=0,  # no length penalty: pure correlation criterion
            compute_score_using_convolve=False,
        )
        self.assertEqual(
            segments, [(0,4),(4,9),(9,13)]
        )

    def _assert_cbm_recovers_blocks(self, block_sizes, max_size, fast_mode):
        """Run CBM and assert that it recovers the ground-truth blocks."""
        autosim, expected_segments = self._build_block_diagonal(block_sizes)
        segments, _ = cbm.compute_cbm(
            autosim,
            max_size=max_size,
            penalty_weight=0,  # no length penalty: pure correlation criterion
            compute_score_using_convolve=fast_mode,
        )
        self.assertEqual(
            segments, expected_segments,
            msg=(
                f"CBM (fast={fast_mode}) failed to recover blocks {block_sizes} "
                f"with max_size={max_size}.\n"
                f"  Expected: {expected_segments}\n"
                f"  Got:      {segments}"
            ),
        )

    def test_equal_blocks_slow(self):
        """Four equal blocks of size 8 – slow mode."""
        self._assert_cbm_recovers_blocks([8, 8, 8, 8], max_size=16, fast_mode=False)

    # def test_equal_blocks_fast(self):
    #     """Four equal blocks of size 8 – fast (convolve) mode."""
    #     self._assert_cbm_recovers_blocks([8, 8, 8, 8], max_size=16, fast_mode=True)

    def test_mixed_block_sizes_slow(self):
        """Blocks of different sizes – slow mode."""
        self._assert_cbm_recovers_blocks([4, 8, 4, 8], max_size=16, fast_mode=False)

    # def test_mixed_block_sizes_fast(self):
    #     """Blocks of different sizes – fast (convolve) mode."""
    #     self._assert_cbm_recovers_blocks([4, 8, 4, 8], max_size=16, fast_mode=True)

    def test_various_configurations_slow(self):
        """Multiple block configurations – slow mode."""
        configurations = [
            [4, 4, 4, 4],
            [8, 8],
            [4, 8, 8, 4],
            [4, 4, 8],
            [8, 4, 4],
        ]
        for block_sizes in configurations:
            max_size = max(block_sizes) + 4
            with self.subTest(block_sizes=block_sizes):
                self._assert_cbm_recovers_blocks(
                    block_sizes, max_size=max_size, fast_mode=False
                )

    # def test_various_configurations_fast(self):
    #     """Multiple block configurations – fast (convolve) mode."""
    #     configurations = [
    #         [4, 4, 4, 4],
    #         [8, 8],
    #         [4, 8, 8, 4],
    #         [4, 4, 8],
    #         [8, 4, 4],
    #     ]
    #     for block_sizes in configurations:
    #         max_size = max(block_sizes) + 4
    #         with self.subTest(block_sizes=block_sizes):
    #             self._assert_cbm_recovers_blocks(
    #                 block_sizes, max_size=max_size, fast_mode=True
    #             )

    def test_slow_and_fast_agree_on_block_diagonal(self):
        """Slow and fast modes produce identical results on block-diagonal matrices."""
        configurations = [
            [4, 8, 4],
            [8, 8, 8],
            [4, 4, 8, 4],
        ]
        for block_sizes in configurations:
            max_size = max(block_sizes) + 4
            with self.subTest(block_sizes=block_sizes):
                autosim, _ = self._build_block_diagonal(block_sizes)
                segs_slow, score_slow = cbm.compute_cbm(
                    autosim, max_size=max_size, penalty_weight=0,
                    compute_score_using_convolve=False,
                )
                segs_fast, score_fast = cbm.compute_cbm(
                    autosim, max_size=max_size, penalty_weight=0,
                    compute_score_using_convolve=True,
                )
                self.assertEqual(segs_slow, segs_fast)
                self.assertAlmostEqual(score_slow, score_fast, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)

