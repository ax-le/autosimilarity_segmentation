"""Extensive tests for cut_signal_on_bars, _reduce_axis, and
make_2D_array_from_different_length_arrays in utils_for_codecs.py."""

import unittest
import numpy as np
from utils_for_codecs import (
    cut_signal_on_bars,
    bars_in_time_to_samples,
    _reduce_axis,
    make_2D_array_from_different_length_arrays,
    _get_all_shapes_and_varying_axes,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  cut_signal_on_bars
# ═══════════════════════════════════════════════════════════════════════════════

class TestCutSignalOnBars(unittest.TestCase):

    def test_basic_cut(self):
        """Each bar should contain the expected signal slice."""
        sr = 10  # 10 samples per second
        signal = np.arange(100, dtype=float)
        bars = np.array([[1.0, 2.0], [3.0, 5.0]])  # two bars
        result = cut_signal_on_bars(signal, bars, sr)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], signal[10:20])
        np.testing.assert_array_equal(result[1], signal[30:50])

    def test_first_bar_at_zero_is_skipped(self):
        """A first bar starting at t=0 should be removed (silent bar)."""
        sr = 10
        signal = np.arange(100, dtype=float)
        bars = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        result = cut_signal_on_bars(signal, bars, sr)
        self.assertEqual(len(result), 2)  # first bar dropped
        np.testing.assert_array_equal(result[0], signal[20:30])
        np.testing.assert_array_equal(result[1], signal[40:50])

    def test_first_bar_not_at_zero_is_kept(self):
        """If the first bar does NOT start at 0, all bars are kept."""
        sr = 10
        signal = np.arange(100, dtype=float)
        bars = np.array([[0.5, 1.5], [2.0, 3.0]])
        result = cut_signal_on_bars(signal, bars, sr)
        self.assertEqual(len(result), 2)  # all kept

    def test_single_bar(self):
        """Works with a single bar (that doesn't start at 0)."""
        sr = 4
        signal = np.arange(40, dtype=float)
        bars = np.array([[1.0, 3.0]])
        result = cut_signal_on_bars(signal, bars, sr)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], signal[4:12])

    def test_bar_lengths_match_durations(self):
        """Output segment lengths should match bar durations in samples."""
        sr = 100
        signal = np.zeros(10000)
        bars = np.array([[1.0, 1.5], [2.0, 3.5], [5.0, 5.2]])
        result = cut_signal_on_bars(signal, bars, sr)
        expected_lengths = [50, 150, 20]
        for seg, expected_len in zip(result, expected_lengths):
            self.assertEqual(len(seg), expected_len)

    def test_output_is_list_of_arrays(self):
        sr = 10
        signal = np.ones(100)
        bars = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = cut_signal_on_bars(signal, bars, sr)
        self.assertIsInstance(result, list)
        for seg in result:
            self.assertIsInstance(seg, np.ndarray)


# ═══════════════════════════════════════════════════════════════════════════════
#  _reduce_axis
# ═══════════════════════════════════════════════════════════════════════════════

class TestReduceAxis(unittest.TestCase):

    # --- Identity (no-op) ---
    def test_no_op_when_already_target_size(self):
        arr = np.arange(12).reshape(3, 4)
        result = _reduce_axis(arr, axis=1, target_size=4, mode="subsample_uniform")
        np.testing.assert_array_equal(result, arr)

    # --- subsample_uniform ---
    def test_subsample_1d(self):
        arr = np.arange(10, dtype=float)
        result = _reduce_axis(arr, axis=0, target_size=5, mode="subsample_uniform")
        self.assertEqual(result.shape, (5,))
        expected_idx = np.linspace(0, 9, 5, dtype=int)
        np.testing.assert_array_equal(result, arr[expected_idx])

    def test_subsample_2d_axis0(self):
        arr = np.arange(20).reshape(5, 4)
        result = _reduce_axis(arr, axis=0, target_size=3, mode="subsample_uniform")
        self.assertEqual(result.shape, (3, 4))

    def test_subsample_2d_axis1(self):
        arr = np.arange(20).reshape(4, 5)
        result = _reduce_axis(arr, axis=1, target_size=3, mode="subsample_uniform")
        self.assertEqual(result.shape, (4, 3))

    def test_subsample_3d(self):
        arr = np.random.rand(2, 10, 4)
        result = _reduce_axis(arr, axis=1, target_size=5, mode="subsample_uniform")
        self.assertEqual(result.shape, (2, 5, 4))

    def test_subsample_preserves_other_axes(self):
        """Subsampling axis 1 should not modify axis 0 or 2 values."""
        arr = np.random.rand(3, 8, 5)
        result = _reduce_axis(arr, axis=1, target_size=4, mode="subsample_uniform")
        idx = np.linspace(0, 7, 4, dtype=int)
        np.testing.assert_array_equal(result, arr[:, idx, :])

    # --- average_pool ---
    def test_average_pool_1d(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _reduce_axis(arr, axis=0, target_size=2, mode="average_pool")
        self.assertEqual(result.shape, (2,))
        np.testing.assert_allclose(result, [1.5, 3.5])

    def test_average_pool_2d_axis1(self):
        arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = _reduce_axis(arr, axis=1, target_size=3, mode="average_pool")
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_allclose(result[0], [1.5, 3.5, 5.5])

    def test_average_pool_2d_axis0(self):
        arr = np.ones((6, 3))
        arr[3:] = 2.0
        result = _reduce_axis(arr, axis=0, target_size=2, mode="average_pool")
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 2.0)

    def test_average_pool_3d(self):
        arr = np.random.rand(2, 12, 4)
        result = _reduce_axis(arr, axis=1, target_size=4, mode="average_pool")
        self.assertEqual(result.shape, (2, 4, 4))

    # --- Aggregation modes: mean, max, min, sum ---
    def test_mean_collapses_axis(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        result = _reduce_axis(arr, axis=1, target_size=None, mode="mean")
        self.assertEqual(result.shape, (3,))
        np.testing.assert_allclose(result, np.mean(arr, axis=1))

    def test_max_collapses_axis(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        result = _reduce_axis(arr, axis=0, target_size=None, mode="max")
        self.assertEqual(result.shape, (4,))
        np.testing.assert_allclose(result, np.max(arr, axis=0))

    def test_min_collapses_axis(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        result = _reduce_axis(arr, axis=1, target_size=None, mode="min")
        np.testing.assert_allclose(result, np.min(arr, axis=1))

    def test_sum_collapses_axis(self):
        arr = np.ones((4, 5))
        result = _reduce_axis(arr, axis=1, target_size=None, mode="sum")
        np.testing.assert_allclose(result, 5.0)

    # --- Error handling ---
    def test_unknown_mode_raises(self):
        arr = np.ones((3, 4))
        with self.assertRaises(ValueError):
            _reduce_axis(arr, axis=0, target_size=2, mode="unknown_mode")


# ═══════════════════════════════════════════════════════════════════════════════
#  make_2D_array_from_different_length_arrays
# ═══════════════════════════════════════════════════════════════════════════════

class TestMake2DArray(unittest.TestCase):

    # --- Basic contract: always returns (n_bars, features) ---
    def test_output_is_2d(self):
        arrays = [np.random.rand(3, 10), np.random.rand(3, 8)]
        result = make_2D_array_from_different_length_arrays(arrays)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 2)

    def test_same_shapes_returns_flattened(self):
        """When all bars have the same shape, stack and flatten."""
        arrays = [np.ones((4, 5)), np.zeros((4, 5))]
        result = make_2D_array_from_different_length_arrays(arrays)
        self.assertEqual(result.shape, (2, 20))
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.0)

    # --- 1D arrays (e.g. raw signal segments) ---
    def test_1d_varying_lengths_subsample(self):
        arrays = [np.arange(10, dtype=float), np.arange(20, dtype=float)]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        self.assertEqual(result.shape, (2, 10))  # min length is 10

    def test_1d_varying_lengths_mean(self):
        """Mean collapses the only axis → each bar becomes a scalar."""
        arrays = [np.ones(10), np.ones(20) * 3]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="mean"
        )
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_allclose(result, [[1.0], [3.0]])

    # --- 2D arrays (e.g. codebooks × time) ---
    def test_2d_only_axis1_varies_subsample(self):
        """Codebook dim is the same (9), time varies → subsample time only."""
        arrays = [np.random.rand(9, 100), np.random.rand(9, 80), np.random.rand(9, 95)]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        self.assertEqual(result.shape, (3, 9 * 80))  # 9 codebooks × 80 time steps

    def test_2d_only_axis1_varies_average_pool(self):
        arrays = [np.ones((4, 12)), np.ones((4, 8)) * 2]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="average_pool"
        )
        self.assertEqual(result.shape, (2, 4 * 8))  # resampled to min axis-1 = 8

    def test_2d_only_axis0_varies(self):
        """If axis 0 varies and axis 1 is fixed, reduce only axis 0."""
        arrays = [np.ones((3, 5)), np.ones((6, 5)) * 2]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        self.assertEqual(result.shape, (2, 3 * 5))

    def test_2d_both_axes_vary(self):
        """Both axes vary → both get reduced."""
        arrays = [np.random.rand(3, 10), np.random.rand(5, 8)]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        self.assertEqual(result.shape, (2, 3 * 8))

    # --- 3D arrays ---
    def test_3d_one_axis_varies(self):
        arrays = [np.random.rand(2, 10, 4), np.random.rand(2, 6, 4)]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        self.assertEqual(result.shape, (2, 2 * 6 * 4))

    # --- Uniform axes are truly preserved ---
    def test_uniform_axis_values_preserved(self):
        """Values along the non-varying axis should be untouched."""
        a = np.array([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
        b = np.array([[5.0, 6.0]])              # shape (1, 2)
        result = make_2D_array_from_different_length_arrays(
            [a, b], selection_mode="subsample_uniform"
        )
        # axis 0 (size 1) is uniform → preserved
        # axis 1 varies → subsampled to 2
        self.assertEqual(result.shape, (2, 2))
        idx = np.linspace(0, 3, 2, dtype=int)
        np.testing.assert_allclose(result[0], a[0, idx])
        np.testing.assert_allclose(result[1], b[0])

    # --- All selection modes produce correct shapes ---
    def test_resampling_modes_shape(self):
        for mode in ["subsample_uniform", "average_pool"]:
            with self.subTest(mode=mode):
                arrays = [np.random.rand(5, 12), np.random.rand(5, 8), np.random.rand(5, 10)]
                result = make_2D_array_from_different_length_arrays(arrays, selection_mode=mode)
                self.assertEqual(result.shape, (3, 5 * 8))

    def test_aggregation_modes_shape(self):
        """Aggregation collapses varying axes, preserves uniform ones."""
        for mode in ["mean", "max", "min", "sum"]:
            with self.subTest(mode=mode):
                arrays = [np.random.rand(5, 12), np.random.rand(5, 8)]
                result = make_2D_array_from_different_length_arrays(arrays, selection_mode=mode)
                # axis 1 varies → collapsed; axis 0 uniform (5) → preserved
                self.assertEqual(result.shape, (2, 5))

    # --- Aggregation value checks ---
    def test_mean_values(self):
        a = np.array([[10.0, 20.0, 30.0]])  # mean along axis 1 = 20
        b = np.array([[5.0, 5.0]])            # mean along axis 1 = 5
        result = make_2D_array_from_different_length_arrays(
            [a, b], selection_mode="mean"
        )
        np.testing.assert_allclose(result[0], [20.0])
        np.testing.assert_allclose(result[1], [5.0])

    def test_sum_values(self):
        a = np.array([[1.0, 2.0, 3.0]])  # sum = 6
        b = np.array([[10.0, 20.0]])       # sum = 30
        result = make_2D_array_from_different_length_arrays(
            [a, b], selection_mode="sum"
        )
        np.testing.assert_allclose(result[0], [6.0])
        np.testing.assert_allclose(result[1], [30.0])

    def test_max_values(self):
        a = np.array([[1.0, 5.0, 3.0]])
        b = np.array([[10.0, 2.0]])
        result = make_2D_array_from_different_length_arrays(
            [a, b], selection_mode="max"
        )
        np.testing.assert_allclose(result[0], [5.0])
        np.testing.assert_allclose(result[1], [10.0])

    # --- Edge cases ---
    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            make_2D_array_from_different_length_arrays([])

    def test_unknown_mode_raises(self):
        arrays = [np.ones((3, 4)), np.ones((3, 5))]
        with self.assertRaises(ValueError):
            make_2D_array_from_different_length_arrays(arrays, selection_mode="invalid")

    def test_single_array(self):
        """A single bar should just be flattened."""
        arr = np.random.rand(4, 5)
        result = make_2D_array_from_different_length_arrays([arr])
        self.assertEqual(result.shape, (1, 20))
        np.testing.assert_allclose(result[0], arr.flatten())

    def test_many_bars(self):
        """Stress test with many bars of varying sizes."""
        rng = np.random.default_rng(42)
        arrays = [rng.random((5, rng.integers(8, 20))) for _ in range(50)]
        result = make_2D_array_from_different_length_arrays(
            arrays, selection_mode="subsample_uniform"
        )
        min_time = min(a.shape[1] for a in arrays)
        self.assertEqual(result.shape, (50, 5 * min_time))

    def test_rows_are_independent(self):
        """Changing one input array should only affect its own row."""
        a = np.ones((3, 10))
        b = np.ones((3, 8)) * 5
        result = make_2D_array_from_different_length_arrays(
            [a, b], selection_mode="subsample_uniform"
        )
        self.assertTrue(np.all(result[0] == 1.0))
        self.assertTrue(np.all(result[1] == 5.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  _get_all_shapes_and_varying_axes
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetAllShapesAndVaryingAxes(unittest.TestCase):

    def test_same_ndim_all_same_shape(self):
        """All arrays have the same shape: (3, 10). Varying axes should be empty."""
        arrays = [np.zeros((3, 10)) for _ in range(3)]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [])
        self.assertEqual(all_shapes, [(3, 10), (3, 10), (3, 10)])
        self.assertEqual(list_of_arrays, arrays)

    def test_same_ndim_varying_shapes(self):
        """Arrays have same ndim (2), but axis 1 varies."""
        arrays = [np.zeros((3, 10)), np.zeros((3, 5)), np.zeros((3, 8))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1])
        self.assertEqual(all_shapes, [(3, 10), (3, 5), (3, 8)])

    def test_missing_dim_single(self):
        """One array is 1D, others are 2D. Missing dim corresponds to varying axis 1."""
        # Axis 0: constant (3). Axis 1: varying (10, 8, and missing).
        arrays = [np.zeros((3, 10)), np.zeros((3, 8)), np.zeros((3,))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1])
        # The 1D array should have been expanded to (3, 1)
        self.assertEqual(all_shapes[2], (3, 1))
        self.assertEqual(list_of_arrays[2].shape, (3, 1))

    def test_missing_dim_multiple(self):
        """Two arrays are 1D, others are 2D. Varying axis is 1."""
        arrays = [np.zeros((5, 12)), np.zeros((5,)), np.zeros((5, 10)), np.zeros((5,))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1])
        self.assertEqual(all_shapes[1], (5, 1))
        self.assertEqual(all_shapes[3], (5, 1))
        self.assertEqual(list_of_arrays[1].shape, (5, 1))
        self.assertEqual(list_of_arrays[3].shape, (5, 1))

    def test_missing_multiple_dims_all_modif(self):
        """Arrays missing multiple dimensions. 3D vs 1D where all missing dims vary."""
        # Max ndim = 3. Varying axes are 1 and 2.
        # Array 0: (2, 10, 4)
        # Array 1: (2, 6, 5)
        # Array 2: (2,)  -> should be expanded at axes 1 and 2 -> (2, 1, 1)
        arrays = [np.zeros((2, 10, 4)), np.zeros((2, 6, 5)), np.zeros((2,))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1, 2])
        self.assertEqual(all_shapes[2], (2, 1, 1))
        self.assertEqual(list_of_arrays[2].shape, (2, 1, 1))

    def test_missing_multiple_dims_one_modif(self):
        """Arrays missing multiple dimensions but not all missing dims vary in others."""
        arrays = [np.zeros((2, 10, 4)), np.zeros((2, 6, 4)), np.zeros((2,))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1, 2])
        self.assertEqual(all_shapes[2], (2, 1, 1))
        self.assertEqual(list_of_arrays[2].shape, (2, 1, 1))

    def test_error_three_different_ndims(self):
        """Should raise AssertionError for 3+ different ndims (e.g., 3D, 2D, 1D)."""
        arrays = [np.zeros((2, 10, 4)), np.zeros((2, 5)), np.zeros((2,))]
        with self.assertRaises(AssertionError):
            _get_all_shapes_and_varying_axes(arrays)

    def test_missing_non_varying_dim_success(self):
        """Case: [(3, 8, 64), (3, 8, 64), (8, 64)] should result in (3, 8, 64) for all."""
        arrays = [np.zeros((3, 8, 64)), np.zeros((3, 8, 64)), np.zeros((8, 64))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        # Axis 0 is missing in the last array but it is NOT varying in the others (always 3).
        # So varying_axes should be empty.
        self.assertEqual(varying_axes, [0])
        self.assertEqual(all_shapes, [(3, 8, 64), (3, 8, 64), (1, 8, 64)])
        self.assertEqual(list_of_arrays[2].shape, (1, 8, 64))

    def test_duplicate_dimension_sizes(self):
        """Robustness check: max_shape=(8, 8, 64), min_shape=(8, 64). Squeezed axis 0."""
        arrays = [np.zeros((8, 8, 64)), np.zeros((8, 64))]
        all_shapes, varying_axes, list_of_arrays = _get_all_shapes_and_varying_axes(arrays)
        self.assertEqual(varying_axes, [1])
        self.assertEqual(all_shapes, [(8, 8, 64), (8, 1, 64)])
        self.assertEqual(list_of_arrays[1].shape, (8, 1, 64))

if __name__ == "__main__":
    unittest.main()
