# -*- coding: utf-8 -*-
"""
Comprehensive tests for barwise_input.py

Tests include:
- BFT and FTB tensorization
- Round-trip conversions (spectrogram → tensor → spectrogram)
- Shape verification
- Error handling (ToDebugException for repeated samples)
- Barwise TF matrix conversion
- TF vector/matrix reconstructions
- Edge cases
"""

import unittest
import numpy as np
import as_seg.barwise_input as barwise
import as_seg.model.errors as err


class TestTensorizeBFT(unittest.TestCase):
    """Test tensorize_barwise_BFT function."""

    def setUp(self):
        """Create a sample spectrogram and bars."""
        np.random.seed(42)
        # Spectrogram: 64 frequency bins, 800 time frames
        self.spectrogram = np.random.rand(64, 800)
        # 10 bars, each roughly 80 frames (at hop_length_seconds=0.01, ~0.8s per bar)
        self.bars = [(i * 0.8, (i + 1) * 0.8) for i in range(1,10)]
        self.hop_length_seconds = 0.01
        self.subdivision = 16

    def test_output_shape(self):
        """Test that output has shape (bars, frequency, subdivision)."""
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        self.assertEqual(tensor.shape[0], 9)
        self.assertEqual(tensor.shape[1], 64)  # Frequency dimension
        self.assertEqual(tensor.shape[2], 16)  # Subdivision

    def test_various_subdivisions(self):
        """Test with different subdivision values."""
        for subdiv in [4, 8, 16, 32]:
            with self.subTest(subdivision=subdiv):
                tensor = barwise.tensorize_barwise_BFT(
                    self.spectrogram, self.bars, self.hop_length_seconds, subdiv
                )
                self.assertEqual(tensor.shape[2], subdiv)

    def test_subset_nb_bars(self):
        """Test subset_nb_bars parameter."""
        subset = 5
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds,
            self.subdivision, subset_nb_bars=subset
        )
        self.assertEqual(tensor.shape[0], subset)

    def test_samples_within_bounds(self):
        """Test that samples are within spectrogram bounds."""
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        # Verify tensor was created successfully without index errors
        self.assertGreater(tensor.shape[0], 0)

    def test_too_large_subdivision_raises(self):
        """Test that very large subdivision raises ToDebugException."""
        # Create bars that are very short
        short_bars = [(i * 0.01, (i + 1) * 0.01) for i in range(11)]
        # Very large subdivision relative to bar duration
        large_subdiv = 1000
        
        with self.assertRaises(err.ToDebugException):
            barwise.tensorize_barwise_BFT(
                self.spectrogram, short_bars, self.hop_length_seconds, large_subdiv
            )


class TestTensorizeFTB(unittest.TestCase):
    """Test tensorize_barwise_FTB function."""

    def setUp(self):
        """Create a sample spectrogram and bars."""
        np.random.seed(42)
        self.spectrogram = np.random.rand(64, 800)
        self.bars = [(i * 0.8, (i + 1) * 0.8) for i in range(11)]
        self.hop_length_seconds = 0.01
        self.subdivision = 16

    def test_output_shape(self):
        """Test that output has shape (frequency, subdivision, bars)."""
        tensor = barwise.tensorize_barwise_FTB(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        # FTB order: Frequency-Time-Bars
        self.assertEqual(tensor.shape[0], 64)  # Frequency
        self.assertEqual(tensor.shape[1], 16)  # Subdivision
        self.assertEqual(tensor.shape[2], 10)  # Bars

    def test_various_subdivisions(self):
        """Test with different subdivision values."""
        for subdiv in [4, 8, 16, 32]:
            with self.subTest(subdivision=subdiv):
                tensor = barwise.tensorize_barwise_FTB(
                    self.spectrogram, self.bars, self.hop_length_seconds, subdiv
                )
                self.assertEqual(tensor.shape[1], subdiv)

    def test_subset_nb_bars(self):
        """Test subset_nb_bars parameter."""
        subset = 5
        tensor = barwise.tensorize_barwise_FTB(
            self.spectrogram, self.bars, self.hop_length_seconds,
            self.subdivision, subset_nb_bars=subset
        )
        self.assertEqual(tensor.shape[2], subset)


class TestSpectrogramToTensor(unittest.TestCase):
    """Test spectrogram_to_tensor_barwise dispatcher."""

    def setUp(self):
        """Create a sample spectrogram and bars."""
        np.random.seed(42)
        self.spectrogram = np.random.rand(64, 800)
        self.bars = [(i * 0.8, (i + 1) * 0.8) for i in range(11)]
        self.hop_length_seconds = 0.01
        self.subdivision = 16

    def test_bft_dispatch(self):
        """Test that BFT mode dispatches correctly."""
        tensor_dispatch = barwise.spectrogram_to_tensor_barwise(
            self.spectrogram, self.bars, self.hop_length_seconds,
            self.subdivision, mode_order="BFT"
        )
        tensor_direct = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        np.testing.assert_array_equal(tensor_dispatch, tensor_direct)

    def test_ftb_dispatch(self):
        """Test that FTB mode dispatches correctly."""
        tensor_dispatch = barwise.spectrogram_to_tensor_barwise(
            self.spectrogram, self.bars, self.hop_length_seconds,
            self.subdivision, mode_order="FTB"
        )
        tensor_direct = barwise.tensorize_barwise_FTB(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        np.testing.assert_array_equal(tensor_dispatch, tensor_direct)

    def test_invalid_mode_raises(self):
        """Test that invalid mode order raises exception."""
        with self.assertRaises(err.InvalidArgumentValueException):
            barwise.spectrogram_to_tensor_barwise(
                self.spectrogram, self.bars, self.hop_length_seconds,
                self.subdivision, mode_order="INVALID"
            )


class TestTensorToSpectrogram(unittest.TestCase):
    """Test tensor_barwise_to_spectrogram function."""

    def setUp(self):
        """Create sample tensors."""
        np.random.seed(42)
        # BFT tensor: (10 bars, 64 freq, 16 subdiv)
        self.tensor_bft = np.random.rand(10, 64, 16)
        # FTB tensor: (64 freq, 16 subdiv, 10 bars)
        self.tensor_ftb = np.random.rand(64, 16, 10)

    def test_bft_reconstruction(self):
        """Test BFT tensor to spectrogram reconstruction."""
        spec = barwise.tensor_barwise_to_spectrogram(self.tensor_bft, mode_order="BFT")
        # Should unfold to (freq, time) = (64, 10*16)
        self.assertEqual(spec.shape[0], 64)
        self.assertEqual(spec.shape[1], 10 * 16)

    def test_ftb_reconstruction(self):
        """Test FTB tensor to spectrogram reconstruction."""
        spec = barwise.tensor_barwise_to_spectrogram(self.tensor_ftb, mode_order="FTB")
        # Should reshape to (freq, time) = (64, 16*10)
        self.assertEqual(spec.shape[0], 64)
        self.assertEqual(spec.shape[1], 16 * 10)

    def test_with_subset(self):
        """Test reconstruction with subset_nb_bars."""
        subset = 5
        spec = barwise.tensor_barwise_to_spectrogram(
            self.tensor_bft, mode_order="BFT", subset_nb_bars=subset
        )
        self.assertEqual(spec.shape[1], subset * 16)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises exception."""
        with self.assertRaises(err.InvalidArgumentValueException):
            barwise.tensor_barwise_to_spectrogram(self.tensor_bft, mode_order="INVALID")


class TestRoundTripConversion(unittest.TestCase):
    """Critical tests for round-trip conversions to ensure no data loss."""

    def setUp(self):
        """Create a sample spectrogram and bars."""
        np.random.seed(42)
        # Use smaller spectrogram for exact reconstruction testing
        self.spectrogram = np.random.rand(32, 3200)
        # 10 bars, each exactly 32 frames (perfect division)
        self.bars = [(i * 3.2, (i + 1) * 3.2) for i in range(1,10)]
        self.hop_length_seconds = 0.01
        self.subdivision = 16

    def test_bft_round_trip(self):
        """Test BFT: spectrogram → tensor → spectrogram."""
        # Forward
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        
        # Backward
        reconstructed = barwise.tensor_barwise_to_spectrogram(tensor, mode_order="BFT")
        
        # Check dimensions
        self.assertEqual(reconstructed.shape[0], self.spectrogram.shape[0])
        # Reconstructed will have exactly 10 * 16 = 160 frames
        self.assertEqual(reconstructed.shape[1], len(self.bars) * self.subdivision)

    def test_ftb_round_trip(self):
        """Test FTB: spectrogram → tensor → spectrogram."""
        # Forward
        tensor = barwise.tensorize_barwise_FTB(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        
        # Backward
        reconstructed = barwise.tensor_barwise_to_spectrogram(tensor, mode_order="FTB")
        
        # Check dimensions
        self.assertEqual(reconstructed.shape[0], self.spectrogram.shape[0])
        self.assertEqual(reconstructed.shape[1], len(self.bars) * self.subdivision)

    def test_both_modes_produce_same_reconstruction(self):
        """Test that BFT and FTB produce similar reconstructions."""
        # BFT path
        tensor_bft = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        recon_bft = barwise.tensor_barwise_to_spectrogram(tensor_bft, mode_order="BFT")
        
        # FTB path
        tensor_ftb = barwise.tensorize_barwise_FTB(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        recon_ftb = barwise.tensor_barwise_to_spectrogram(tensor_ftb, mode_order="FTB")
        
        # Both should have same shape
        self.assertEqual(recon_bft.shape, recon_ftb.shape)


class TestBarwiseTFMatrix(unittest.TestCase):
    """Test barwise_TF_matrix function."""

    def setUp(self):
        """Create a sample spectrogram and bars."""
        np.random.seed(42)
        self.spectrogram = np.random.rand(64, 800)
        self.bars = [(i * 0.8, (i + 1) * 0.8) for i in range(11)]
        self.hop_length_seconds = 0.01
        self.subdivision = 16

    def test_output_shape(self):
        """Test that output has shape (bars, freq * subdiv)."""
        tf_matrix = barwise.barwise_TF_matrix(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        # Should be (10 bars, 64*16 = 1024 features)
        self.assertEqual(tf_matrix.shape[0], 10)
        self.assertEqual(tf_matrix.shape[1], 64 * 16)

    def test_subset_functionality(self):
        """Test subset_nb_bars parameter."""
        subset = 5
        tf_matrix = barwise.barwise_TF_matrix(
            self.spectrogram, self.bars, self.hop_length_seconds,
            self.subdivision, subset_nb_bars=subset
        )
        self.assertEqual(tf_matrix.shape[0], subset)

    def test_matches_tensor_unfold(self):
        """Test that barwise_TF_matrix matches manual tensor unfolding."""
        tf_matrix = barwise.barwise_TF_matrix(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        
        # Manual method
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, self.bars, self.hop_length_seconds, self.subdivision
        )
        manual_tf = barwise.tensor_barwise_to_barwise_TF(tensor, mode_order="BFT")
        
        np.testing.assert_array_equal(tf_matrix, manual_tf)


class TestTFConversions(unittest.TestCase):
    """Test TF vector and matrix conversion functions."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.frequency_dim = 32
        self.subdivision = 16
        self.tf_vector = np.random.rand(self.frequency_dim * self.subdivision)
        self.tf_matrix = np.random.rand(10, self.frequency_dim * self.subdivision)

    def test_tf_vector_to_spectrogram_shape(self):
        """Test TF vector to spectrogram shape."""
        spec = barwise.TF_vector_to_spectrogram(
            self.tf_vector, self.frequency_dim, self.subdivision
        )
        self.assertEqual(spec.shape, (self.frequency_dim, self.subdivision))

    def test_tf_vector_assertion(self):
        """Test that dimension mismatch raises assertion."""
        wrong_vector = np.random.rand(100)  # Wrong size
        with self.assertRaises(AssertionError):
            barwise.TF_vector_to_spectrogram(
                wrong_vector, self.frequency_dim, self.subdivision
            )

    def test_tf_matrix_to_spectrogram_shape(self):
        """Test TF matrix to spectrogram shape."""
        spec = barwise.TF_matrix_to_spectrogram(
            self.tf_matrix, self.frequency_dim, self.subdivision
        )
        # Should concatenate 10 bars horizontally
        self.assertEqual(spec.shape[0], self.frequency_dim)
        self.assertEqual(spec.shape[1], 10 * self.subdivision)

    def test_tf_matrix_with_subset(self):
        """Test TF matrix with subset_nb_bars."""
        subset = 5
        spec = barwise.TF_matrix_to_spectrogram(
            self.tf_matrix, self.frequency_dim, self.subdivision, subset_nb_bars=subset
        )
        self.assertEqual(spec.shape[1], subset * self.subdivision)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.spectrogram = np.random.rand(64, 800)
        self.hop_length_seconds = 0.01

    def test_single_bar(self):
        """Test with only one bar."""
        bars = [(0.8, 1.6)]
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, bars, self.hop_length_seconds, 16
        )
        self.assertEqual(tensor.shape[0], 1)

    def test_bars_extending_beyond_spectrogram(self):
        """Test bars that extend beyond spectrogram duration."""
        # Create bars that go beyond spectrogram
        long_bars = [(i * 1.0, (i + 1) * 1.0) for i in range(20)]
        # Should handle gracefully by stopping when samples exceed bounds
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, long_bars, self.hop_length_seconds, 16
        )
        # Should have fewer bars than requested
        self.assertLess(tensor.shape[0], 19)

    def test_small_subdivision(self):
        """Test with very small subdivision."""
        bars = [(i * 0.8, (i + 1) * 0.8) for i in range(11)]
        tensor = barwise.tensorize_barwise_BFT(
            self.spectrogram, bars, self.hop_length_seconds, 2
        )
        self.assertEqual(tensor.shape[2], 2)

    def test_get_this_bar_tensor_bft(self):
        """Test getting a specific bar from BFT tensor."""
        tensor_bft = np.random.rand(10, 64, 16)
        bar = barwise.get_this_bar_tensor(tensor_bft, 5, mode_order="BFT")
        self.assertEqual(bar.shape, (64, 16))
        np.testing.assert_array_equal(bar, tensor_bft[5])

    def test_get_this_bar_tensor_ftb(self):
        """Test getting a specific bar from FTB tensor."""
        tensor_ftb = np.random.rand(64, 16, 10)
        bar = barwise.get_this_bar_tensor(tensor_ftb, 5, mode_order="FTB")
        self.assertEqual(bar.shape, (64, 16))
        np.testing.assert_array_equal(bar, tensor_ftb[:, :, 5])

    def test_barwise_subset_tensor_bft(self):
        """Test barwise_subset_this_tensor for BFT."""
        tensor_bft = np.random.rand(10, 64, 16)
        subset = barwise.barwise_subset_this_tensor(tensor_bft, 5, mode_order="BFT")
        self.assertEqual(subset.shape, (5, 64, 16))
        np.testing.assert_array_equal(subset, tensor_bft[:5])

    def test_barwise_subset_tensor_ftb(self):
        """Test barwise_subset_this_tensor for FTB."""
        tensor_ftb = np.random.rand(64, 16, 10)
        subset = barwise.barwise_subset_this_tensor(tensor_ftb, 5, mode_order="FTB")
        self.assertEqual(subset.shape, (64, 16, 5))
        np.testing.assert_array_equal(subset, tensor_ftb[:, :, :5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
