import unittest
import numpy as np
from as_seg.data_manipulation import (
    segments_to_frontiers,
    frontiers_to_segments,
    frontiers_from_bar_to_time,
    frontiers_from_time_to_bar,
    segments_from_bar_to_time,
    segments_from_time_to_bar,
    align_frontiers_on_bars,
    align_segments_on_bars
)
class TestDataManipulation(unittest.TestCase):

    def test_segments_frontiers_conversion_invariance(self):
        """
        Test that doing segments to frontiers, and then frontiers to segments,
        multiple times does not alter the content of the arrays.
        """
        # Define initial segments
        original_segments = np.array([
            [0.0, 1.5],
            [1.5, 3.2],
            [3.2, 4.0],
            [4.0, 5.5],
            [5.5, 7.1]
        ])

        # Convert to frontiers
        frontiers_1 = segments_to_frontiers(original_segments)
        
        # Convert back to segments
        segments_1 = frontiers_to_segments(frontiers_1)
        
        # Convert to frontiers again
        frontiers_2 = segments_to_frontiers(segments_1)
        
        # Convert back to segments again
        segments_2 = frontiers_to_segments(frontiers_2)

        # Assertions
        np.testing.assert_array_almost_equal(original_segments, segments_1, err_msg="Segments altered after first conversion cycle")
        np.testing.assert_array_almost_equal(frontiers_1, frontiers_2, err_msg="Frontiers altered between multiple conversions")
        np.testing.assert_array_almost_equal(segments_1, segments_2, err_msg="Segments altered between multiple conversions")

    def test_frontiers_segments_conversion_invariance(self):
        """
        Test that doing frontiers to segments, and then segments to frontiers,
        multiple times does not alter the content of the arrays.
        """
        # Define initial frontiers
        original_frontiers = np.array([0.0, 1.2, 3.4, 4.5, 6.7, 8.9])

        # Convert to segments
        segments_1 = frontiers_to_segments(original_frontiers)
        
        # Convert back to frontiers
        frontiers_1 = segments_to_frontiers(segments_1)
        
        # Convert to segments again
        segments_2 = frontiers_to_segments(frontiers_1)
        
        # Convert back to frontiers again
        frontiers_2 = segments_to_frontiers(segments_2)

        # Assertions
        np.testing.assert_array_almost_equal(original_frontiers, frontiers_1, err_msg="Frontiers altered after first conversion cycle")
        np.testing.assert_array_almost_equal(segments_1, segments_2, err_msg="Segments altered between multiple conversions")
        np.testing.assert_array_almost_equal(frontiers_1, frontiers_2, err_msg="Frontiers altered between multiple conversions")


class TestBarTimeConversion(unittest.TestCase):
    def setUp(self):
        # Fake bars defined as (start, end) time tuples
        self.fake_bars = [
            (0.0, 1.5),
            (1.5, 3.0),
            (3.0, 4.2),
            (4.2, 5.5),
            (5.5, 7.0)
        ]

    def test_frontiers_bar_to_time(self):
        """Test converting frontier indices back to absolute time.
        Boundary `i` is the start of bar `i`."""
        fake_frontier_indices = [0, 2, 4, 5]
        expected_times = [0.0, 3.0, 5.5, 7.0]

        times = frontiers_from_bar_to_time(fake_frontier_indices, self.fake_bars)
        self.assertEqual(times, expected_times)

    def test_segments_bar_to_time(self):
        """Test converting segment indices to segment absolute times."""
        # Using lists as inputs for segments array format
        fake_segment_indices = [[0, 2], [2, 4], [4, 5]]
        expected_segments = np.array([
            [0.0, 3.0],
            [3.0, 5.5],
            [5.5, 7.0]
        ])

        times = segments_from_bar_to_time(fake_segment_indices, self.fake_bars)
        np.testing.assert_array_almost_equal(times, expected_segments)

    def test_frontiers_time_to_bar(self):
        """Test mapping absolute time to the closest bar index/frontiers."""
        fake_times = [0.0, 1.5, 4.2, 7.0]
        expected_indices = [0, 1, 3, 5]

        indices = frontiers_from_time_to_bar(fake_times, self.fake_bars)
        
        # Testing align_frontiers_on_bars returns the same indices
        # align_frontiers_on_bars realigns the frontiers on the bars
        realigned_frontiers = align_frontiers_on_bars(fake_times, self.fake_bars)

        np.testing.assert_array_almost_equal(indices, expected_indices, err_msg="Frontiers not indexed properly on bars")
        np.testing.assert_array_almost_equal(realigned_frontiers, fake_times, err_msg="Frontiers not realigned correctly")

    def test_segments_time_to_bar(self):
        """Test converting segments in time to bar index segments."""
        fake_time_segments = np.array([
            [0.0, 1.5],
            [1.5, 4.2],
            [4.2, 7.0]
        ])

        expected_index_segments = np.array([
            [0, 1],
            [1, 3],
            [3, 5]
        ])

        indices = segments_from_time_to_bar(fake_time_segments, self.fake_bars)
        np.testing.assert_array_equal(indices, expected_index_segments)
        
    def test_align_segments_on_bars(self):
        """Test realigning segments on the exact bar bounds."""
        fake_time_segments = np.array([
            [0.1, 1.6],
            [1.6, 4.1],
            [4.1, 7.1]
        ])
        expected_time_segments_aligned = np.array([
            [0.0, 1.5],
            [1.5, 4.2],
            [4.2, 7.0]
        ])

        # BARS:
        # (0.0, 1.5),
        # (1.5, 3.0),
        # (3.0, 4.2),
        # (4.2, 5.5),
        # (5.5, 7.0)

        aligned_segments = align_segments_on_bars(fake_time_segments, self.fake_bars)
        np.testing.assert_array_almost_equal(aligned_segments, expected_time_segments_aligned)

    def test_invariance_time_to_bar_to_time(self):
        """Test conversions are stable when going from time -> bar index -> time multiple times."""
        # Using exact boundaries to ensure clean matching
        fake_time_segments = np.array([
            [0.0, 1.5],
            [1.5, 3.0],
            [3.0, 5.5],
            [5.5, 7.0]
        ])



        # cycle 1
        bars_1 = segments_from_time_to_bar(fake_time_segments, self.fake_bars)
        times_1 = segments_from_bar_to_time(bars_1, self.fake_bars)
        
        # cycle 2
        bars_2 = segments_from_time_to_bar(times_1, self.fake_bars)
        times_2 = segments_from_bar_to_time(bars_2, self.fake_bars)

        np.testing.assert_array_almost_equal(fake_time_segments, times_1)
        np.testing.assert_array_equal(bars_1, bars_2)
        np.testing.assert_array_almost_equal(times_1, times_2)

    def test_out_of_bounds_frontiers(self):
        """Test how absolute times outside the bar grid are handled."""
        fake_times = [-1.0, 8.0]
        # -1.0 < bars[0][0] (which is 0.0) -> index 0
        # 8.0 > bars[-1][1] (which is 7.0) -> index len(bars) = 5
        expected_indices = [0, 5]
        
        indices = frontiers_from_time_to_bar(fake_times, self.fake_bars)
        np.testing.assert_array_equal(indices, expected_indices)
        
        # align_frontiers_on_bars handles out-of-bound
        # -1.0 -> bars[0][0] (0.0)
        # 8.0 -> bars[-1][1] (7.0)
        expected_times = [0.0, 7.0]
        realigned_times = align_frontiers_on_bars(fake_times, self.fake_bars)
        np.testing.assert_array_equal(realigned_times, expected_times)

    def test_collapsing_short_segments(self):
        """Test that segments starting and ending near the same boundary disappear correctly."""
        # This segment is centered tightly around the boundary 3.0 (bar index 2)
        fake_time_segments = np.array([
            [2.9, 3.1],
        ])
        
        # Both start and end fall closest to boundary 3.0 (index 2). 
        # Since they map to the same frontier, it doesn't form a segment.
        indices = segments_from_time_to_bar(fake_time_segments, self.fake_bars)
        self.assertEqual(len(indices), 0, "Segment should have been collapsed and removed.")
        
        aligned_segments = align_segments_on_bars(fake_time_segments, self.fake_bars)
        self.assertEqual(len(aligned_segments), 0, "Aligned segment should have been collapsed and removed.")

if __name__ == '__main__':
    unittest.main()