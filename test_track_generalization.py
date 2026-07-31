import unittest
from track_generalization import generate_dynamic_colors

class TestTrackGeneralization(unittest.TestCase):
    def test_generate_dynamic_colors_exists(self):
        # Test basic output length
        colors = generate_dynamic_colors(5)
        self.assertEqual(len(colors), 5)
        
        # Test color formatting (should be hex strings, e.g. #abcdef)
        for col in colors:
            self.assertTrue(col.startswith('#'))
            self.assertEqual(len(col), 7)
            # Check hex digits
            int(col[1:], 16)
            
        # Uniqueness for small values of n
        colors_10 = generate_dynamic_colors(10)
        self.assertEqual(len(colors_10), 10)
        self.assertEqual(len(set(colors_10)), 10)

    def test_generate_dynamic_colors_zero_or_negative(self):
        # Handle n <= 0 gracefully by returning []
        self.assertEqual(generate_dynamic_colors(0), [])
        self.assertEqual(generate_dynamic_colors(-5), [])

if __name__ == "__main__":
    unittest.main()
