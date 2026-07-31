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

    def test_curated_colors_returned(self):
        distinct_colors = [
            "#0072B2",  # Deep Blue
            "#E69F00",  # Warm Orange
            "#009E73",  # Bluish Green
            "#D55E00",  # Vermillion/Red-Orange
            "#CC79A7",  # Reddish Purple/Pink
            "#56B4E9",  # Sky Blue
            "#332288",  # Indigo/Dark Purple
            "#CC6677",  # Rose/Dusty Red
            "#999933",  # Olive/Yellow-Green
            "#117733",  # Dark Green
            "#882255",  # Wine/Burgundy
            "#44AA99",  # Teal
            "#AA4499",  # Purple
            "#DDCC77",  # Sand/Tan
            "#661100"   # Dark Brown/Maroon
        ]
        
        # Ensure exact curated colors are returned for N <= 15
        for n in range(1, 16):
            self.assertEqual(generate_dynamic_colors(n), distinct_colors[:n])
            
    def test_larger_n_uniqueness_and_formatting(self):
        distinct_colors = [
            "#0072B2",  # Deep Blue
            "#E69F00",  # Warm Orange
            "#009E73",  # Bluish Green
            "#D55E00",  # Vermillion/Red-Orange
            "#CC79A7",  # Reddish Purple/Pink
            "#56B4E9",  # Sky Blue
            "#332288",  # Indigo/Dark Purple
            "#CC6677",  # Rose/Dusty Red
            "#999933",  # Olive/Yellow-Green
            "#117733",  # Dark Green
            "#882255",  # Wine/Burgundy
            "#44AA99",  # Teal
            "#AA4499",  # Purple
            "#DDCC77",  # Sand/Tan
            "#661100"   # Dark Brown/Maroon
        ]
        
        n = 20
        colors = generate_dynamic_colors(n)
        self.assertEqual(len(colors), n)
        
        # Verify first 15 are the curated colors
        self.assertEqual(colors[:15], distinct_colors)
        
        # Verify formatting of all colors
        for col in colors:
            self.assertTrue(col.startswith('#'))
            self.assertEqual(len(col), 7)
            int(col[1:], 16)
            
        # Verify uniqueness
        self.assertEqual(len(set(colors)), n)

if __name__ == "__main__":
    unittest.main()
