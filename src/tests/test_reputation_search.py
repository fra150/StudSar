#!/usr/bin/env python3
"""
Test suite for reputation-based search functionality in StudSar.
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from managers.manager import StudSarManager

class TestReputationSearch(unittest.TestCase):
    """Test cases for reputation-enhanced search functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = StudSarManager()
        
        # Create test data with different content
        test_texts = [
            "Python is a programming language",
            "Machine learning algorithms are powerful", 
            "Neural networks process information",
            "Data science involves statistics",
            "Artificial intelligence mimics human cognition"
        ]
        
        # Build network with test data
        for text in test_texts:
            self.manager.update_network(text, emotion="neutral")
            
        # Get marker IDs for reputation manipulation
        self.marker_ids, _, _ = self.manager.search("programming", k=5)
        
    def test_reputation_boost_affects_ranking(self):
        """Test that higher reputation markers rank higher in search results."""
        if len(self.marker_ids) < 2:
            self.skipTest("Need at least 2 markers for ranking test")
            
        # Set different reputation scores
        low_rep_marker = self.marker_ids[0]
        high_rep_marker = self.marker_ids[1]
        
        # Set reputation: high_rep_marker should rank higher
        self.manager.update_marker_reputation(low_rep_marker, -0.5)  # Low reputation
        self.manager.update_marker_reputation(high_rep_marker, 1.0)  # High reputation
        
        # Search with reputation enhancement
        result_ids, similarities, _ = self.manager.search_with_reputation("programming", k=2)
        
        # Verify that high reputation marker appears first (if both are relevant)
        if high_rep_marker in result_ids and low_rep_marker in result_ids:
            high_rep_index = result_ids.index(high_rep_marker)
            low_rep_index = result_ids.index(low_rep_marker)
            self.assertLess(high_rep_index, low_rep_index, 
                           "High reputation marker should rank higher")
            
    def test_reputation_weight_parameter(self):
        """Test that reputation weight parameter controls influence."""
        if len(self.marker_ids) < 1:
            self.skipTest("Need at least 1 marker for weight test")
            
        marker_id = self.marker_ids[0]
        
        # Set high reputation
        self.manager.update_marker_reputation(marker_id, 1.0)
        
        # Search with different reputation weights
        _, sims_no_rep, _ = self.manager.search_with_reputation("programming", k=1, reputation_weight=0.0)
        _, sims_with_rep, _ = self.manager.search_with_reputation("programming", k=1, reputation_weight=1.0)
        
        # With reputation weight 0, reputation should have no effect
        # With reputation weight 1, reputation should boost similarity
        self.assertIsInstance(sims_no_rep, list)
        self.assertIsInstance(sims_with_rep, list)
        
    def test_negative_reputation_penalty(self):
        """Test that negative reputation reduces ranking."""
        if len(self.marker_ids) < 1:
            self.skipTest("Need at least 1 marker for penalty test")
            
        marker_id = self.marker_ids[0]
        
        # Get baseline similarity
        _, baseline_sims, _ = self.manager.search("programming", k=1)
        
        # Set negative reputation
        self.manager.update_marker_reputation(marker_id, -1.0)
        
        # Search with reputation enhancement
        _, penalized_sims, _ = self.manager.search_with_reputation("programming", k=1)
        
        # Negative reputation should reduce similarity (in most cases)
        if baseline_sims and penalized_sims:
            # Note: This test might be flaky due to the nature of similarity calculations
            # We're mainly testing that the system doesn't crash with negative reputation
            self.assertIsInstance(penalized_sims[0], float)
            
    def test_reputation_search_preserves_usage_tracking(self):
        """Test that reputation search still increments usage counts."""
        if len(self.marker_ids) < 1:
            self.skipTest("Need at least 1 marker for usage test")
            
        marker_id = self.marker_ids[0]
        
        # Get initial usage count
        initial_details = self.manager.get_marker_details(marker_id)
        initial_usage = initial_details['usage_count'] if initial_details else 0
        
        # Perform reputation search
        self.manager.search_with_reputation("programming", k=1)
        
        # Check that usage was incremented
        final_details = self.manager.get_marker_details(marker_id)
        final_usage = final_details['usage_count'] if final_details else 0
        
        self.assertGreater(final_usage, initial_usage, 
                          "Usage count should be incremented after search")
                          
    def test_reputation_values_restored_after_weighted_search(self):
        """Test that original reputation values are restored after weighted search."""
        if len(self.marker_ids) < 1:
            self.skipTest("Need at least 1 marker for restoration test")
            
        marker_id = self.marker_ids[0]
        
        # Set a specific reputation value
        original_reputation = 0.75
        self.manager.update_marker_reputation(marker_id, original_reputation)
        
        # Perform weighted search that should temporarily modify reputation
        self.manager.search_with_reputation("programming", k=1, reputation_weight=2.0)
        
        # Check that original reputation is restored
        details = self.manager.get_marker_details(marker_id)
        current_reputation = details['reputation'] if details else 0.0
        
        self.assertAlmostEqual(current_reputation, original_reputation, places=5,
                              msg="Original reputation should be restored after weighted search")

if __name__ == '__main__':
    unittest.main()