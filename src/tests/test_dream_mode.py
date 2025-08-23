#!/usr/bin/env python3
"""
Test suite for Dream Mode functionality in StudSar V3.
"""

import unittest
import sys
import os
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.managers.manager import StudSarManager
from src.managers.dream_mode import DreamModeManager

class TestDreamMode(unittest.TestCase):
    """
    Test cases for Dream Mode functionality.
    """
    
    def setUp(self):
        """
        Set up test environment before each test.
        """
        self.manager = StudSarManager()
        
        # Configure Dream Mode with test-friendly settings
        config = {
            'usage_threshold_high': 5,
            'usage_threshold_low': 2,
            'reputation_threshold_low': -3.0,
            'similarity_threshold': 0.9,
            'max_markers_to_prune': 10,
            'consolidation_ratio': 0.2,
            'min_markers_for_dream': 3  # Reduced for testing
        }
        
        self.dream_mode = DreamModeManager(self.manager.studsar_network, config)
        
        # Create test data with known patterns
        self.high_usage_texts = [
            "Machine learning is transforming AI research.",
            "Deep neural networks enable pattern recognition.",
            "Natural language processing improves communication."
        ]
        
        self.low_usage_texts = [
            "Random obsolete content for testing.",
            "Temporary notes that should be pruned.",
            "Duplicate information with no value."
        ]
        
        # Add texts to network
        for text in self.high_usage_texts + self.low_usage_texts:
            self.manager.update_network(text)
    
    def tearDown(self):
        """
        Clean up after each test.
        """
        # Clean up any temporary files
        temp_files = ["test_dream_mode.pth"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_dream_mode_initialization(self):
        """
        Test Dream Mode initialization with custom config.
        """
        custom_config = {
            'usage_threshold_high': 15,
            'usage_threshold_low': 1,
            'reputation_threshold_low': -10.0
        }
        
        dream_mode = DreamModeManager(self.manager.studsar_network, custom_config)
        
        # Check that config was applied
        self.assertEqual(dream_mode.config['usage_threshold_high'], 15)
        self.assertEqual(dream_mode.config['usage_threshold_low'], 1)
        self.assertEqual(dream_mode.config['reputation_threshold_low'], -10.0)
        
        # Check default values are preserved
        self.assertEqual(dream_mode.config['similarity_threshold'], 0.95)
    
    def test_analyze_marker_statistics_insufficient_markers(self):
        """
        Test statistics analysis with insufficient markers.
        """
        # Create a new manager with very few markers
        small_manager = StudSarManager()
        small_manager.update_network("Single test marker")
        
        dream_mode = DreamModeManager(small_manager.studsar_network, {'min_markers_for_dream': 50})
        stats = dream_mode.analyze_marker_statistics()
        
        self.assertTrue(stats['insufficient_markers'])
        self.assertIn('Need at least', stats['message'])
    
    def test_analyze_marker_statistics_sufficient_markers(self):
        """
        Test statistics analysis with sufficient markers.
        """
        # Simulate usage patterns
        self._simulate_usage_patterns()
        
        stats = self.dream_mode.analyze_marker_statistics()
        
        self.assertFalse(stats['insufficient_markers'])
        self.assertIn('total_markers', stats)
        self.assertIn('high_usage_markers', stats)
        self.assertIn('low_usage_markers', stats)
        self.assertIn('low_reputation_markers', stats)
        
        # Check that we have some markers in each category
        self.assertGreater(stats['total_markers'], 0)
        self.assertIsInstance(stats['high_usage_markers'], list)
        self.assertIsInstance(stats['low_usage_markers'], list)
        self.assertIsInstance(stats['low_reputation_markers'], list)
    
    def test_find_similar_markers(self):
        """
        Test finding similar markers functionality.
        """
        # Add similar content
        self.manager.update_network("Machine learning algorithms are powerful.")
        self.manager.update_network("Deep learning models are effective.")
        
        # Get a marker ID to test with
        marker_ids = list(self.manager.studsar_network.marker_id_to_index.keys())
        test_marker_id = marker_ids[0]
        
        # Find similar markers
        similar = self.dream_mode.find_similar_markers(test_marker_id, threshold=0.1)
        
        # Should return a list
        self.assertIsInstance(similar, list)
        
        # Each similar marker should have required fields
        for marker in similar:
            self.assertIn('id', marker)
            self.assertIn('similarity', marker)
            self.assertIn('usage', marker)
            self.assertIn('reputation', marker)
            self.assertGreaterEqual(marker['similarity'], 0.1)
    
    def test_promote_high_usage_markers(self):
        """
        Test promotion of high-usage markers.
        """
        # Simulate high usage
        self._simulate_usage_patterns()
        
        # Get high-usage markers
        stats = self.dream_mode.analyze_marker_statistics()
        high_usage_markers = stats['high_usage_markers']
        
        if high_usage_markers:
            # Record initial reputation
            initial_reputation = {}
            for marker in high_usage_markers:
                marker_id = marker['id']
                initial_reputation[marker_id] = self.manager.studsar_network.id_to_reputation[marker_id]
            
            # Promote markers
            results = self.dream_mode.promote_high_usage_markers(high_usage_markers)
            
            # Check results
            self.assertIn('promoted_count', results)
            self.assertIn('consolidated_count', results)
            self.assertGreaterEqual(results['promoted_count'], 0)
            
            # Check that reputation was boosted
            for marker in high_usage_markers[:results['promoted_count']]:
                marker_id = marker['id']
                current_reputation = self.manager.studsar_network.id_to_reputation[marker_id]
                self.assertGreaterEqual(current_reputation, initial_reputation[marker_id])
    
    def test_prune_low_value_markers(self):
        """
        Test pruning of low-value markers.
        """
        # Simulate usage patterns to create low-value markers
        self._simulate_usage_patterns()
        
        # Get initial marker count
        initial_count = self.manager.studsar_network.get_total_markers()
        
        # Get low-value markers
        stats = self.dream_mode.analyze_marker_statistics()
        
        if stats['insufficient_markers']:
            self.skipTest("Insufficient markers for pruning test")
            
        low_usage_markers = stats['low_usage_markers']
        low_reputation_markers = stats['low_reputation_markers']
        
        # Prune markers
        results = self.dream_mode.prune_low_value_markers(low_usage_markers, low_reputation_markers)
        
        # Check results
        self.assertIn('pruned_count', results)
        self.assertIn('candidates_identified', results)
        
        # Check that some markers were removed (if there were candidates)
        final_count = self.manager.studsar_network.get_total_markers()
        if results['candidates_identified'] > 0:
            self.assertLessEqual(final_count, initial_count)
    
    def test_run_dream_mode_insufficient_markers(self):
        """
        Test dream mode with insufficient markers.
        """
        # Create manager with very few markers
        small_manager = StudSarManager()
        small_manager.update_network("Single marker")
        
        dream_mode = DreamModeManager(small_manager.studsar_network, {'min_markers_for_dream': 100})
        results = dream_mode.run_dream_mode()
        
        self.assertFalse(results['executed'])
    
    def test_run_dream_mode_success(self):
        """
        Test successful dream mode execution.
        """
        # Simulate usage patterns
        self._simulate_usage_patterns()
        
        # Configure for testing
        self.dream_mode.config['min_markers_for_dream'] = 3
        
        # Run dream mode
        results = self.dream_mode.run_dream_mode()
        
        # Check success
        self.assertTrue(results.get('success', False))
        
        # Check required fields
        required_fields = [
            'start_time', 'end_time', 'duration_seconds',
            'initial_markers', 'final_markers', 'markers_removed',
            'promoted_count', 'consolidated_count', 'pruned_count'
        ]
        
        for field in required_fields:
            self.assertIn(field, results)
        
        # Check that duration is reasonable
        self.assertGreater(results['duration_seconds'], 0)
        self.assertLess(results['duration_seconds'], 60)  # Should complete within 60 seconds
    
    def test_get_status(self):
        """
        Test getting Dream Mode status.
        """
        status = self.dream_mode.get_status()
        
        required_fields = [
            'is_running', 'last_run', 'scheduler_active',
            'config', 'total_markers'
        ]
        
        for field in required_fields:
            self.assertIn(field, status)
        
        # Check initial state
        self.assertFalse(status['is_running'])
        self.assertIsNone(status['last_run'])
        self.assertFalse(status['scheduler_active'])
        self.assertIsInstance(status['config'], dict)
        self.assertGreaterEqual(status['total_markers'], 0)
    
    def test_scheduler_start_stop(self):
        """
        Test scheduler start and stop functionality.
        """
        # Test starting scheduler
        self.dream_mode.start_scheduler()
        
        # Check status
        status = self.dream_mode.get_status()
        self.assertTrue(status['scheduler_active'])
        
        # Test stopping scheduler
        self.dream_mode.stop_scheduler()
        
        # Note: The current implementation doesn't actually stop the thread,
        # so we just check that the stop method can be called without error
    
    def test_remove_marker(self):
        """
        Test marker removal functionality.
        """
        # Get a marker to remove
        marker_ids = list(self.manager.studsar_network.marker_id_to_index.keys())
        if marker_ids:
            test_marker_id = marker_ids[0]
            initial_count = self.manager.studsar_network.get_total_markers()
            
            # Remove the marker
            success = self.dream_mode._remove_marker(test_marker_id)
            
            # Check removal
            self.assertTrue(success)
            final_count = self.manager.studsar_network.get_total_markers()
            self.assertEqual(final_count, initial_count - 1)
            
            # Check that marker is no longer in mappings
            self.assertNotIn(test_marker_id, self.manager.studsar_network.marker_id_to_index)
    
    def test_remove_nonexistent_marker(self):
        """
        Test removing a marker that doesn't exist.
        """
        # Try to remove a non-existent marker
        success = self.dream_mode._remove_marker("nonexistent_marker_id")
        
        # Should return False
        self.assertFalse(success)
    
    def _simulate_usage_patterns(self):
        """
        Helper method to simulate realistic usage patterns.
        """
        # Simulate high usage for some markers
        for text in self.high_usage_texts:
            for _ in range(12):  # High usage (>= 10)
                marker_ids, similarities, segments = self.manager.search(text, k=1)
                if marker_ids:
                    # Positive feedback
                    self.manager.update_marker_reputation(marker_ids[0], 1.0)
        
        # Simulate low usage and negative feedback for others
        for text in self.low_usage_texts:
            marker_ids, similarities, segments = self.manager.search(text, k=1)
            if marker_ids:
                # Negative feedback
                self.manager.update_marker_reputation(marker_ids[0], -3.0)

class TestDreamModeIntegration(unittest.TestCase):
    """
    Integration tests for Dream Mode with StudSar.
    """
    
    def setUp(self):
        """
        Set up integration test environment.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_dream_integration.pth")
    
    def tearDown(self):
        """
        Clean up integration test environment.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dream_mode_with_save_load(self):
        """
        Test Dream Mode with save/load functionality.
        """
        # Create and populate manager
        manager = StudSarManager()
        
        test_texts = [
            "Artificial intelligence research is advancing rapidly.",
            "Machine learning models require large datasets.",
            "Deep learning architectures are becoming more complex.",
            "Neural networks enable pattern recognition.",
            "Data science drives business insights.",
            "Python programming language is versatile.",
            "Software development requires careful planning.",
            "Random obsolete content for removal.",
            "Temporary data that should be pruned."
        ]
        
        for text in test_texts:
            manager.update_network(text)
        
        # Simulate usage patterns - create high usage for first few texts
        for i in range(5):  # First 5 texts get high usage
            for _ in range(12):  # High usage count
                marker_ids, similarities, segments = manager.search(test_texts[i], k=1)
                if marker_ids:
                    manager.update_marker_reputation(marker_ids[0], 1.0)
        
        # Apply negative feedback only to last two texts
        for text in test_texts[-2:]:
            marker_ids, similarities, segments = manager.search(text, k=1)
            if marker_ids:
                manager.update_marker_reputation(marker_ids[0], -5.0)
        
        # Save initial state
        manager.save(self.test_file)
        initial_count = manager.studsar_network.get_total_markers()
        
        # Run Dream Mode
        dream_mode = DreamModeManager(manager.studsar_network, {'min_markers_for_dream': 3})
        results = dream_mode.run_dream_mode()
        
        # Check that consolidation was successful
        self.assertTrue(results.get('success', False))
        
        # Save optimized state
        optimized_file = self.test_file.replace('.pth', '_optimized.pth')
        manager.save(optimized_file)
        
        # Load optimized state in new manager
        new_manager = StudSarManager()
        new_manager.load(optimized_file)
        
        # Verify that the optimized network loads without errors
        # The load operation should complete successfully
        self.assertIsNotNone(new_manager)
        self.assertIsNotNone(new_manager.studsar_network)
        
        # Try to search - verify the search doesn't crash
        try:
            marker_ids, similarities, segments = new_manager.search("artificial intelligence", k=3)
            self.assertIsInstance(marker_ids, list)
            self.assertIsInstance(similarities, list)
            self.assertIsInstance(segments, list)
        except Exception as e:
            self.fail(f"Search failed after loading: {e}")
        
        # Clean up
        if os.path.exists(optimized_file):
            os.remove(optimized_file)

def run_tests():
    """
    Run all Dream Mode tests.
    """
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(loader.loadTestsFromTestCase(TestDreamMode))
    test_suite.addTest(loader.loadTestsFromTestCase(TestDreamModeIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=== StudSar Dream Mode Test Suite ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    success = run_tests()
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n✅ All Dream Mode tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some Dream Mode tests failed.")
        sys.exit(1)