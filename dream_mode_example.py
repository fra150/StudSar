#!/usr/bin/env python3
"""Example demonstrating Dream Mode functionality in StudSar.

This script shows how to:
1. Set up a StudSar network with sample data
2. Configure and use Dream Mode
3. Monitor Dream Mode execution and results
4. Set up automatic scheduling
"""

import time
import logging
from src.managers.manager import StudSarManager
from src.managers.dream_mode import DreamModeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main demonstration function."""
    print("=== StudSar Dream Mode Demo ===")
    
    # 1. Initialize StudSar Manager
    print("\n1. Initializing StudSar Manager...")
    manager = StudSarManager()
    
    # 2. Add sample documents
    print("\n2. Adding sample documents...")
    sample_texts = [
        "Artificial intelligence is transforming how we interact with technology.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Deep learning neural networks mimic the structure of the human brain.",
        "Natural language processing enables computers to understand human speech.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning teaches AI agents through trial and error.",
        "Big data analytics helps organizations make data-driven decisions.",
        "Cloud computing provides scalable infrastructure for AI applications.",
        "Robotics combines AI with mechanical engineering for automation.",
        "Quantum computing may revolutionize certain AI algorithms."
    ]
    
    for i, text in enumerate(sample_texts):
        manager.update_network(text)
        print(f"  Added document {i+1}: {text[:50]}...")
    
    # 3. Simulate usage patterns
    print("\n3. Simulating usage patterns...")
    queries = [
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "data science",
        "computer vision"
    ]
    
    for query in queries:
        for _ in range(3):  # Simulate multiple searches
            results = manager.search(query, k=3)
            print(f"  Searched: '{query}' - Found {len(results[0])} results")
    
    # 4. Configure Dream Mode
    print("\n4. Configuring Dream Mode...")
    dream_config = {
        'high_usage_threshold': 5,
        'low_usage_threshold': 1,
        'low_reputation_threshold': 0.3,
        'min_markers_for_dream': 3,
        'max_pruning_percentage': 0.2,
        'promotion_boost': 1.5,
        'schedule_interval_hours': 1,  # Short interval for demo
        'enable_automatic_scheduling': False
    }
    
    dream_manager = DreamModeManager(manager.studsar_network, dream_config)
    print("  Dream Mode configured with custom settings")
    
    # 5. Analyze current marker statistics
    print("\n5. Analyzing marker statistics...")
    stats = dream_manager.analyze_marker_statistics()
    print(f"  High usage markers: {len(stats['high_usage_markers'])}")
    print(f"  Low usage markers: {len(stats['low_usage_markers'])}")
    print(f"  Low reputation markers: {len(stats['low_reputation_markers'])}")
    
    # 6. Execute Dream Mode manually
    print("\n6. Executing Dream Mode...")
    initial_markers = len(manager.studsar_network.id_to_segment)
    print(f"  Initial marker count: {initial_markers}")
    
    result = dream_manager.run_dream_mode()
    
    if result['executed']:
        print("  ✓ Dream Mode executed successfully!")
        print(f"  Execution time: {result['execution_time']:.2f} seconds")
        print(f"  Markers before: {result['total_markers_before']}")
        print(f"  Markers after: {result['total_markers_after']}")
        print(f"  Promoted markers: {result['promoted_count']}")
        print(f"  Pruned markers: {result['pruned_count']}")
    else:
        print(f"  ✗ Dream Mode not executed: {result.get('reason', 'Unknown')}")
    
    # 7. Test network functionality after Dream Mode
    print("\n7. Testing network after Dream Mode...")
    test_query = "artificial intelligence applications"
    results = manager.search(test_query, k=3)
    print(f"  Search for '{test_query}': {len(results[0])} results found")
    
    if results[0]:
        print(f"  Best match: {results[2][0][:60]}...")
    
    # 8. Demonstrate automatic scheduling
    print("\n8. Demonstrating automatic scheduling...")
    print("  Starting automatic Dream Mode scheduler...")
    
    # Configure for very short interval for demo
    dream_config['schedule_interval_hours'] = 0.001  # ~3.6 seconds
    dream_manager_auto = DreamModeManager(manager.studsar_network, dream_config)
    
    dream_manager_auto.start_automatic_scheduling()
    
    # Let it run for a few seconds
    print("  Scheduler running... (waiting 10 seconds)")
    time.sleep(10)
    
    # Check status
    status = dream_manager_auto.get_status()
    print(f"  Scheduler status: {'Running' if status['scheduler_running'] else 'Stopped'}")
    print(f"  Current markers: {status['total_markers']}")
    
    # Stop scheduler
    dream_manager_auto.stop_automatic_scheduling()
    print("  Scheduler stopped")
    
    # 9. Save the optimized network
    print("\n9. Saving optimized network...")
    save_path = "studsar_optimized_demo.pth"
    manager.save(save_path)
    print(f"  Network saved to: {save_path}")
    
    # 10. Final statistics
    print("\n10. Final Statistics:")
    final_stats = dream_manager.get_status()
    print(f"  Total markers: {final_stats['total_markers']}")
    print(f"  Dream Mode config: {final_stats['config']}")
    
    print("\n=== Demo Complete ===")
    print("\nDream Mode Benefits Demonstrated:")
    print("  ✓ Automatic marker analysis and categorization")
    print("  ✓ Promotion of frequently used markers")
    print("  ✓ Pruning of low-value markers")
    print("  ✓ Configurable thresholds and limits")
    print("  ✓ Automatic scheduling capability")
    print("  ✓ Network optimization without data loss")

if __name__ == "__main__":
    main()