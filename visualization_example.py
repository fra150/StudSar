#!/usr/bin/env python3
"""
Advanced Visualization Example for StudSar
Demonstrates the enhanced visualization capabilities with Dream Mode insights.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from managers.manager import StudSarManager
from managers.dream_mode import DreamModeManager

def main():
    print("=== StudSar Advanced Visualization Demo ===")
    
    # Initialize StudSar manager
    print("\n1. Initializing StudSar Manager...")
    manager = StudSarManager()
    
    # Create sample data with varied content
    sample_texts = [
        "Python is a powerful programming language used for data science and machine learning.",
        "Machine learning algorithms can process large datasets to find patterns and insights.",
        "Neural networks are inspired by biological neurons and can learn complex relationships.",
        "Data visualization helps communicate insights from complex datasets effectively.",
        "Artificial intelligence is transforming industries through automation and smart systems.",
        "Deep learning models require significant computational resources for training.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Big data analytics involves processing massive volumes of structured and unstructured data.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        "Cybersecurity protects digital systems from threats and unauthorized access.",
        "Software engineering practices ensure reliable and maintainable code development."
    ]
    
    print("\n2. Building network from sample texts...")
    # Build network with varied emotions
    emotions = ["informative", "technical", "neutral", "important", "innovative", 
               "complex", "analytical", "visual", "comprehensive", "scalable", 
               "security", "methodical"]
    
    for i, text in enumerate(sample_texts):
        emotion = emotions[i % len(emotions)]
        manager.update_network(text, emotion=emotion)
    
    print(f"Network built with {manager.studsar_network.get_total_markers()} markers.")
    
    # Simulate some usage patterns
    print("\n3. Simulating usage patterns...")
    queries = [
        "machine learning", "programming", "data analysis", "neural networks",
        "artificial intelligence", "computer vision", "big data", "cybersecurity"
    ]
    
    for query in queries:
        # Regular search (increments usage)
        ids, _, _ = manager.search(query, k=2)
        
        # Reputation-based search
        if hasattr(manager, 'search_with_reputation'):
            manager.search_with_reputation(query, k=1, reputation_weight=0.5)
    
    # Update some reputations
    print("\n4. Setting reputation scores...")
    all_ids = list(range(manager.studsar_network.get_total_markers()))
    
    # Set varied reputation scores
    for i, marker_id in enumerate(all_ids[:8]):
        if i < 3:
            manager.update_marker_reputation(marker_id, 1.0)  # High reputation
        elif i < 6:
            manager.update_marker_reputation(marker_id, 0.5)  # Medium reputation
        else:
            manager.update_marker_reputation(marker_id, -0.3)  # Low reputation
    
    # Initialize Dream Mode manager
    print("\n5. Initializing Dream Mode for analysis...")
    dream_manager = DreamModeManager(manager)
    
    # Create basic visualization
    print("\n6. Creating basic visualization...")
    try:
        manager.visualize_graph(output_file="studsar_basic_viz.png")
        print("✅ Basic visualization created: studsar_basic_viz.png")
    except Exception as e:
        print(f"❌ Basic visualization failed: {e}")
    
    # Create advanced visualization with Dream Mode insights
    print("\n7. Creating advanced visualization with Dream Mode insights...")
    try:
        manager.visualize_with_dream_insights(
            dream_mode_manager=dream_manager,
            output_file="studsar_advanced_viz.png"
        )
        print("✅ Advanced visualization created: studsar_advanced_viz.png")
    except Exception as e:
        print(f"❌ Advanced visualization failed: {e}")
    
    # Demonstrate Dream Mode consolidation and visualization
    print("\n8. Running Dream Mode consolidation...")
    try:
        # Configure Dream Mode for demonstration
        dream_manager.config.update({
            'min_network_size': 5,  # Lower threshold for demo
            'high_usage_threshold': 2,
            'low_usage_threshold': 0,
            'reputation_boost': 0.2,
            'max_removals_per_session': 3
        })
        
        # Run consolidation
        results = dream_manager.consolidate_memory()
        
        if results:
            print(f"Dream Mode results: {results}")
            
            # Create post-consolidation visualization
            print("\n9. Creating post-consolidation visualization...")
            manager.visualize_graph(
                output_file="studsar_post_consolidation.png",
                dream_mode_insights={
                    'promoted_markers': results.get('promoted_markers', []),
                    'removed_markers': results.get('removed_markers', []),
                    'total_markers': manager.studsar_network.get_total_markers(),
                    'efficiency_gain': results.get('efficiency_improvement', 0),
                    'memory_reduction': len(results.get('removed_markers', [])) / 
                                      (manager.studsar_network.get_total_markers() + len(results.get('removed_markers', [])))
                }
            )
            print("✅ Post-consolidation visualization created: studsar_post_consolidation.png")
        else:
            print("No consolidation performed (network may be too small or no changes needed).")
            
    except Exception as e:
        print(f"❌ Dream Mode consolidation failed: {e}")
    
    # Display final network statistics
    print("\n=== Final Network Statistics ===")
    print(f"Total markers: {manager.studsar_network.get_total_markers()}")
    
    if hasattr(manager.studsar_network, 'id_to_reputation'):
        reputations = list(manager.studsar_network.id_to_reputation.values())
        if reputations:
            print(f"Reputation range: {min(reputations):.3f} to {max(reputations):.3f}")
            print(f"Average reputation: {sum(reputations)/len(reputations):.3f}")
    
    if hasattr(manager.studsar_network, 'id_to_usage'):
        usages = list(manager.studsar_network.id_to_usage.values())
        if usages:
            print(f"Total usage count: {sum(usages)}")
            print(f"Most used marker: {max(usages)} accesses")
    
    print("\n=== Visualization Demo Complete ===")
    print("Check the generated PNG files for visual results!")

if __name__ == "__main__":
    main()