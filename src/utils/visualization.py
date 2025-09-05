#!/usr/bin/env python3
"""
Visualization utilities for StudSar semantic networks.
Provides advanced graph visualization with Dream Mode insights.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Some visualization features will be limited.")

def plot_semantic_graph(id_to_embedding_map, id_to_segment, similarity_threshold=0.85, 
                       output_file="studsar_graph.png", id_to_emotion=None, 
                       id_to_reputation=None, id_to_usage=None, dream_mode_insights=None):
    """
    Creates an advanced visualization of the StudSar semantic network.
    
    Args:
        id_to_embedding_map: Dict mapping marker IDs to embeddings
        id_to_segment: Dict mapping marker IDs to text segments
        similarity_threshold: Minimum similarity to draw edges
        output_file: Output filename for the plot
        id_to_emotion: Dict mapping marker IDs to emotions (V2)
        id_to_reputation: Dict mapping marker IDs to reputation scores (V2)
        id_to_usage: Dict mapping marker IDs to usage counts (V2)
        dream_mode_insights: Dict with Dream Mode analysis results
    """
    
    if not id_to_embedding_map:
        print("No markers to visualize.")
        return
    
    print(f"Visualizing {len(id_to_embedding_map)} markers...")
    
    # Extract data
    marker_ids = list(id_to_embedding_map.keys())
    embeddings = np.array([id_to_embedding_map[mid] for mid in marker_ids])
    
    # Reduce dimensionality for 2D visualization
    if embeddings.shape[1] > 2:
        if len(marker_ids) > 50:
            # Use t-SNE for larger datasets
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(marker_ids)-1))
        else:
            # Use PCA for smaller datasets
            reducer = PCA(n_components=2, random_state=42)
        
        positions_2d = reducer.fit_transform(embeddings)
    else:
        positions_2d = embeddings
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main network graph
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
    
    # Statistics panels
    ax_stats = plt.subplot2grid((3, 4), (0, 3))
    ax_reputation = plt.subplot2grid((3, 4), (1, 3))
    ax_usage = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    ax_dream = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    
    # Plot main network
    _plot_main_network(ax_main, marker_ids, positions_2d, id_to_segment, 
                      id_to_emotion, id_to_reputation, id_to_usage, 
                      embeddings, similarity_threshold)
    
    # Plot statistics
    _plot_statistics(ax_stats, marker_ids, id_to_emotion, id_to_reputation, id_to_usage)
    
    # Plot reputation distribution
    _plot_reputation_distribution(ax_reputation, id_to_reputation)
    
    # Plot usage patterns
    _plot_usage_patterns(ax_usage, id_to_usage)
    
    # Plot Dream Mode insights
    _plot_dream_mode_insights(ax_dream, dream_mode_insights)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to: {output_file}")
    plt.show()

def _plot_main_network(ax, marker_ids, positions_2d, id_to_segment, 
                      id_to_emotion, id_to_reputation, id_to_usage, 
                      embeddings, similarity_threshold):
    """Plot the main semantic network graph."""
    
    # Prepare node colors based on reputation
    if id_to_reputation:
        reputations = [id_to_reputation.get(mid, 0.0) for mid in marker_ids]
        # Normalize reputation for color mapping
        rep_min, rep_max = min(reputations), max(reputations)
        if rep_max > rep_min:
            norm_reputations = [(r - rep_min) / (rep_max - rep_min) for r in reputations]
        else:
            norm_reputations = [0.5] * len(reputations)
    else:
        norm_reputations = [0.5] * len(marker_ids)
    
    # Prepare node sizes based on usage
    if id_to_usage:
        usages = [id_to_usage.get(mid, 0) for mid in marker_ids]
        max_usage = max(usages) if usages else 1
        node_sizes = [50 + (usage / max_usage) * 200 for usage in usages]
    else:
        node_sizes = [100] * len(marker_ids)
    
    # Create colormap for reputation
    colors = plt.cm.RdYlGn(norm_reputations)
    
    # Plot nodes
    scatter = ax.scatter(positions_2d[:, 0], positions_2d[:, 1], 
                        c=colors, s=node_sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add edges based on similarity
    if NETWORKX_AVAILABLE and len(marker_ids) < 100:  # Limit edges for performance
        _add_similarity_edges(ax, marker_ids, positions_2d, embeddings, similarity_threshold)
    
    # Add labels for high-reputation or high-usage nodes
    _add_selective_labels(ax, marker_ids, positions_2d, id_to_segment, 
                         id_to_reputation, id_to_usage)
    
    ax.set_title('StudSar Semantic Network\n(Size = Usage, Color = Reputation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Semantic Dimension 1')
    ax.set_ylabel('Semantic Dimension 2')
    
    # Add colorbar for reputation
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Reputation Score', rotation=270, labelpad=15)

def _add_similarity_edges(ax, marker_ids, positions_2d, embeddings, threshold):
    """Add edges between similar markers."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Add edges
    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            if sim_matrix[i, j] >= threshold:
                x_coords = [positions_2d[i, 0], positions_2d[j, 0]]
                y_coords = [positions_2d[i, 1], positions_2d[j, 1]]
                ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)

def _add_selective_labels(ax, marker_ids, positions_2d, id_to_segment, 
                         id_to_reputation, id_to_usage):
    """Add labels only for important nodes."""
    
    # Determine which nodes to label
    important_nodes = set()
    
    if id_to_reputation:
        # Top reputation nodes
        rep_items = [(mid, id_to_reputation.get(mid, 0.0)) for mid in marker_ids]
        rep_items.sort(key=lambda x: x[1], reverse=True)
        important_nodes.update([mid for mid, _ in rep_items[:5]])
    
    if id_to_usage:
        # Top usage nodes
        usage_items = [(mid, id_to_usage.get(mid, 0)) for mid in marker_ids]
        usage_items.sort(key=lambda x: x[1], reverse=True)
        important_nodes.update([mid for mid, _ in usage_items[:5]])
    
    # Add labels
    for i, mid in enumerate(marker_ids):
        if mid in important_nodes:
            segment = id_to_segment.get(mid, f"Marker {mid}")
            label = segment[:30] + "..." if len(segment) > 30 else segment
            ax.annotate(label, (positions_2d[i, 0], positions_2d[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

def _plot_statistics(ax, marker_ids, id_to_emotion, id_to_reputation, id_to_usage):
    """Plot general network statistics."""
    
    stats_text = f"Network Statistics\n\n"
    stats_text += f"Total Markers: {len(marker_ids)}\n\n"
    
    if id_to_emotion:
        emotions = list(id_to_emotion.values())
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        stats_text += "Emotions:\n"
        for emotion, count in sorted(emotion_counts.items()):
            stats_text += f"  {emotion}: {count}\n"
        stats_text += "\n"
    
    if id_to_reputation:
        reputations = list(id_to_reputation.values())
        stats_text += f"Reputation:\n"
        stats_text += f"  Avg: {np.mean(reputations):.3f}\n"
        stats_text += f"  Min: {np.min(reputations):.3f}\n"
        stats_text += f"  Max: {np.max(reputations):.3f}\n\n"
    
    if id_to_usage:
        usages = list(id_to_usage.values())
        stats_text += f"Usage:\n"
        stats_text += f"  Total: {sum(usages)}\n"
        stats_text += f"  Avg: {np.mean(usages):.1f}\n"
        stats_text += f"  Max: {max(usages)}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def _plot_reputation_distribution(ax, id_to_reputation):
    """Plot reputation score distribution."""
    
    if not id_to_reputation:
        ax.text(0.5, 0.5, 'No reputation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reputation Distribution')
        ax.axis('off')
        return
    
    reputations = list(id_to_reputation.values())
    ax.hist(reputations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(reputations), color='red', linestyle='--', label=f'Mean: {np.mean(reputations):.3f}')
    ax.set_title('Reputation Distribution')
    ax.set_xlabel('Reputation Score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_usage_patterns(ax, id_to_usage):
    """Plot usage patterns over markers."""
    
    if not id_to_usage:
        ax.text(0.5, 0.5, 'No usage data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Usage Patterns')
        ax.axis('off')
        return
    
    marker_ids = sorted(id_to_usage.keys())
    usages = [id_to_usage[mid] for mid in marker_ids]
    
    ax.bar(range(len(marker_ids)), usages, alpha=0.7, color='lightgreen')
    ax.set_title('Usage Count by Marker')
    ax.set_xlabel('Marker ID')
    ax.set_ylabel('Usage Count')
    
    # Show only some x-axis labels to avoid clutter
    if len(marker_ids) > 20:
        step = len(marker_ids) // 10
        ax.set_xticks(range(0, len(marker_ids), step))
        ax.set_xticklabels([str(marker_ids[i]) for i in range(0, len(marker_ids), step)])
    
    ax.grid(True, alpha=0.3)

def _plot_dream_mode_insights(ax, dream_mode_insights):
    """Plot Dream Mode analysis insights."""
    
    if not dream_mode_insights:
        ax.text(0.5, 0.5, 'No Dream Mode data\navailable', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Dream Mode Insights')
        ax.axis('off')
        return
    
    # Extract insights
    high_value = dream_mode_insights.get('high_value_markers', [])
    low_value = dream_mode_insights.get('low_value_markers', [])
    promoted = dream_mode_insights.get('promoted_markers', [])
    removed = dream_mode_insights.get('removed_markers', [])
    
    # Create summary text
    insights_text = "Dream Mode Analysis\n\n"
    insights_text += f"High-value markers: {len(high_value)}\n"
    insights_text += f"Low-value markers: {len(low_value)}\n"
    insights_text += f"Promoted: {len(promoted)}\n"
    insights_text += f"Removed: {len(removed)}\n\n"
    
    # Add efficiency metrics if available
    if 'efficiency_gain' in dream_mode_insights:
        insights_text += f"Efficiency gain: {dream_mode_insights['efficiency_gain']:.1%}\n"
    
    if 'memory_reduction' in dream_mode_insights:
        insights_text += f"Memory reduction: {dream_mode_insights['memory_reduction']:.1%}\n"
    
    # Create a simple pie chart for marker categories
    if high_value or low_value:
        categories = ['High-value', 'Low-value', 'Other']
        total_markers = dream_mode_insights.get('total_markers', len(high_value) + len(low_value))
        other_count = max(0, total_markers - len(high_value) - len(low_value))
        sizes = [len(high_value), len(low_value), other_count]
        colors = ['lightgreen', 'lightcoral', 'lightgray']
        
        # Only show non-zero categories
        non_zero_idx = [i for i, size in enumerate(sizes) if size > 0]
        if non_zero_idx:
            filtered_sizes = [sizes[i] for i in non_zero_idx]
            filtered_colors = [colors[i] for i in non_zero_idx]
            filtered_labels = [categories[i] for i in non_zero_idx]
            
            ax.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                  autopct='%1.1f%%', startangle=90)
    
    ax.set_title('Dream Mode Insights')

def create_advanced_visualization(studsar_manager, output_file="studsar_advanced.png", 
                                dream_mode_insights=None):
    """
    Create a comprehensive visualization of the StudSar network with all available data.
    
    Args:
        studsar_manager: StudSarManager instance
        output_file: Output filename
        dream_mode_insights: Optional Dream Mode analysis results
    """
    
    if not studsar_manager.studsar_network:
        print("Error: StudSar network not initialized.")
        return
    
    # Gather all data
    id_to_embedding_map = studsar_manager.studsar_network.get_all_embeddings_and_ids()
    id_to_segment = studsar_manager.studsar_network.id_to_segment
    id_to_emotion = getattr(studsar_manager.studsar_network, 'id_to_emotion', None)
    id_to_reputation = getattr(studsar_manager.studsar_network, 'id_to_reputation', None)
    id_to_usage = getattr(studsar_manager.studsar_network, 'id_to_usage', None)
    
    # Create visualization
    plot_semantic_graph(
        id_to_embedding_map=id_to_embedding_map,
        id_to_segment=id_to_segment,
        similarity_threshold=0.85,
        output_file=output_file,
        id_to_emotion=id_to_emotion,
        id_to_reputation=id_to_reputation,
        id_to_usage=id_to_usage,
        dream_mode_insights=dream_mode_insights
    )

if __name__ == "__main__":
    print("StudSar Visualization Utilities")
    print("Import this module and use create_advanced_visualization() or plot_semantic_graph()")