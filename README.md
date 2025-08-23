# StudSar – AI Semantic Memory System (V2 Prototype)
**StudSar** is a prototype AI memory system based on a **custom neural network (`StudSarNeural`)**, implemented in PyTorch. The goal is to emulate a system that learns and associates information semantically, incorporating features inspired by cognitive processes. This V2 introduces several enhancements focusing on richer context and feedback mechanisms.
---
## System Overview (V2)
StudSar operates in the following steps:
1.  **Text Segmentation:** Breaks input text into logical units (segments).
    *   **V2 Goal:** Utilize a fine-tuned Transformer model for context-aware segmentation.
    *   **Current Implementation:** Uses spaCy for sentence-based segmentation (if available) or falls back to word-count based segmentation. A placeholder function (`segment_text_transformer_placeholder`) exists in `src/utils/text.py` for future integration.
2.  **Associative Marker Generation:** Creates semantic vector embeddings ("markers") for each segment using a pre-trained `Sentence Transformers` model.
3.  **Memory Storage:** Stores markers (embeddings) and their corresponding text segments within the `StudSarNeural` network (`torch.nn.Module`).
    *   **V2 Feature:** Allows associating optional **emotional tags** (e.g., "neutral", "informative", "important") with each marker upon creation or update.
4.  **Dynamic Association:** Semantic connections between markers are implicitly represented by their proximity in the high-dimensional embedding space. Associations are calculated dynamically during search via cosine similarity.
5.  **Internal Search:** Finds the `k` most similar markers/segments to a given query embedding using cosine similarity.
    *   **V2 Feature:** Search results trigger an increment in the **usage count** for the retrieved markers.
    *   **V2 Planned:** Reputation scores could potentially influence search ranking in future iterations.
6.  **Incremental Updates:** Supports continuous learning by allowing new text segments (with optional emotional tags) to be added to the network at any time.
7.  **Reputation Feedback (V2):** Allows associating a numerical **reputation score** with markers based on external feedback (positive or negative). This score is stored alongside the marker.
8.  **Usage Tracking (V2):** Monitors how often each marker is accessed during search operations. This data is stored and can be used for features like consolidation.
9.  **Offline Consolidation (V2 - "Dream Mode" - IMPLEMENTED):** Complete automatic optimization system that analyzes, promotes, and removes markers based on usage and reputation. Includes automatic scheduling and flexible configuration.
10. **Internal Graph Visualization (V2 - Placeholder):** Includes placeholder methods and necessary libraries (`networkx`, `matplotlib` in `requirements.txt`) to enable plotting the semantic graph of markers based on their embedding similarity. The plotting function itself (`visualize_graph` in `StudSarManager`) is a basic placeholder.
11. **Persistence:** The entire state of the `StudSarNeural` network, including embeddings, segment mappings, and V2 attributes (emotion tags, reputation scores, usage counts), can be saved to and reloaded from a file using `torch.save` and `torch.load`.

---
## Key Features (V2)

-   **Central Neural Network (`StudSarNeural`):** A custom `torch.nn.Module` acting as a dynamic, resizable vector memory.
-   **Tensor-Based Memory:** Embeddings (markers) are stored efficiently as PyTorch tensors within the network's buffer (`memory_embeddings`).
-   **Implicit Associations:** Semantic relationships are determined by vector similarity (cosine similarity) in the embedding space, not predefined links.
-   **High-Quality Embeddings:** Leverages `Sentence Transformers` models (e.g., `all-MiniLM-L6-v2`) for generating semantically rich text embeddings.
-   **Advanced Segmentation (V2 Goal):** Aims to integrate a fine-tuned Transformer for superior, context-aware text segmentation (currently uses spaCy/word-based fallback).
-   **Emotional Tagging (V2 Implemented):** Allows associating simple string tags representing emotional context with memories during creation/update.
-   **Reputation System (V2 Implemented):** Enables reinforcing or weakening memories by updating a numerical reputation score based on external feedback.
-   **Usage Tracking (V2 Implemented):** Monitors marker access frequency during search operations.
-   **Dream Mode (V2 IMPLEMENTED):** Complete automatic consolidation system that optimizes the network during idle periods, with promotion of high-usage markers and intelligent removal of low-value ones.
-   **Internal Graph Visualization (V2 Placeholder):** Basic setup and placeholder function for visualizing marker relationships.
-   **Internal Retrieval Engine:** Search performed efficiently using PyTorch tensor operations (cosine similarity, `torch.topk`).
-   **Persistence (V2 Implemented):** The network’s state, including V2 attributes (emotion, reputation, usage), can be saved and reloaded reliably.

---

## Project Structure (V2)

```
StudSar/
├── examples/
│   ├── basic_example.py         # Demonstrates core V2 functionalities
│   └── pycache /
├── src/
│   ├── managers/
│   │   ├── manager.py           # StudSarManager class (main interface)
│   │   ├── dream_mode.py        # DreamModeManager class (automatic optimization)
│   │   └── pycache /
│   ├── models/
│   │   ├── neural.py            # StudSarNeural class (core network)
│   │   ├── init .py
│   │   └── pycache /
│   ├── utils/
│   │   ├── text.py              # Text segmentation functions (incl. placeholder)
│   │   ├── visualization.py     # Placeholder for graph plotting
│   │   ├── init .py
│   │   └── pycache /
│   ├── init .py
│   └── studsar.py             # Older monolithic version (potentially deprecated/ref)
├── tests/
│   └── test_dream_mode.py       # Comprehensive Dream Mode tests
├── dream_mode_example.py        # Dream Mode demonstration script
├── .gitignore
├── README.md                    # This file
├── requirements.txt             # Project dependencies (incl. V2 additions)
└── RICERCA.MD                   # Research/Notes document (contains V2 planning) 
``` 


---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd StudSar
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` includes `torch` (CPU version by default), `sentence-transformers`, `numpy`, `scikit-learn`, `networkx`, and `matplotlib`.*
4.  **(Optional) Install spaCy for better segmentation:**
    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm
    ```
    *If spaCy is not installed or the model download fails, the system will automatically fall back to word-based segmentation.*

---

## Basic Usage

See `examples/basic_example.py` for a demonstration of:
- Initializing the `StudSarManager`.
- Building the network from text with a default emotion.
- Performing searches (which now track usage).
- Updating the network with new text and a specific emotion.
- Updating the reputation of a marker.
- Retrieving detailed information about a marker (segment, emotion, reputation, usage count).
- Saving the network state (including V2 attributes).
- Reloading the network state.
- Calling placeholder visualization function.

```python
# Example snippet from basic_example.py
from src.managers.manager import StudSarManager

# Initialize
manager = StudSarManager()

# Build with default emotion
manager.build_network_from_text("Some initial text.", default_emotion="neutral")

# Add new text with specific emotion
new_id = manager.update_network("More important text.", emotion="important")

# Search (increments usage count)
ids, sims, segs = manager.search("Query text", k=1)

# Update reputation
if ids:
    manager.update_marker_reputation(ids[0], 1.0) # Positive feedback

# Get details
if ids:
    details = manager.get_marker_details(ids[0])
    print(details) # {'segment': ..., 'emotion': ..., 'reputation': ..., 'usage_count': ...}

# Save and Load
manager.save("my_memory.pth")
reloaded_manager = StudSarManager.load("my_memory.pth")

## Dream Mode - Automatic Optimization System ✅

Dream Mode is an advanced automatic consolidation system that optimizes the StudSar neural network during low-activity periods, simulating memory consolidation processes similar to REM sleep.

### Key Features:

#### 🔍 **Intelligent Marker Analysis**
- Automatically categorizes markers based on usage and reputation
- Identifies high-value markers (frequently used)
- Detects low-value markers (candidates for removal)
- Verifies configurable thresholds to avoid operations on networks that are too small

#### ⬆️ **Automatic Promotion**
- Boosts the reputation of the most used markers
- Applies configurable boost factors
- Improves future search performance

#### 🗑️ **Intelligent Pruning**
- Selectively removes markers with low usage AND low reputation
- Respects configurable percentage limits (max 30% per cycle)
- Preserves network integrity by avoiding excessive removals

#### ⏰ **Automatic Scheduling**
- Scheduled execution in background
- Dedicated thread to not interfere with main operations
- Complete lifecycle control (start/stop)

#### ⚙️ **Flexible Configuration**
```python
config = {
    'high_usage_threshold': 10,        # Threshold for high-usage markers
    'low_usage_threshold': 2,          # Threshold for low-usage markers
    'reputation_threshold_low': 0.0,   # Low reputation threshold
    'min_markers_for_dream': 5,        # Minimum markers to activate Dream Mode
    'max_pruning_percentage': 0.3,     # Max % of markers removable per cycle
    'promotion_boost': 1.2,            # Reputation boost factor
    'schedule_interval_hours': 24      # Scheduling interval (hours)
}
```

### Complete Usage Example:

```python
from src.managers.dream_mode import DreamModeManager
from src.managers.manager import StudSarManager

# Initialize StudSar
manager = StudSarManager()
manager.build_network_from_text("Initial text for the network.")

# Configure Dream Mode
config = {
    'high_usage_threshold': 10,
    'low_usage_threshold': 2,
    'min_markers_for_dream': 5,
    'max_pruning_percentage': 0.3,
    'promotion_boost': 1.2
}

# Initialize Dream Mode Manager
dream_manager = DreamModeManager(manager.studsar_network, config)

# Manual execution
result = dream_manager.run_dream_mode()
print(f"Promoted markers: {result['promoted_count']}")
print(f"Removed markers: {result['pruned_count']}")

# Automatic scheduling
dream_manager.start_automatic_scheduling()
print("Dream Mode active in background")

# Check status
status = dream_manager.get_status()
print(f"Scheduler active: {status['scheduler_active']}")
print(f"Last execution: {status['last_run']}")

# Stop scheduling
dream_manager.stop_automatic_scheduling()
```

### Monitoring and Diagnostics:

```python
# Detailed marker analysis
stats = dream_manager.analyze_marker_statistics()
print(f"High-usage markers: {len(stats['high_usage_markers'])}")
print(f"Low-usage markers: {len(stats['low_usage_markers'])}")
print(f"Low-reputation markers: {len(stats['low_reputation_markers'])}")

# System status
status = dream_manager.get_status()
print(f"Dream Mode running: {status['is_running']}")
print(f"Total markers: {status['total_markers']}")
```

### Testing and Validation:
- ✅ **13 comprehensive tests** in `tests/test_dream_mode.py`
- ✅ **Complete coverage** of all functionalities
- ✅ **Integration tests** with save/load
- ✅ **Configuration validation** and error handling
- ✅ **Demonstration script** in `dream_mode_example.py`

### Dream Mode Benefits:
1. **Automatic optimization**: Improves performance without manual intervention
2. **Memory management**: Prevents uncontrolled network growth
3. **Result quality**: Promotes valuable content and removes noise
4. **Efficiency**: Background operations with no impact on performance
5. **Flexibility**: Completely customizable configuration

## Future Work / V3 Goals
- **Transformer Segmentation**: Replace the placeholder with a functional transformer model for segmentation.
- **Advanced Visualization**: Complete the visualize_graph function for meaningful graph output with Dream Mode insights.
- **Reputation-Based Search**: Incorporate reputation scores into the search ranking algorithm.
- **Advanced Dream Mode**: 
  - Implement marker consolidation (merge similar high-value markers)
  - Add adaptive thresholds based on network size and usage patterns
  - Implement marker relationship analysis for smarter pruning
- **Performance Optimization**: GPU acceleration for large-scale networks
- **Distributed Memory**: Support for distributed StudSar networks
