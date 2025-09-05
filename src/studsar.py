import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
"""
StudSar - Sistema di memoria semantica AI basato su rete neurale personalizzata.
Implementato con PyTorch e SentenceTransformers.
"""

import warnings
# Ignore specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning) # Example for common PyTorch/SentenceTransformers warnings

# --- Language Model Configuration ---
# Tentativo di importare spaCy, ma gestito come opzionale
try:
    import spacy
    SPACY_MODEL_NAME = "en_core_web_sm"
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
        print(f"SpaCy model '{SPACY_MODEL_NAME}' loaded.")
        SPACY_AVAILABLE = True
    except Exception as e:
        nlp = None
        SPACY_AVAILABLE = False
        print(f"SpaCy model '{SPACY_MODEL_NAME}' not loaded ({e}). Word segmentation will be used as fallback.")
except ImportError:
    spacy = None
    nlp = None
    SPACY_AVAILABLE = False
    print("SpaCy not installed. Word segmentation will be used as fallback.")
    print("To install spaCy (optional): pip install spacy && python -m spacy download en_core_web_sm")

# --- Segmentation Function ---
def segment_text(text, segment_length=100, use_spacy=True, spacy_sentences_per_segment=3):
    """Segments the text (words or sentences via spaCy)."""
    segments = []
    # Solo usa spaCy se è disponibile e l'utente lo vuole usare
    if use_spacy and SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences: return []
            for i in range(0, len(sentences), spacy_sentences_per_segment):
                segments.append(" ".join(sentences[i:i + spacy_sentences_per_segment]))
        except Exception as e:
            print(f"SpaCy error, fallback to words: {e}")
            use_spacy = False # Force fallback
    
    # Fallback a segmentazione basata su parole
    if not (use_spacy and SPACY_AVAILABLE and nlp): 
        words = text.split()
        if not words: return []
        for i in range(0, len(words), segment_length):
            segments.append(" ".join(words[i:i + segment_length]))

    # Filter empty segments
    segments = [seg for seg in segments if seg.strip()]
    print(f"Text segmented into {len(segments)} blocks.")
    return segments

# --- StudSar Neural Network ---
class StudSarNeural(nn.Module):
    """
    StudSar Neural Network implemented as PyTorch nn.Module.
    Stores associative markers (embeddings) and enables similarity search.
    """
    def __init__(self, embedding_dim, initial_capacity=1024, device=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"StudSarNeural will use device: {self.device}")

        # Main memory: a tensor for embeddings. Not a trainable layer by default.
        # We use a registered buffer so it's saved with state_dict but isn't a parameter.
        # We initialize it empty and expand it dynamically (less efficient, but more flexible).
        self.register_buffer('memory_embeddings', torch.empty((0, self.embedding_dim), device=self.device, dtype=torch.float32))

        # Mapping from internal ID (tensor index) to original segment
        self.id_to_segment = {}
        # Counter for next available ID
        self.next_id = 0

        print(f"StudSarNeural network initialized with embedding dimension {self.embedding_dim}.")
        print(f"Initial memory capacity: Dynamic (grows on-demand)")

    def add_marker(self, segment_text, embedding_vector):
        """Adds a marker (embedding) to the network's memory."""
        if not isinstance(embedding_vector, np.ndarray):
             print("Error: embedding_vector is not a numpy ndarray.")
             return None
        if embedding_vector.shape[0] != self.embedding_dim:
             print(f"Error: Embedding dimension ({embedding_vector.shape[0]}) does not match network dimension ({self.embedding_dim}).")
             return None

        # Convert to tensor and move to correct device
        embedding_tensor = torch.tensor(embedding_vector, dtype=torch.float32).to(self.device)

        # Add embedding to memory tensor
        # Dynamic expansion: create a new larger tensor and copy data
        # Warning: Potentially expensive operation for very frequent updates!
        new_memory = torch.cat((self.memory_embeddings, embedding_tensor.unsqueeze(0)), dim=0)
        # Deregister old buffer and register new one (necessary when changing tensor)
        if hasattr(self, 'memory_embeddings'):
             del self._buffers['memory_embeddings']
        self.register_buffer('memory_embeddings', new_memory)

        # Store mapping
        current_id = self.next_id
        self.id_to_segment[current_id] = segment_text

        self.next_id += 1
        return current_id

    def search_similar_markers(self, query_embedding, k=1):
        """
        Finds k most similar markers in network memory.
        Returns indices, similarities and corresponding segments.
        """
        if self.next_id == 0: # Empty memory
            return [], [], []

        if not isinstance(query_embedding, np.ndarray):
             print("Error: query_embedding is not a numpy ndarray.")
             return [], [], []

        # Convert query to tensor and move to device
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(self.device).unsqueeze(0) # Shape: (1, embedding_dim)

        try:
            similarities = F.cosine_similarity(query_tensor, self.memory_embeddings, dim=1)

        except Exception as e:
             print(f"Error during similarity calculation: {e}")
             return [], [], []

        # Find top k indices and values
        # Make sure k is not greater than available elements
        actual_k = min(k, self.next_id)
        if actual_k <= 0: return [], [], [] # Invalid k

        top_k_similarities, top_k_indices = torch.topk(similarities, k=actual_k)

        # Retrieve corresponding segments
        top_k_segments = [self.id_to_segment[idx.item()] for idx in top_k_indices]

        # Move results to CPU and convert to list/numpy if needed for output
        top_k_indices_list = top_k_indices.cpu().tolist()
        top_k_similarities_list = top_k_similarities.cpu().tolist()

        return top_k_indices_list, top_k_similarities_list, top_k_segments

    def get_marker_by_id(self, marker_id):
        """Retrieves embedding and segment given an ID."""
        if marker_id < 0 or marker_id >= self.next_id:
            return None, None
        embedding = self.memory_embeddings[marker_id].cpu().numpy()
        segment = self.id_to_segment.get(marker_id, None)
        return embedding, segment

    def get_total_markers(self):
        """Returns total number of stored markers."""
        return self.next_id


# --- StudSar Manager Class ---
class StudSarManager:
    """
    Manages interaction with StudSarNeural network and auxiliary operations
    (embedding generation, segmentation, persistence).
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', initial_capacity=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"StudSarManager will use device: {self.device}")
        # Load model to generate markers ("understanding" phase)
        self.embedding_generator = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.embedding_generator.get_sentence_embedding_dimension()

        # Initialize StudSar neural network
        self.studsar_network = StudSarNeural(self.embedding_dim, initial_capacity, device=self.device).to(self.device)

        print(f"\n--- StudSarManager Initialization ---")
        print(f"Embedding Generator Model: {model_name} (Dim: {self.embedding_dim})")
        print(f"StudSarNeural network ready on device: {self.studsar_network.device}")
        print(f"-----------------------------------\n")


    def generate_embedding(self, text):
        """Generates embedding for a text using loaded model."""
        if not text or not isinstance(text, str): return None
        embedding = self.embedding_generator.encode(text, convert_to_numpy=True)
        return embedding


    def build_network_from_text(self, text, segment_length=100, use_spacy_segmentation=True, spacy_sentences_per_segment=3):
        """Segments text and populates StudSarNeural network."""
        print("\n--- Building StudSar Network from Text ---")
        self.studsar_network = StudSarNeural(self.embedding_dim, device=self.device).to(self.device)
        print("StudSarNeural network reset.")


        segments = segment_text(text, segment_length, use_spacy_segmentation, spacy_sentences_per_segment)
        if not segments:
            print("No segments generated. Network construction interrupted.")
            return

        print(f"Generating and adding markers for {len(segments)} segments...")
        added_count = 0
        for i, seg in enumerate(segments):
            if not seg.strip(): continue # Skip empty segments after join
            embedding = self.generate_embedding(seg)
            if embedding is not None:
                marker_id = self.studsar_network.add_marker(seg, embedding)
                if marker_id is not None:
                     added_count += 1
            if (i + 1) % 50 == 0: print(f"  Processed {i+1}/{len(segments)} segments...")

        print(f"Added {added_count} markers to StudSar network.")
        print(f"Network memory now contains: {self.studsar_network.get_total_markers()} markers.")
        print("--- Network Construction Complete ---\n")


    def search(self, query_text, k=1):
        """Performs a search in StudSar network."""
        print(f"\n--- Query Search ---")
        print(f"Query: '{query_text}'")
        if not query_text or not isinstance(query_text, str):
             print("Invalid query.")
             return [], [], []

        query_embedding = self.generate_embedding(query_text)
        if query_embedding is None:
            print("Unable to generate embedding for query.")
            return [], [], []

        indices, similarities, segments = self.studsar_network.search_similar_markers(query_embedding, k=k)

        if not indices:
             print("No results found.")
        else:
            print(f"Found {len(indices)} results:")

        print("--- Search Complete ---\n")
        return indices, similarities, segments

    def update_network(self, new_text_segment):
        """Adds a new segment to existing StudSar network."""
        print("\n--- Updating StudSar Network ---")
        print(f"Adding new segment: '{new_text_segment[:100]}...'")
        if not new_text_segment or not isinstance(new_text_segment, str):
             print("Invalid segment for update.")
             return None

        embedding = self.generate_embedding(new_text_segment)
        if embedding is None:
            print("Unable to generate embedding for new segment.")
            return None

        marker_id = self.studsar_network.add_marker(new_text_segment, embedding)

        if marker_id is not None:
            print(f"New marker added with ID {marker_id}.")
            print(f"Network memory now contains: {self.studsar_network.get_total_markers()} markers.")
            print("--- Update Complete ---\n")
            return marker_id
        else:
             print("Update failed.")
             print("--- Update Failed ---\n")
             return None

    def save(self, filepath="studsar_neural_memory.pth"):
        """Saves StudSarNeural network state and mappings."""
        print(f"\n--- Saving StudSar State ---")
        state = {
            'network_state_dict': self.studsar_network.state_dict(),
            'id_to_segment': self.studsar_network.id_to_segment,
            'next_id': self.studsar_network.next_id,
            'embedding_dim': self.studsar_network.embedding_dim,
            'embedding_model_name': 'all-MiniLM-L6-v2' # Save actual model name instead of class name
        }
        try:
            torch.save(state, filepath)
            print(f"StudSar state saved to: {filepath}")
            print("--- Save Complete ---\n")
            return True
        except Exception as e:
            print(f"Error during save: {e}")
            print("--- Save Failed ---\n")
            return False

    @classmethod
    def load(cls, filepath="studsar_neural_memory.pth", model_name=None):
        """Loads state from file."""
        print(f"\n--- Loading StudSar State ---")
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            print("--- Load Failed ---\n")
            return None

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(filepath, map_location=device) # Load to correct device

            # Determine which embedding model to use
            saved_model_name = state.get('embedding_model_name', state.get('embedding_generator_name'))
            if model_name is None:
                model_name = saved_model_name if saved_model_name and saved_model_name != 'SentenceTransformer' else 'all-MiniLM-L6-v2' # Use saved or default
                print(f"Loading will use embedding model: '{model_name}' (detected or default)")
            else:
                 print(f"Loading will use embedding model: '{model_name}' (forced by user)")
                 if saved_model_name and model_name != saved_model_name and saved_model_name != 'SentenceTransformer':
                      print(f"Warning: Specified model '{model_name}' is different from saved model '{saved_model_name}'.")


            # Create new manager instance
            manager = cls(model_name=model_name)

            # Verify embedding dimension consistency
            saved_embedding_dim = state.get('embedding_dim')
            if saved_embedding_dim != manager.embedding_dim:
                print(f"CRITICAL ERROR: Saved embedding dimension ({saved_embedding_dim}) "
                      f"does not match loaded model dimension ({manager.embedding_dim}). Loading interrupted.")
                print("--- Load Failed ---\n")
                return None


            # Reconstruct network and load state
            manager.studsar_network = StudSarNeural(manager.embedding_dim, device=manager.device).to(manager.device)
            
            # Prepare memory_embeddings with correct size before loading state_dict
            if 'network_state_dict' in state and 'memory_embeddings' in state['network_state_dict']:
                memory_shape = state['network_state_dict']['memory_embeddings'].shape
                if memory_shape[0] > 0:  # If there are embeddings to load
                    # Pre-allocate memory_embeddings with correct shape
                    manager.studsar_network.register_buffer('memory_embeddings', 
                                                           torch.zeros(memory_shape, device=manager.device))
                    print(f"Pre-allocated memory_embeddings with shape {memory_shape}")
            
            # Now load the state dict
            manager.studsar_network.load_state_dict(state['network_state_dict'])
            manager.studsar_network.id_to_segment = state.get('id_to_segment', {})
            manager.studsar_network.next_id = state.get('next_id', 0)

            print(f"StudSar state loaded from: {filepath}")
            print(f"Number of markers loaded: {manager.studsar_network.get_total_markers()}")
            print("--- Load Complete ---\n")
            return manager

        except Exception as e:
            print(f"General error during loading: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debug
            print("--- Load Failed ---\n")
            return None


# --- Usage Example ---
if __name__ == "__main__":

    # 1. EXAMPLE TEXT
    example_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to the natural intelligence displayed by humans or animals. "
        "Leading AI textbooks define the field as the study of 'intelligent agents': "
        "any system that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. "
        # Add more text for better testing...
        "Some popular accounts use the term 'artificial intelligence' to describe machines that mimic 'cognitive' functions that humans associate with the human mind, "
        "such as 'learning' and 'problem-solving', however, this definition is rejected by major AI researchers. "
        "AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative tools, "
        "and competing at the highest level in strategic games. "
        "Machine learning is a core part of modern AI."
    )

    # 2. CREATE MANAGER AND BUILD NETWORK
    studsar_manager = StudSarManager()
    studsar_manager.build_network_from_text(
        example_text,
        use_spacy_segmentation=True,
        spacy_sentences_per_segment=2
    )

    # 3. EXECUTE A QUERY
    query = "What are AI applications?"
    ids, sims, segs = studsar_manager.search(query, k=2)
    if ids:
         print("Best results found for query:")
         for i in range(len(ids)):
              print(f"  {i+1}. ID: {ids[i]}, Sim: {sims[i]:.4f} - Seg: '{segs[i][:120]}...'")


    # 4. UPDATE THE NETWORK
    new_text = "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. It enables computers to learn from experience and understand the world in terms of a hierarchy of concepts."
    new_id = studsar_manager.update_network(new_text)

    # 5. RE-EXECUTE QUERY TO SEE IF NEW SEGMENT IS FOUND
    query_dl = "Tell me about deep learning"
    ids_dl, sims_dl, segs_dl = studsar_manager.search(query_dl, k=1)
    if ids_dl:
         print("Best result for 'deep learning':")
         print(f"  - ID: {ids_dl[0]}, Sim: {sims_dl[0]:.4f} - Seg: '{segs_dl[0][:120]}...'")
         # Check if ID matches the one just added
         if new_id is not None and ids_dl[0] == new_id:
              print("  (Correctly identified newly added segment!)")


    # 6. SAVE STATE
    save_path = "studsar_neural_demo.pth"
    studsar_manager.save(save_path)

    # 7. DELETE AND RELOAD
    del studsar_manager
    print("\nManager instance deleted. Attempting reload...")
    studsar_reloaded = StudSarManager.load(save_path)

    # 8. VERIFY LOADING AND RE-EXECUTE QUERY
    if studsar_reloaded:
        print("\nStudSar instance successfully reloaded!")
        query_post_load = "What is the definition of AI?"
        ids_post, sims_post, segs_post = studsar_reloaded.search(query_post_load, k=1)
        if ids_post:
             print("Best result (post-load):")
             print(f"  - ID: {ids_post[0]}, Sim: {sims_post[0]:.4f} - Seg: '{segs_post[0][:120]}...'")
    else:
        print("Error during loading.")

    print("\n--- Example Complete ---")

# Versione del pacchetto
__version__ = "1.0.0"
