"""
Modulo di utilità per la segmentazione del testo in StudSar.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

# --- V3: Advanced Transformer Segmentation ---
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("sentence-transformers not available for advanced segmentation. Using fallback methods.")

def segment_text_transformer(text, min_segment_length=50, max_segment_length=300, 
                           similarity_threshold=0.7, model_name='all-MiniLM-L6-v2'):
    """Advanced transformer-based semantic segmentation.
    
    Args:
        text: Input text to segment
        min_segment_length: Minimum characters per segment
        max_segment_length: Maximum characters per segment  
        similarity_threshold: Cosine similarity threshold for segment boundaries
        model_name: Sentence transformer model to use
    
    Returns:
        List of semantically coherent text segments
    """
    if not TRANSFORMER_AVAILABLE:
        print("\n--- Transformer Segmentation Unavailable - Using Fallback ---")
        return segment_text(text, use_spacy=True)
    
    print("\n--- Using Advanced Transformer Segmentation ---")
    
    try:
        # Initialize sentence transformer model
        model = SentenceTransformer(model_name)
        
        # First, split into sentences using spaCy if available, otherwise basic split
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Basic sentence splitting as fallback
            sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        
        # Generate embeddings for all sentences
        embeddings = model.encode(sentences)
        
        # Calculate semantic similarity between consecutive sentences
        segments = []
        current_segment = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and segment context
            current_embedding = embeddings[i:i+1]
            segment_embedding = np.mean(embeddings[max(0, i-3):i], axis=0, keepdims=True)
            
            similarity = cosine_similarity(current_embedding, segment_embedding)[0][0]
            sentence_length = len(sentences[i])
            
            # Decide whether to continue current segment or start new one
            should_break = (
                similarity < similarity_threshold or  # Low semantic similarity
                current_length + sentence_length > max_segment_length or  # Too long
                (current_length > min_segment_length and similarity < similarity_threshold + 0.1)  # Good length + low similarity
            )
            
            if should_break and current_length >= min_segment_length:
                # Finalize current segment
                segments.append(' '.join(current_segment))
                current_segment = [sentences[i]]
                current_length = sentence_length
            else:
                # Continue current segment
                current_segment.append(sentences[i])
                current_length += sentence_length + 1  # +1 for space
        
        # Add final segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        # Post-process: merge very short segments with neighbors
        final_segments = []
        for segment in segments:
            if len(segment) < min_segment_length and final_segments:
                # Merge with previous segment
                final_segments[-1] += ' ' + segment
            else:
                final_segments.append(segment)
        
        print(f"Transformer segmentation created {len(final_segments)} semantic segments.")
        return final_segments
        
    except Exception as e:
        print(f"Transformer segmentation failed ({e}). Using fallback.")
        return segment_text(text, use_spacy=True)

# Backward compatibility alias
segment_text_transformer_placeholder = segment_text_transformer
# --- END V3 ---

def segment_text(text, segment_length=100, use_spacy=True, spacy_sentences_per_segment=3):
    """Segments the text (words or sentences via spaCy)."""
    segments = []
    #  spaCy for the user
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
    # Fallback to word-based segmentation
    if not (use_spacy and SPACY_AVAILABLE and nlp): 
        words = text.split()
        if not words: return []
        for i in range(0, len(words), segment_length):
            segments.append(" ".join(words[i:i + segment_length]))
    # Filter empty segments
    segments = [seg for seg in segments if seg.strip()]
    print(f"Text segmented into {len(segments)} blocks.")
    return segments
