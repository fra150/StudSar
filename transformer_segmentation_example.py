#!/usr/bin/env python3
"""
Esempio di utilizzo della segmentazione transformer avanzata in StudSar.
Questo script dimostra le capacità di segmentazione semantica basata su transformer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.managers.manager import StudSarManager
from src.utils.text import segment_text_transformer, segment_text, TRANSFORMER_AVAILABLE

def main():
    print("=== StudSar Transformer Segmentation Demo ===")
    print(f"Transformer segmentation available: {TRANSFORMER_AVAILABLE}")
    print()
    
    # Testo di esempio con diversi argomenti semantici
    sample_text = """
    Artificial intelligence has revolutionized many industries in recent years. Machine learning algorithms 
    can now process vast amounts of data and identify patterns that humans might miss. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers to solve complex problems.
    
    In the field of natural language processing, transformer models have become the gold standard. 
    These models, such as BERT and GPT, can understand context and generate human-like text. 
    They have applications in translation, summarization, and question answering systems.
    
    Computer vision is another area where AI has made significant strides. Convolutional neural networks 
    can classify images, detect objects, and even generate realistic synthetic images. This technology 
    is used in autonomous vehicles, medical imaging, and security systems.
    
    The future of AI looks promising but also raises important ethical questions. Issues of bias, 
    privacy, and job displacement need to be carefully considered. Responsible AI development 
    requires collaboration between technologists, policymakers, and society at large.
    """
    
    print("Testo di esempio:")
    print(sample_text[:200] + "...")
    print()
    
    # Confronto tra diversi metodi di segmentazione
    print("=== Confronto Metodi di Segmentazione ===")
    
    # 1. Segmentazione basata su parole (fallback)
    print("\n1. Segmentazione basata su parole:")
    word_segments = segment_text(sample_text, segment_length=50, use_spacy=False)
    print(f"Numero di segmenti: {len(word_segments)}")
    for i, seg in enumerate(word_segments[:3]):
        print(f"  Segmento {i+1}: {seg[:80]}...")
    
    # 2. Segmentazione spaCy (se disponibile)
    print("\n2. Segmentazione spaCy:")
    spacy_segments = segment_text(sample_text, use_spacy=True, spacy_sentences_per_segment=2)
    print(f"Numero di segmenti: {len(spacy_segments)}")
    for i, seg in enumerate(spacy_segments[:3]):
        print(f"  Segmento {i+1}: {seg[:80]}...")
    
    # 3. Segmentazione transformer (se disponibile)
    if TRANSFORMER_AVAILABLE:
        print("\n3. Segmentazione Transformer Semantica:")
        transformer_segments = segment_text_transformer(
            sample_text, 
            min_segment_length=100,
            max_segment_length=400,
            similarity_threshold=0.6
        )
        print(f"Numero di segmenti: {len(transformer_segments)}")
        for i, seg in enumerate(transformer_segments):
            print(f"  Segmento {i+1} ({len(seg)} caratteri): {seg[:100]}...")
    else:
        print("\n3. Segmentazione Transformer: Non disponibile")
    
    print("\n=== Test con StudSarManager ===")
    
    # Inizializza StudSarManager
    try:
        manager = StudSarManager()
        print("StudSarManager inizializzato con successo.")
        
        # Test con segmentazione transformer
        if TRANSFORMER_AVAILABLE:
            print("\nCostruendo rete con segmentazione transformer...")
            manager.build_network_from_text(
                sample_text,
                use_transformer_segmentation=True,
                transformer_params={
                    'min_segment_length': 80,
                    'max_segment_length': 300,
                    'similarity_threshold': 0.65
                },
                default_emotion='neutral'
            )
            
            # Test di ricerca
            print("\nTest di ricerca:")
            queries = [
                "What is machine learning?",
                "How do transformer models work?",
                "What are the ethical concerns of AI?"
            ]
            
            for query in queries:
                print(f"\nQuery: '{query}'")
                ids, similarities, segments = manager.search(query, k=2)
                for i, (id_, sim, seg) in enumerate(zip(ids, similarities, segments)):
                    print(f"  Risultato {i+1} (ID: {id_}, Sim: {sim:.3f}): {seg[:100]}...")
        
        else:
            print("\nSegmentazione transformer non disponibile. Usando metodo standard.")
            manager.build_network_from_text(sample_text, default_emotion='neutral')
            
        print(f"\nRete costruita con {manager.studsar_network.get_total_markers()} marker.")
        
    except Exception as e:
        print(f"Errore durante l'inizializzazione: {e}")
        print("Questo potrebbe essere dovuto a problemi con le dipendenze.")
    
    print("\n=== Demo Completata ===")

if __name__ == "__main__":
    main()