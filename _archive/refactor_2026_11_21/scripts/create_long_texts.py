#!/usr/bin/env python3
"""
Create Long-Text Training Data from Short Sentences.

Strategy:
1. Group consecutive sentences into long documents (500-800 words)
2. Create sliding window sequences for training
3. Save as new training file
"""

import random
from pathlib import Path

def create_long_texts(input_file, output_file, target_words=650, window_size=100):
    """
    Create long texts by combining short sentences.
    
    Args:
        input_file: Path to short sentences file
        output_file: Path to output long texts file
        target_words: Target word count per document (500-800 range)
        window_size: Sliding window stride in words
    """
    
    # Read all sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    random.seed(42)
    random.shuffle(sentences)  # Shuffle to avoid domain bias
    
    # Tokenize sentences into words
    all_words = []
    sentence_boundaries = []
    current_pos = 0
    
    for sent in sentences:
        words = sent.split()
        all_words.extend(words)
        sentence_boundaries.append(current_pos + len(words))
        current_pos += len(words)
    
    print(f"Total words: {len(all_words):,}")
    print(f"Total sentences: {len(sentences):,}")
    
    # Create long documents (500-800 words each)
    min_words = 500
    max_words = 800
    
    long_texts = []
    word_counts = []
    
    # Sliding window to create overlapping documents
    for start in range(0, len(all_words) - target_words, window_size):
        # Random length between min and max
        length = random.randint(min_words, max_words)
        
        if start + length > len(all_words):
            break
        
        doc_words = all_words[start:start + length]
        doc_text = ' '.join(doc_words)
        
        long_texts.append(doc_text)
        word_counts.append(len(doc_words))
    
    print(f"\nCreated {len(long_texts):,} long documents")
    print(f"Word count range: {min(word_counts):,} - {max(word_counts):,}")
    print(f"Average: {sum(word_counts) / len(word_counts):.0f} words")
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in long_texts:
            f.write(doc + '\n')
    
    print(f"\nSaved to: {output_file}")
    
    # Statistics
    return {
        'num_documents': len(long_texts),
        'min_words': min(word_counts),
        'max_words': max(word_counts),
        'avg_words': sum(word_counts) / len(word_counts)
    }


if __name__ == "__main__":
    INPUT_FILE = "data/turkish_sentences.txt"
    OUTPUT_FILE = "data/turkish_long_texts.txt"
    
    stats = create_long_texts(
        INPUT_FILE, 
        OUTPUT_FILE, 
        target_words=650,
        window_size=100
    )
    
    print("\n=== SUMMARY ===")
    print(f"Documents: {stats['num_documents']:,}")
    print(f"Word range: {stats['min_words']:,} - {stats['max_words']:,}")
    print(f"Average: {stats['avg_words']:.0f} words")
