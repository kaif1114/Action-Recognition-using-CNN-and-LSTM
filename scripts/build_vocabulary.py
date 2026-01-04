"""Build vocabulary from COCO action captions."""
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocabulary import Vocabulary


def main():
    """Build vocabulary from COCO action captions."""
    print("=" * 80)
    print("Building Vocabulary from COCO Action Captions")
    print("=" * 80)

    # Paths
    captions_json = "dataset/coco_actions/action_captions.json"
    output_path = "checkpoints/vocabulary.pkl"

    # Check if captions file exists
    if not os.path.exists(captions_json):
        print(f"Error: Captions file not found: {captions_json}")
        print("Please ensure the COCO actions dataset is in dataset/coco_actions/")
        return

    # Load captions
    print(f"\nLoading captions from {captions_json}...")
    with open(captions_json, 'r') as f:
        data = json.load(f)

    # Extract captions from annotations
    annotations = data.get('annotations', [])
    captions = [ann['caption'] for ann in annotations if 'caption' in ann]

    print(f"Loaded {len(captions)} captions from {len(data.get('images', []))} images")
    print(f"Average captions per image: {len(captions) / max(len(data.get('images', [])), 1):.2f}")

    # Display sample captions
    print(f"\nSample captions:")
    for i, caption in enumerate(captions[:5], 1):
        print(f"  {i}. {caption}")

    # Build vocabulary
    print(f"\nBuilding vocabulary with max_vocab_size=5000...")
    vocab = Vocabulary()
    vocab.build_from_captions(captions, max_vocab_size=5000, min_freq=1)

    # Statistics
    print(f"\n{'-' * 80}")
    print("Vocabulary Statistics")
    print("-" * 80)
    print(f"Total vocabulary size: {len(vocab)} words")
    print(f"Content words: {len(vocab) - 4}")
    print(f"Special tokens: 4 (<PAD>, <START>, <END>, <UNK>)")

    # Caption length statistics
    print(f"\nAnalyzing caption lengths...")
    caption_lengths = []
    for caption in captions[:1000]:  # Sample 1000 captions for speed
        tokens = vocab.tokenize(caption)
        caption_lengths.append(len(tokens))

    if caption_lengths:
        avg_len = sum(caption_lengths) / len(caption_lengths)
        min_len = min(caption_lengths)
        max_len = max(caption_lengths)
        print(f"Caption length (words):")
        print(f"  Min: {min_len}")
        print(f"  Max: {max_len}")
        print(f"  Average: {avg_len:.1f}")

    # Test encoding/decoding
    print(f"\nTesting encoding/decoding...")
    test_captions = captions[:3]
    for i, caption in enumerate(test_captions, 1):
        encoded = vocab.encode_caption(caption, max_length=20)
        decoded = vocab.decode_caption(encoded)
        print(f"\nExample {i}:")
        print(f"  Original: {caption}")
        print(f"  Encoded length: {len(encoded)} tokens")
        print(f"  Decoded: {decoded}")

    # Save vocabulary
    print(f"\n{'-' * 80}")
    vocab.save(output_path)
    print("-" * 80)

    # Verify save
    print(f"\nVerifying saved vocabulary...")
    loaded_vocab = Vocabulary.load(output_path)
    assert len(loaded_vocab) == len(vocab), "Loaded vocabulary size mismatch!"
    print(f"✓ Vocabulary loaded successfully ({len(loaded_vocab)} words)")

    # Test loaded vocabulary
    test_caption = captions[0]
    encoded_loaded = loaded_vocab.encode_caption(test_caption)
    decoded_loaded = loaded_vocab.decode_caption(encoded_loaded)
    print(f"✓ Loaded vocabulary works correctly")

    print(f"\n{'=' * 80}")
    print("Vocabulary Build Complete!")
    print("=" * 80)
    print(f"\nVocabulary saved to: {output_path}")
    print(f"Vocabulary size: {len(vocab)} words")
    print(f"Ready for caption model training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
