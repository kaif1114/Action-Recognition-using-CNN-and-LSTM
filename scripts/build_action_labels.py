"""Script to build action labels from Stanford 40 Actions dataset."""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.action_labels import ActionLabels


def main():
    """Build action labels from Stanford 40 Actions dataset."""
    print("="*60)
    print("Building Action Labels from Stanford 40 Actions Dataset")
    print("="*60)

    # Paths
    actions_file = "dataset/stanford_40_actions/ImageSplits/actions.txt"
    output_path = "checkpoints/action_labels.pkl"

    # Check if actions file exists
    if not os.path.exists(actions_file):
        print(f"Error: Actions file not found: {actions_file}")
        print("Please make sure the Stanford 40 dataset is in the 'dataset' directory.")
        return

    # Build action labels
    print(f"\nBuilding labels from: {actions_file}")
    action_labels = ActionLabels()
    action_labels.build_from_actions_file(actions_file)

    # Save action labels
    print(f"\nSaving labels to: {output_path}")
    action_labels.save(output_path)

    # Print statistics
    print(f"\n{'='*60}")
    print("Action Labels Statistics:")
    print(f"{'='*60}")
    print(f"Total action classes: {len(action_labels)}")
    print(f"\nAll {action_labels.num_classes} actions:")
    all_actions = action_labels.get_all_actions()
    for i in range(0, len(all_actions), 5):
        # Print 5 actions per line
        actions_line = all_actions[i:i+5]
        print(f"  {', '.join(actions_line)}")

    # Test encoding/decoding
    print(f"\n{'='*60}")
    print("Testing Encoding/Decoding:")
    print(f"{'='*60}")
    test_actions = ["applauding", "cooking", "riding_a_bike", "reading", "jumping"]

    for action in test_actions:
        if action in action_labels.word2idx:
            idx = action_labels.encode_action(action)
            decoded = action_labels.decode_action(idx)
            print(f"  {action} -> {idx} -> {decoded}")
        else:
            print(f"  {action} -> NOT FOUND")

    print(f"\n{'='*60}")
    print("Action labels built successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
