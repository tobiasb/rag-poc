#!/usr/bin/env python3
"""
Interactive tool to compare semantic similarity between two words or phrases.

This script uses the same embedding model to show exactly how similar
any two pieces of text are, with detailed similarity metrics.
"""
import numpy as np
from fastembed import TextEmbedding
import textwrap


class SimilarityComparer:
    def __init__(self):
        """Initialize the embedding model."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"🤖 Loading embedding model: {self.model_name}")
        self.embedder = TextEmbedding(model_name=self.model_name)
        print("✅ Model loaded successfully!")

    def get_embeddings(self, texts):
        """Get embeddings for a list of texts."""
        # Convert generator to list
        embeddings = list(self.embedder.embed(texts))
        return np.array(embeddings)

    def compare_phrases(self, phrase1, phrase2):
        """Compare two phrases and return similarity metrics."""
        # Get embeddings
        embeddings = self.get_embeddings([phrase1, phrase2])

        # Calculate cosine similarity manually
        # cosine_similarity = (A·B) / (||A|| * ||B||)
        vec1, vec2 = embeddings[0], embeddings[1]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = dot_product / (norm1 * norm2)

        # Calculate euclidean distance
        euclidean_dist = np.linalg.norm(vec1 - vec2)

        # Calculate angular distance (in degrees)
        # Clamp cosine_sim to [-1, 1] to avoid numerical errors
        cosine_sim_clamped = np.clip(cosine_sim, -1.0, 1.0)
        angular_dist = np.arccos(cosine_sim_clamped) * 180 / np.pi

        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'dot_product': dot_product,
            'angular_distance': angular_dist,
            'embedding_dim': len(vec1)
        }

    def display_results(self, phrase1, phrase2, metrics):
        """Display comparison results in a nice format."""
        print("\n" + "="*70)
        print("📊 SIMILARITY COMPARISON RESULTS")
        print("="*70)

        # Display phrases
        print(f"\n📝 Phrase 1: \"{phrase1}\"")
        print(f"📝 Phrase 2: \"{phrase2}\"")

        print("\n" + "-"*70)

        # Cosine similarity with visual bar
        cos_sim = metrics['cosine_similarity']
        # Convert from [-1, 1] to [0, 1] for percentage
        similarity_pct = (cos_sim + 1) / 2 * 100

        bar_length = 30
        filled = int(similarity_pct / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n🎯 Cosine Similarity: {cos_sim:.4f}")
        print(f"   [{bar}] {similarity_pct:.1f}%")

        # Interpretation
        if cos_sim >= 0.9:
            interpretation = "Nearly identical meaning"
            emoji = "🟢"
        elif cos_sim >= 0.7:
            interpretation = "Very similar"
            emoji = "🟢"
        elif cos_sim >= 0.5:
            interpretation = "Moderately similar"
            emoji = "🟡"
        elif cos_sim >= 0.3:
            interpretation = "Somewhat related"
            emoji = "🟡"
        elif cos_sim >= 0.0:
            interpretation = "Weakly related"
            emoji = "🟠"
        else:
            interpretation = "Opposite or unrelated"
            emoji = "🔴"

        print(f"   {emoji} {interpretation}")

        # Other metrics
        print(f"\n📐 Angular Distance: {metrics['angular_distance']:.2f}°")
        print(f"   (0° = identical, 90° = unrelated, 180° = opposite)")

        print(f"\n📏 Euclidean Distance: {metrics['euclidean_distance']:.4f}")
        print(f"   (Lower = more similar)")

        print(f"\n🔢 Dot Product: {metrics['dot_product']:.4f}")
        print(f"   (Higher = more similar)")

        print(f"\n🧮 Embedding Dimensions: {metrics['embedding_dim']}")


def display_examples():
    """Display example comparisons."""
    print("\n💡 Example comparisons to try:")
    print("  - 'cat' vs 'kitten' (very similar)")
    print("  - 'happy' vs 'joyful' (synonyms)")
    print("  - 'hot' vs 'cold' (opposites)")
    print("  - 'car' vs 'automobile' (same thing)")
    print("  - 'computer' vs 'pizza' (unrelated)")
    print("  - 'king' vs 'queen' (related concepts)")
    print("  - 'I love programming' vs 'I enjoy coding' (similar sentences)")


def interactive_compare():
    """Run the interactive comparison loop."""
    comparer = SimilarityComparer()

    print("\n" + "="*70)
    print("🔤 PHRASE SIMILARITY COMPARER")
    print("="*70)
    print("\nCompare any two words or phrases to see how similar they are.")
    print("The tool shows multiple similarity metrics to help you understand")
    print("how the embedding model perceives the relationship.")

    display_examples()

    print("\nCommands: 'help' for examples, 'quit' to exit")
    print("-"*70)

    while True:
        try:
            # Get first phrase
            print()
            phrase1 = input("📝 Enter first phrase: ").strip()

            if not phrase1:
                continue

            if phrase1.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if phrase1.lower() == 'help':
                display_examples()
                continue

            # Get second phrase
            phrase2 = input("📝 Enter second phrase: ").strip()

            if not phrase2:
                print("❌ Please enter a second phrase.")
                continue

            # Compare phrases
            print("\n⏳ Computing similarity...")
            metrics = comparer.compare_phrases(phrase1, phrase2)

            # Display results
            comparer.display_results(phrase1, phrase2, metrics)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again with different phrases.")


def main():
    """Main function."""
    print("🚀 Starting Phrase Similarity Comparer...")
    interactive_compare()


if __name__ == "__main__":
    main()