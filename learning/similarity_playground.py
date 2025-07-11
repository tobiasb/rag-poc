#!/usr/bin/env python3
"""
Interactive ChromaDB similarity playground.

This script creates a temporary ChromaDB instance with sample data
and provides an interactive prompt to explore how semantic similarity works.
"""
import chromadb
from chromadb.utils import embedding_functions
from fastembed import TextEmbedding
import tempfile
import shutil
from pathlib import Path

# Sample data - diverse words, phrases, and sentences
SAMPLE_DATA = [
    # Animals
    "cat",
    "dog",
    "puppy",
    "kitten",
    "elephant",
    "mouse",
    "lion",
    "tiger",
    "fish",
    "shark",
    "dolphin",
    "whale",

    # Food
    "pizza",
    "pasta",
    "bread",
    "sandwich",
    "apple",
    "orange",
    "banana",
    "fruit salad",
    "vegetable soup",
    "ice cream",
    "chocolate cake",

    # Technology
    "computer",
    "laptop",
    "smartphone",
    "artificial intelligence",
    "machine learning",
    "neural network",
    "database",
    "programming",
    "Python code",
    "JavaScript",

    # Actions/Verbs
    "running",
    "walking",
    "jumping",
    "swimming",
    "flying",
    "coding",
    "eating",
    "sleeping",
    "thinking",
    "learning",

    # Emotions
    "happy",
    "sad",
    "angry",
    "excited",
    "joyful",
    "depressed",
    "anxious",
    "calm",
    "peaceful",
    "frustrated",

    # Short sentences
    "The cat is sleeping on the couch",
    "Dogs love to play fetch",
    "I am learning how to code",
    "The weather is sunny today",
    "Machine learning is fascinating",
    "Pizza is my favorite food",
    "I feel happy when I'm coding",
    "The ocean is deep and blue",
    "Birds fly south for winter",
    "Coffee helps me wake up",

    # Related concepts
    "automobile",
    "car",
    "vehicle",
    "transportation",
    "bicycle",
    "motorcycle",

    # Opposites
    "hot",
    "cold",
    "big",
    "small",
    "fast",
    "slow",
    "light",
    "dark",
    "up",
    "down",

    # Abstract concepts
    "love",
    "hate",
    "freedom",
    "justice",
    "beauty",
    "truth",
    "wisdom",
    "knowledge",
    "understanding",
    "consciousness",
]


def create_temp_database():
    """Create a temporary ChromaDB instance and populate it with sample data."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="chroma_playground_")
    print(f"📁 Created temporary database at: {temp_dir}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=temp_dir)

    # Use a lightweight embedding model for speed
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"🤖 Using embedding model: {embedding_model}")

    embedder = TextEmbedding(model_name=embedding_model)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedder.model_name
    )

    # Create collection
    collection = client.create_collection(
        name="similarity_playground",
        embedding_function=embedding_fn
    )

    # Add sample data
    print(f"📊 Adding {len(SAMPLE_DATA)} samples to database...")

    ids = [f"sample_{i}" for i in range(len(SAMPLE_DATA))]
    metadatas = [
        {
            "type": "animal" if any(animal in text.lower() for animal in ["cat", "dog", "elephant", "lion", "tiger", "fish", "shark", "dolphin", "whale", "puppy", "kitten", "mouse"]) else
                   "food" if any(food in text.lower() for food in ["pizza", "pasta", "bread", "apple", "orange", "banana", "fruit", "vegetable", "ice cream", "chocolate", "sandwich"]) else
                   "tech" if any(tech in text.lower() for tech in ["computer", "laptop", "smartphone", "artificial intelligence", "machine learning", "neural network", "database", "programming", "python", "javascript"]) else
                   "emotion" if any(emotion in text.lower() for emotion in ["happy", "sad", "angry", "excited", "joyful", "depressed", "anxious", "calm", "peaceful", "frustrated"]) else
                   "action" if any(action in text.lower() for action in ["running", "walking", "jumping", "swimming", "flying", "coding", "eating", "sleeping", "thinking", "learning"]) else
                   "other",
            "length": len(text),
            "word_count": len(text.split())
        }
        for text in SAMPLE_DATA
    ]

    collection.add(
        documents=SAMPLE_DATA,
        ids=ids,
        metadatas=metadatas
    )

    print("✅ Database populated successfully!")
    return client, collection, temp_dir


def search_and_display(collection, query, num_results=10):
    """Search the collection and display results with similarity scores."""
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 60)

    # Perform search
    results = collection.query(
        query_texts=[query],
        n_results=min(num_results, collection.count())
    )

    if not results["documents"][0]:
        print("No results found.")
        return

    # Display results
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        # Convert distance to similarity score (0-100%)
        similarity = max(0, 1 - dist) * 100

        # Create visual similarity bar
        bar_length = 20
        filled = int(similarity / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n{i}. [{bar}] {similarity:.1f}% similar")
        print(f"   Text: '{doc}'")
        print(f"   Type: {meta['type']} | Words: {meta['word_count']}")


def interactive_search(collection):
    """Run interactive search loop."""
    print("\n" + "="*60)
    print("🎮 SIMILARITY PLAYGROUND")
    print("="*60)
    print("\nEnter search queries to see how semantic similarity works.")
    print("The database contains animals, food, technology terms, emotions, and more.")
    print("\nCommands:")
    print("  'list' - Show all items in database")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("-"*60)

    while True:
        try:
            query = input("\n🔎 Enter search query (or command): ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            elif query.lower() == 'list':
                print("\n📋 All items in database:")
                print("-"*60)
                for i, item in enumerate(SAMPLE_DATA, 1):
                    print(f"{i:3d}. {item}")

            elif query.lower() == 'help':
                print("\n💡 Try these example searches:")
                print("  - 'feline' (will match cat, kitten)")
                print("  - 'happy' (will match joyful, excited)")
                print("  - 'programming' (will match coding, Python)")
                print("  - 'vehicle' (will match car, automobile)")
                print("  - 'delicious food' (semantic matching)")
                print("  - 'angry animal' (combines concepts)")

            else:
                search_and_display(collection, query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function."""
    print("🚀 Starting ChromaDB Similarity Playground...")

    # Create temporary database
    client, collection, temp_dir = create_temp_database()

    try:
        # Run interactive search
        interactive_search(collection)
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary database...")
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete!")


if __name__ == "__main__":
    main()