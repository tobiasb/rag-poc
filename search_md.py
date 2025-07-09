#!/usr/bin/env python3
# Usage: ./search_md.py "your search query here"
# (Make this file executable with: chmod +x search_md.py)
"""
Search through indexed markdown files using semantic search.

This script queries the ChromaDB vector database to find the most
relevant chunks of text based on natural language queries.
"""
import os
import re
import sys
from pathlib import Path
from textwrap import fill

import chromadb
from chromadb.utils import embedding_functions
from fastembed import TextEmbedding

# Import configuration
from config import *


def preprocess_query(query):
    """
    Preprocess the query to improve search relevance.
    """
    if not ENABLE_QUERY_PREPROCESSING:
        return query

    # Split query into words
    words = query.lower().split()

    # Remove stop words but keep important ones
    filtered_words = []
    for word in words:
        if word not in STOP_WORDS or len(word) > 3:  # Keep longer stop words
            filtered_words.append(word)

    # Reconstruct query
    processed_query = ' '.join(filtered_words)

    # If query is too short after filtering, use original
    if len(processed_query.split()) < MIN_QUERY_LENGTH:
        processed_query = query

    return processed_query


def format_result(doc, meta, distance, index, terminal_width=TERMINAL_WIDTH):
    """Format a single search result for display."""
    print(f"\n{'='*terminal_width}")
    print(f"📄 Result #{index}")

    # Show source file
    source_path = Path(meta["source"])
    print(f"📁 Source: {source_path.name}")
    print(f"   Path: {source_path}")

    # Show chunk metadata if enabled
    if SHOW_CHUNK_METADATA:
        if "chunk_type" in meta:
            print(f"📋 Chunk Type: {meta['chunk_type']}")
        if "headers" in meta and meta["headers"]:
            print(f"📑 Headers: {meta['headers']}")
        if "chunk_size" in meta:
            print(f"📏 Size: {meta['chunk_size']} chars")

    # Show relevance score if enabled
    if SHOW_RELEVANCE_SCORES:
        relevance = max(0, 1 - distance)
        relevance_bar = "█" * int(relevance * 20) + "░" * (20 - int(relevance * 20))
        print(f"📊 Relevance: [{relevance_bar}] {relevance:.1%}")

    # Show the text content with word wrapping
    print(f"\n📝 Content:")
    wrapped_text = fill(
        doc, width=terminal_width - 3, initial_indent="   ", subsequent_indent="   "
    )
    print(wrapped_text)


def main(query, num_results=DEFAULT_NUM_RESULTS):
    """Main search function."""
    if not os.path.exists(DB_PATH):
        print("❌ Error: No indexed documents found!")
        print(
            "   Please run 'python index_md.py /path/to/notes' first to index your markdown files."
        )
        sys.exit(1)

    print(f"🔍 Searching for: '{query}'")
    print("⏳ Loading vector database...")

    try:
                # Initialize ChromaDB client (new API)
        client = chromadb.PersistentClient(path=DB_PATH)

        # Use configured embedding model
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedder.model_name
        )

        # Get collection
        if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
            print("❌ Error: No documents have been indexed yet!")
            print("   Please run 'python index_md.py /path/to/notes' first.")
            sys.exit(1)

        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )

        # Check if collection is empty
        count = collection.count()
        if count == 0:
            print("⚠️  Warning: The database is empty. No documents have been indexed.")
            sys.exit(1)

        print(f"📚 Searching through {count} text chunks...")

        # Preprocess query
        processed_query = preprocess_query(query)
        if processed_query != query:
            print(f"🔧 Processed query: '{processed_query}'")

        # Perform the search
        results = collection.query(
            query_texts=[processed_query],
            n_results=min(num_results, count),
        )

        if not results["documents"][0]:
            print("\n😕 No relevant results found. Try:")
            print("   - Rephrasing your query")
            print("   - Using different keywords")
            print("   - Removing filters if any were applied")
            return

        # Filter results by relevance threshold
        filtered_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            relevance = max(0, 1 - dist)
            if relevance >= MIN_RELEVANCE_THRESHOLD:
                filtered_results.append((doc, meta, dist))

        if not filtered_results:
            print(f"\n😕 No results above relevance threshold ({MIN_RELEVANCE_THRESHOLD:.1%}).")
            print("   Try lowering the threshold in config.py or rephrasing your query.")
            return

        # Display results
        print(f"\n✨ Found {len(filtered_results)} relevant results:")

        for i, (doc, meta, dist) in enumerate(filtered_results, 1):
            format_result(doc, meta, dist, i)

        print(f"\n{'='*80}")
        print("💡 Tips for better results:")
        print("   - Use specific technical terms")
        print("   - Try different phrasings")
        print("   - Use quotes for exact phrases")

    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        print(
            "   Make sure you've indexed some documents first with 'python index_md.py'"
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python search_md.py "your search query here" [options]')
        print("\nOptions:")
        print("  --results=N     Number of results (default: 5)")
        print("\nExamples:")
        print('  python search_md.py "how to configure vim"')
        print('  python search_md.py "meeting notes from project X" --results=10')
        sys.exit(1)

    # Parse arguments
    query_parts = []
    num_results = DEFAULT_NUM_RESULTS

    for arg in sys.argv[1:]:
        if arg.startswith("--results="):
            try:
                num_results = int(arg.split("=")[1])
                if num_results > MAX_NUM_RESULTS:
                    print(f"⚠️  Warning: Limiting results to {MAX_NUM_RESULTS} (configured maximum)")
                    num_results = MAX_NUM_RESULTS
            except ValueError:
                print("❌ Error: --results must be a number")
                sys.exit(1)
        else:
            query_parts.append(arg)

    if not query_parts:
        print("❌ Error: No search query provided")
        sys.exit(1)

    query = " ".join(query_parts)
    main(query, num_results)
