#!/usr/bin/env python3
# Usage: ./index_md.py /path/to/markdown_dir
# (Make this file executable with: chmod +x index_md.py)
"""
Index markdown files into a ChromaDB vector database for semantic search.

This script recursively finds all .md files in a directory, chunks them,
and stores their embeddings for later retrieval.
"""
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from fastembed import TextEmbedding

# Import configuration
from config import *


def smart_chunk_markdown(
    text,
    size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
    min_size=MIN_CHUNK_SIZE,
    max_size=MAX_CHUNK_SIZE,
):
    """
    Split markdown text into intelligent chunks that respect structure.

    Args:
        text: The markdown text to chunk
        size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_size: Minimum chunk size
        max_size: Maximum chunk size

    Returns:
        List of text chunks with metadata
    """
    if len(text) <= size:
        return [{"text": text, "type": "single_chunk"}]

    # Split by markdown headers first (##, ###, etc.)
    header_pattern = r"^(#{1,6}\s+.+)$"
    sections = re.split(header_pattern, text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""
    current_headers = []

    for i, section in enumerate(sections):
        if not section.strip():
            continue

        # Check if this is a header
        if re.match(header_pattern, section, re.MULTILINE):
            # If we have content, save the current chunk
            if current_chunk.strip() and len(current_chunk) >= min_size:
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "type": "section",
                        "headers": current_headers.copy(),
                    }
                )

            # Start new chunk with this header
            current_chunk = section + "\n\n"
            current_headers = [section.strip()]
        else:
            # This is content
            current_chunk += section + "\n\n"

            # If chunk is getting too large, split it
            if len(current_chunk) > max_size:
                # Try to split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", current_chunk)
                if len(sentences) == 1:  # Fallback: no sentence boundaries found
                    # Split by fixed size
                    for start in range(0, len(current_chunk), size):
                        chunk = current_chunk[start:start+size]
                        if len(chunk.strip()) >= min_size:
                            chunks.append({
                                "text": chunk.strip(),
                                "type": "content",
                                "headers": current_headers.copy(),
                            })
                    current_chunk = ""
                else:
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk + sentence) <= size:
                            temp_chunk += sentence + " "
                        else:
                            if temp_chunk.strip() and len(temp_chunk) >= min_size:
                                chunks.append({
                                    "text": temp_chunk.strip(),
                                    "type": "content",
                                    "headers": current_headers.copy(),
                                })
                            temp_chunk = sentence + " "
                    current_chunk = temp_chunk

    # Add the final chunk
    if current_chunk.strip() and len(current_chunk) >= min_size:
        chunks.append(
            {
                "text": current_chunk.strip(),
                "type": "final",
                "headers": current_headers.copy(),
            }
        )

    # If we still have chunks that are too large, apply sliding window
    final_chunks = []
    for chunk_data in chunks:
        text = chunk_data["text"]
        if len(text) <= max_size:
            final_chunks.append(chunk_data)
        else:
            # Apply sliding window with overlap
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk_text = text[start:end]

                # Try to end at a sentence boundary
                if end < len(text):
                    last_sentence = re.search(r"[.!?][^.!?]*$", chunk_text)
                    if last_sentence:
                        end = start + last_sentence.end()
                        chunk_text = text[start:end]

                if len(chunk_text) >= min_size:
                    final_chunks.append({
                        "text": chunk_text.strip(),
                        "type": "windowed",
                        "headers": chunk_data["headers"],
                    })

                next_start = end - overlap
                if next_start <= start:
                    # Prevent infinite loop: always advance at least one character
                    next_start = start + 1
                start = next_start
                if start >= len(text):
                    break

    return final_chunks


def chunk_markdown(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Legacy function for backward compatibility.
    Now uses smart_chunk_markdown internally.
    """
    chunk_data = smart_chunk_markdown(text, size, overlap)
    return [chunk["text"] for chunk in chunk_data]


def read_markdown_files(root):
    """Find all markdown files recursively."""
    return glob.glob(os.path.join(root, "**/*.md"), recursive=True)


def hash_chunk(filepath, chunk):
    """Create a unique ID for each chunk based on file path and content."""
    return hashlib.sha256((filepath + chunk).encode("utf-8")).hexdigest()


def main(directory):
    """Main indexing function."""
    if not os.path.isdir(directory):
        print(f"❌ Error: '{directory}' is not a valid directory")
        sys.exit(1)

    print(f"🔍 Scanning for markdown files in: {directory}")

    # Initialize ChromaDB client (new API)
    print("📊 Initializing vector database...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Use configured embedding model
    print(f"🤖 Using embedding model: {EMBEDDING_MODEL}")

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedder.model_name
    )

    # Get or create collection
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    # Get already indexed IDs
    try:
        existing_data = collection.get(include=["metadatas"])
        indexed_ids = set()
        if existing_data["metadatas"]:
            # Extract IDs from the collection using a different approach
            # Since we can't get IDs directly, we'll use the count to check if collection is empty
            count = collection.count()
            if count > 0:
                # Get all documents to extract their IDs
                all_data = collection.get(limit=count)
                indexed_ids = set(all_data["ids"]) if "ids" in all_data else set()
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve existing IDs: {e}")
        indexed_ids = set()

    print(f"📚 Found {len(indexed_ids)} existing chunks in database")

    # Find all markdown files
    md_files = list(read_markdown_files(directory))
    if not md_files:
        print("⚠️  No markdown files found!")
        return

    print(f"📄 Found {len(md_files)} markdown files")

    added = 0
    skipped_files = 0

    for i, filepath in enumerate(md_files, 1):
        try:
            # Show progress
            relative_path = os.path.relpath(filepath, directory)
            print(
                f"\r[{i}/{len(md_files)}] Processing: {relative_path}...",
                end="",
                flush=True,
            )

            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                skipped_files += 1
                continue

            # Use smart chunking with metadata
            chunk_data = smart_chunk_markdown(content)
            chunks = [chunk["text"] for chunk in chunk_data]
            chunk_ids = [hash_chunk(filepath, chunk) for chunk in chunks]

            # Filter out already indexed chunks
            new_chunks = [
                (cid, chunk, chunk_meta)
                for cid, chunk, chunk_meta in zip(chunk_ids, chunks, chunk_data)
                if cid not in indexed_ids
            ]

            if not new_chunks:
                skipped_files += 1
                continue

            ids, docs, chunk_metas = zip(*new_chunks)

            # Enhanced metadata
            metadatas = []
            for i, chunk_meta in enumerate(chunk_metas):
                metadata = {
                    "source": filepath,
                    "chunk_index": i,
                    "chunk_type": chunk_meta.get("type", ""),
                    "chunk_size": len(chunk_meta.get("text", "")),
                    "filename": os.path.basename(filepath),
                    "headers": (
                        " | ".join(chunk_meta.get("headers", []))
                        if chunk_meta.get("headers")
                        else ""
                    ),
                }
                metadatas.append(metadata)

            collection.add(documents=list(docs), ids=list(ids), metadatas=metadatas)
            added += len(ids)

        except Exception as e:
            print(f"\n⚠️  Error processing {filepath}: {e}")
            continue

    print(f"\n\n✅ Indexing complete!")
    print(f"📊 Statistics:")
    print(f"   - {added} new chunks added")
    print(f"   - {skipped_files} files skipped (already indexed or empty)")
    print(f"   - {len(indexed_ids) + added} total chunks in database")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python index_md.py /path/to/markdown_dir")
            print("\nExample:")
            print("  python index_md.py ~/Documents/notes")
            sys.exit(1)
        main(sys.argv[1])
    except KeyboardInterrupt:
        print("\n🛑 Indexing interrupted by user (Ctrl+C). Exiting cleanly.")
        sys.exit(130)
