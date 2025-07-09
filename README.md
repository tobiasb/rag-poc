# 🧠 Personal Notes RAG System

This is Cursor helping me understand RAG a bit better.

## 🎯 What is this?

This RAG (Retrieval-Augmented Generation) system indexes all your markdown notes and lets you search through them using natural language.

## 🚀 Features

- **Lightning-fast indexing** of all your `.md` files
- **Semantic search** - find notes by meaning, not just keywords
- **Local-first** - your notes never leave your machine
- **Incremental updates** - only indexes new content
- Uses **ChromaDB** for vector storage and **FastEmbed** for embeddings

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management because pip is so 2023.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this repo
git clone <your-repo-url>
cd rag-poc

# Install dependencies
uv sync
```

## 🎮 Usage

### 1. Index Your Notes

First, tell the system where your markdown notes live:

```bash
uv run python index_md.py /path/to/your/notes

# Example:
uv run python index_md.py ~/Documents/obsidian-vault
```

The indexer will:
- Recursively find all `.md` files
- Chunk them into digestible pieces
- Create embeddings and store them locally
- Skip already indexed content (it's smart like that!)

### 2. Search Your Brain Dump

Now the fun part - ask questions!

```bash
uv run python search_md.py "how do I configure neovim?"

# Or get philosophical:
uv run python search_md.py "what did I think about that book on productivity?"

# Or practical:
uv run python search_md.py "meeting notes from last Tuesday"
```

## 🧪 How It Works

1. **Chunking**: Your notes are split into ~500 character chunks (because even AI has attention limits)
2. **Embedding**: Each chunk gets converted into a vector using `BAAI/bge-base-en-v1.5`
3. **Storage**: ChromaDB stores these vectors locally in `./chroma_db`
4. **Search**: Your query gets embedded and compared against all chunks
5. **Results**: Top 5 most relevant chunks are returned with their sources

## 🛠️ Configuration

Want to tweak things? Here's what you can adjust:

- `CHUNK_SIZE` in `index_md.py` - Make chunks bigger/smaller
- `n_results` in `search_md.py` - Get more/fewer results
- Change the embedding model (but `bge-base-en-v1.5` is pretty solid)

## 🔄 Resetting the Database

Need to start fresh? Maybe you changed your chunking strategy or want to re-index everything:

```bash
# Delete the entire database
rm -rf chroma_db/

# Then re-index your notes
uv run python index_md.py /path/to/your/notes
```

This will completely remove all indexed data and start over from scratch.

## 📁 Project Structure

```
rag-poc/
├── index_md.py      # The indexer - feeds your notes to the AI
├── search_md.py     # The searcher - asks the AI about your notes
├── pyproject.toml   # Dependencies and project config
├── chroma_db/       # Your local vector database (gitignored)
└── README.md        # You are here! 👋
```
