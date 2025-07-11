# 🎓 ChromaDB Learning Tools

Interactive tools to help you understand how semantic similarity and vector embeddings work in ChromaDB.

## 🚀 Quick Start

Both tools use the same embedding model as the main RAG system, so you can see exactly how your searches work under the hood.

```bash
# From the project root directory:

# Tool 1: Explore a database of sample data
uv run python learning/similarity_playground.py

# Tool 2: Compare any two phrases directly
uv run python learning/compare_similarity.py
```

## 📚 Tool 1: Similarity Playground

An interactive environment with ~90 pre-loaded items to explore how semantic search works.

### Features
- Pre-populated with diverse data: animals, food, tech terms, emotions, actions, and sentences
- Search anything and see the top 10 most similar items
- Visual similarity bars show how close matches are (0-100%)
- Temporary database that's cleaned up automatically

### Commands
- **Search**: Type any word or phrase to find similar items
- `list` - Show all 90 items in the database
- `help` - Show example searches
- `quit` or `exit` - Exit the program

### Example Searches to Try
- `feline` - Will match "cat" and "kitten" (synonyms)
- `happy` - Will match "joyful", "excited" (emotional similarity)
- `programming` - Will match "coding", "Python" (conceptual similarity)
- `vehicle` - Will match "car", "automobile", "transportation"
- `delicious food` - Semantic matching across concepts
- `angry animal` - Combines multiple concepts

### Sample Data Categories
- **Animals**: cat, dog, elephant, lion, fish, etc.
- **Food**: pizza, pasta, fruit, vegetables, etc.
- **Technology**: computer, AI, programming, database, etc.
- **Emotions**: happy, sad, excited, anxious, etc.
- **Actions**: running, coding, sleeping, learning, etc.
- **Sentences**: Complete thoughts that combine concepts

## 🔍 Tool 2: Phrase Similarity Comparer

Direct comparison of any two words or phrases with detailed metrics.

### Features
- Enter any two pieces of text to compare
- Multiple similarity metrics explained
- Visual interpretation with color coding
- Shows the actual math behind similarity

### Metrics Explained

1. **Cosine Similarity** (-1 to 1)
   - The main metric used by ChromaDB
   - 1.0 = identical meaning
   - 0.0 = unrelated
   - -1.0 = opposite meaning
   - Shown with visual bar and percentage

2. **Angular Distance** (0° to 180°)
   - How "far apart" two meanings are in vector space
   - 0° = identical
   - 90° = completely unrelated
   - 180° = opposite meanings

3. **Euclidean Distance**
   - Direct distance between embedding vectors
   - Lower = more similar
   - No upper bound

4. **Dot Product**
   - Another similarity measure
   - Higher = more similar

### Visual Interpretation
- 🟢 **Green** (>0.7): Very similar or nearly identical
- 🟡 **Yellow** (0.3-0.7): Moderately similar or somewhat related
- 🟠 **Orange** (0.0-0.3): Weakly related
- 🔴 **Red** (<0.0): Opposite or unrelated

### Example Comparisons
- `cat` vs `kitten` → Very high similarity (same animal, different age)
- `happy` vs `joyful` → Very high similarity (synonyms)
- `hot` vs `cold` → Low/negative similarity (opposites)
- `car` vs `automobile` → Nearly identical (same thing)
- `computer` vs `pizza` → Low similarity (unrelated concepts)
- `I love programming` vs `I enjoy coding` → High similarity (same sentiment)

```
======================================================================
📊 SIMILARITY COMPARISON RESULTS
======================================================================

📝 Phrase 1: "what is the clients outstanding balance"
📝 Phrase 2: "how much does the client have outstanding"

----------------------------------------------------------------------

🎯 Cosine Similarity: 0.7535
   [██████████████████████████░░░░] 87.7%
   🟢 Very similar
```

## 🧠 Understanding the Results

### Why Some Surprising Matches Occur
- **Context matters**: "bank" might match both "river" and "money"
- **Word forms**: "run", "running", "ran" are all similar
- **Conceptual bridges**: "doctor" connects to "hospital", "medicine", "health"

### What Affects Similarity
1. **Semantic meaning**: The actual meaning matters more than exact words
2. **Context**: Words used together influence each other
3. **Training data**: The model learned from millions of text examples
4. **Vector dimensions**: 384 dimensions capture nuanced relationships

## 💡 Learning Exercises

1. **Find Synonyms**: Try words and find their synonyms
2. **Explore Opposites**: See how opposites score (usually 0.0 to -0.3)
3. **Test Analogies**: Compare "king:queen" similarity to "man:woman"
4. **Sentence Variations**: Rephrase sentences and see similarity
5. **Cross-Domain**: Compare words from different domains

## 🔧 Technical Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimensions**: 384
- **Distance Metric**: Cosine similarity (same as main RAG system)
- **Database**: Temporary ChromaDB instance (auto-cleaned)

## 🎯 Why This Matters for RAG

Understanding similarity helps you:
- Write better search queries
- Understand why certain results appear
- Debug unexpected search results
- Optimize your note-taking for better retrieval
- Choose appropriate chunk sizes
