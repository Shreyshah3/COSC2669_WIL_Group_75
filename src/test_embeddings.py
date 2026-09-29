from langchain_ollama import OllamaEmbeddings

print("STEP 1: Starting embedding test")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("STEP 2: Embedding model connected")

text = "How do I block my bank card?"

vector = embeddings.embed_query(text)

print("STEP 3: Embedding generated")
print("Vector length:", len(vector))
print("First 10 values:", vector[:10])

print("STEP 4: Embedding test finished")