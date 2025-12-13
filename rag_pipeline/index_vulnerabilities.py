"""Index all vulnerabilities into Qdrant."""

from data_loader import load_vulnerabilities, create_embedding_text
from embedder import Embedder
from vector_store import VectorStore


def main():
    print("Loading vulnerabilities...")
    vulnerabilities = load_vulnerabilities()
    print(f"Found {len(vulnerabilities)} vulnerabilities.")
    
    print("Initializing embedder and vector store...")
    embedder = Embedder()
    store = VectorStore()
    store.create_collection(recreate=True)
    
    # Prepare texts for embedding
    texts = [create_embedding_text(v) for v in vulnerabilities]
    
    # Embed in batches of 100
    print("Embedding vulnerabilities...")
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    
    # Prepare points for Qdrant
    points = []
    for idx, (vuln, embedding) in enumerate(zip(vulnerabilities, all_embeddings)):
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {
                "contract_name": vuln["contract_name"],
                "file_path": vuln["file_path"],
                "category": vuln["category"],
                "vulnerable_lines": vuln["vulnerable_lines"],
                "code_snippet": vuln["code_snippet"],
                "description": vuln["description"]
            }
        })
    
    print("Storing in Qdrant...")
    store.upsert(points)
    print(f"Successfully indexed {len(points)} vulnerabilities!")


if __name__ == "__main__":
    main()

