import chromadb

client = chromadb.PersistentClient(
    path=r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/chroma_storage"
)
collection = client.get_or_create_collection("species_embeddings")

collections = client.list_collections()
count = collection.count()
print(count)