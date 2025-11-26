import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("species_embeddings")

collections = client.list_collections()
print(collections)