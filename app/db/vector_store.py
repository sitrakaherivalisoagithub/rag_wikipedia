import uuid
from typing import List

from langchain_core.documents import Document


class VectorStoreManager:
    """
    Manages all interactions with the vector store, providing a clear
    interface for inserting, searching, and deleting documents.
    """

    def __init__(self, store):
        """
        Initializes the manager with a vector store instance.

        Args:
            store: An instance of a LangChain vector store (e.g., PGVector).
        """
        self.store = store

    def insert_documents(self, documents: List[Document]) -> List[str]:
        """
        Inserts a list of documents into the vector store in batches.
        The 'id' in metadata is used as the primary key for additions.

        Args:
            documents: A list of LangChain Document objects. Each document's
                       metadata should contain a unique 'id'.

        Returns:
            A list of the IDs of the added documents.
        """
        batch_size = 20  # Adjust batch size as needed
        all_inserted_ids = []
        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            ids = [str(doc.metadata.get("id", uuid.uuid4())) for doc in batch_documents]
            print(f"--- Inserting batch of {len(batch_documents)} documents ---")
            inserted_ids = self.store.add_documents(batch_documents, ids=ids)
            all_inserted_ids.extend(inserted_ids)

        print(f"--- Successfully inserted {len(all_inserted_ids)} documents in total ---")
        return all_inserted_ids

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search.

        Args:
            query: The search query string.
            k: The number of documents to return.

        Returns:
            A list of matching documents.
        """
        print(f"--- Searching for '{query}' ---")
        return self.store.similarity_search(
            query,
            k=k
        )

    def delete_all_documents(self):
        """
        Deletes all vectors from the collection.
        """
        print(f"--- Deleting all documents from collection \'{self.store.collection_name}\' ---")
        self.store.client.recreate_collection(
            collection_name=self.store.collection_name,
            vectors_config=self.store.vector_params,
        )
        print("All documents deleted successfully.")
        return True

