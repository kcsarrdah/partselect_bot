# Split Document objects into smaller, semantically meaningful chunks for better retrieval accuracy.
"""
Chunking Service
Splits documents into smaller chunks for better retrieval accuracy.
Applies smart chunking based on document type.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.document_loader import DocumentLoader

class ChunkingService:
    """Service for chunking documents based on their type."""
    
    def __init__(
        self,
        blog_chunk_size: int = 1000,
        blog_chunk_overlap: int = 200
    ):
        """
        Initialize the chunking service.
        
        Args:
            blog_chunk_size: Size of chunks for blog documents (in characters)
            blog_chunk_overlap: Overlap between chunks (in characters)
        """
        self.blog_chunk_size = blog_chunk_size
        self.blog_chunk_overlap = blog_chunk_overlap
        
        # Initialize text splitter for blogs
        self.blog_splitter = RecursiveCharacterTextSplitter(
            chunk_size=blog_chunk_size,
            chunk_overlap=blog_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document based on its type.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunked documents (or single document if no chunking needed)
        """
        doc_type = document.metadata.get('type', 'unknown')
        
        # Only chunk blogs - parts and repairs stay as-is
        if doc_type == 'blog':
            return self._chunk_blog(document)
        else:
            # Parts and repairs: keep as-is (already atomic)
            return [document]
    
    def _chunk_blog(self, document: Document) -> List[Document]:
        """
        Chunk a blog document into smaller pieces.
        
        Args:
            document: Blog document to chunk
            
        Returns:
            List of chunked documents with preserved metadata
        """
        # Split the text
        chunks = self.blog_splitter.split_text(document.page_content)
        
        # If document is small enough, don't chunk
        if len(chunks) <= 1:
            return [document]
        
        # Create new documents for each chunk
        chunked_docs = []
        for idx, chunk_content in enumerate(chunks):
            # Preserve original metadata and add chunk info
            chunk_metadata = document.metadata.copy()
            chunk_metadata['chunk_index'] = idx
            chunk_metadata['total_chunks'] = len(chunks)
            chunk_metadata['chunk_id'] = f"{document.metadata.get('doc_id', 'unknown')}_chunk_{idx}"
            
            chunked_doc = Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            )
            chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_documents = []
        
        for doc in documents:
            chunked_documents.extend(self.chunk_document(doc))
        
        return chunked_documents
    
    def chunk_by_type(self, documents: List[Document]) -> dict:
        """
        Chunk documents and group by type for analysis.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            Dictionary with chunking statistics by type
        """
        chunked_docs = []
        stats = {
            'blog': {'original': 0, 'chunked': 0},
            'part': {'original': 0, 'chunked': 0},
            'repair': {'original': 0, 'chunked': 0},
            'unknown': {'original': 0, 'chunked': 0}
        }
        
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            
            # Track original count
            if doc_type in stats:
                stats[doc_type]['original'] += 1
            else:
                stats['unknown']['original'] += 1
            
            # Chunk the document
            chunks = self.chunk_document(doc)
            chunked_docs.extend(chunks)
            
            # Track chunked count
            if doc_type in stats:
                stats[doc_type]['chunked'] += len(chunks)
            else:
                stats['unknown']['chunked'] += len(chunks)
        
        return {
            'documents': chunked_docs,
            'stats': stats
        }
    
    def get_chunk_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about document chunking without actually chunking.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_docs': len(documents),
            'by_type': {},
            'estimated_chunks': 0
        }
        
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            
            if doc_type not in stats['by_type']:
                stats['by_type'][doc_type] = {
                    'count': 0,
                    'avg_length': 0,
                    'total_length': 0,
                    'will_chunk': doc_type == 'blog'
                }
            
            stats['by_type'][doc_type]['count'] += 1
            doc_length = len(doc.page_content)
            stats['by_type'][doc_type]['total_length'] += doc_length
            
            # Estimate chunks for blogs
            if doc_type == 'blog':
                estimated = max(1, doc_length // self.blog_chunk_size)
                stats['estimated_chunks'] += estimated
            else:
                stats['estimated_chunks'] += 1
        
        # Calculate averages
        for doc_type in stats['by_type']:
            count = stats['by_type'][doc_type]['count']
            total = stats['by_type'][doc_type]['total_length']
            stats['by_type'][doc_type]['avg_length'] = total // count if count > 0 else 0
        
        return stats


# Convenience function
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Convenience function to chunk documents.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Size of chunks for blog documents
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents
    """
    service = ChunkingService(
        blog_chunk_size=chunk_size,
        blog_chunk_overlap=chunk_overlap
    )
    return service.chunk_documents(documents)


if __name__ == "__main__":
    # Test with document loader
    
    print("Testing Chunking Service...")
    print("\n=== Loading documents ===")
    # Load documents
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_all_documents()
    
    if not documents:
        print("No documents loaded. Exiting.")
        exit(1)
    
    print(f"Loaded {len(documents)} documents")
    
    # Get stats before chunking
    service = ChunkingService()
    print("\n=== Chunk Statistics (Before) ===")
    stats = service.get_chunk_stats(documents)
    print(f"Total documents: {stats['total_docs']}")
    print(f"Estimated chunks after processing: {stats['estimated_chunks']}")
    print("\nBy type:")
    for doc_type, type_stats in stats['by_type'].items():
        print(f"  {doc_type}:")
        print(f"    Count: {type_stats['count']}")
        print(f"    Avg length: {type_stats['avg_length']} chars")
        print(f"    Will chunk: {type_stats['will_chunk']}")
    
    # Chunk documents
    print("\n=== Chunking documents ===")
    result = service.chunk_by_type(documents)
    chunked_docs = result['documents']
    chunk_stats = result['stats']
    
    print(f"\nOriginal documents: {len(documents)}")
    print(f"Chunked documents: {len(chunked_docs)}")
    print("\nChunking by type:")
    for doc_type, type_stats in chunk_stats.items():
        if type_stats['original'] > 0:
            print(f"  {doc_type}: {type_stats['original']} → {type_stats['chunked']}")
    
    # Show sample chunks
    print("\n=== Sample Chunks ===")
    for doc_type in ['blog', 'part', 'repair']:
        sample = next((d for d in chunked_docs if d.metadata.get('type') == doc_type), None)
        if sample:
            print(f"\n{doc_type.upper()} sample:")
            print(f"  Content length: {len(sample.page_content)} chars")
            print(f"  Metadata: {sample.metadata}")
            if 'chunk_index' in sample.metadata:
                print(f"  Chunk {sample.metadata['chunk_index'] + 1} of {sample.metadata['total_chunks']}")