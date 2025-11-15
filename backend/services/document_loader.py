# should take data from the data/raw folder and process it and stores it into data/processed foleder. 

"""
Document Loader Service
Loads CSV files and converts them into LangChain Document objects with metadata.
"""

import os
import csv
from typing import List, Dict, Any
from langchain_core.documents import Document


class DocumentLoader:
    """Load and process CSV files into LangChain Documents."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the document loader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
    
    def load_blogs_csv(self, file_path: str) -> List[Document]:
        """
        Load blog entries from CSV.
        
        Expected columns: title, url
        
        Args:
            file_path: Path to blogs CSV file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping blogs.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('title') or not row.get('url'):
                        print(f"Warning: Skipping blog row {idx} - missing title or url")
                        continue
                    
                    # Create page content
                    page_content = f"""Title: {row['title']}
URL: {row['url']}"""
                    
                    # Create metadata
                    metadata = {
                        "source": "blogs",
                        "type": "blog",
                        "title": row['title'],
                        "url": row['url'],
                        "doc_id": f"blog_{idx}"
                    }
                    
                    documents.append(Document(
                        page_content=page_content,
                        metadata=metadata
                    ))
            
            print(f"Loaded {len(documents)} blog documents from {file_path}")
            
        except Exception as e:
            print(f"Error loading blogs CSV: {e}")
            return []
        
        return documents
    
    def load_parts_csv(self, file_path: str, appliance_type: str) -> List[Document]:
        """
        Load parts from CSV.
        
        Expected columns: part_name, part_id, mpn_id, part_price, 
                         install_difficulty, install_time, symptoms, 
                         product_types, brand, availability, product_url
        
        Args:
            file_path: Path to parts CSV file
            appliance_type: 'refrigerator' or 'dishwasher'
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {appliance_type} parts.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('part_name'):
                        print(f"Warning: Skipping part row {idx} - missing part_name")
                        continue
                    
                    # Create rich page content
                    page_content = f"""Part: {row.get('part_name', 'N/A')}
Part ID: {row.get('part_id', 'N/A')}
MPN: {row.get('mpn_id', 'N/A')}
Price: {row.get('part_price', 'N/A')}
Brand: {row.get('brand', 'N/A')}
Installation Difficulty: {row.get('install_difficulty', 'N/A')}
Installation Time: {row.get('install_time', 'N/A')}
Availability: {row.get('availability', 'N/A')}
Fixes these symptoms: {row.get('symptoms', 'N/A')}
Compatible with: {row.get('product_types', 'N/A')}
Replaces parts: {row.get('replace_parts', 'N/A')}"""
                    
                    # Create metadata
                    metadata = {
                        "source": "parts",
                        "type": "part",
                        "appliance": appliance_type,
                        "part_name": row.get('part_name', 'N/A'),
                        "part_id": row.get('part_id', 'N/A'),
                        "mpn_id": row.get('mpn_id', 'N/A'),
                        "brand": row.get('brand', 'N/A'),
                        "price": row.get('part_price', 'N/A'),
                        "difficulty": row.get('install_difficulty', 'N/A'),
                        "url": row.get('product_url', 'N/A'),
                        "doc_id": f"part_{appliance_type}_{row.get('part_id', idx)}"
                    }
                    
                    documents.append(Document(
                        page_content=page_content,
                        metadata=metadata
                    ))
            
            print(f"Loaded {len(documents)} {appliance_type} part documents from {file_path}")
            
        except Exception as e:
            print(f"Error loading parts CSV: {e}")
            return []
        
        return documents
    
    def load_repairs_csv(self, file_path: str, appliance_type: str) -> List[Document]:
        """
        Load repair guides from CSV.
        
        Expected columns: Product, symptom, description, percentage, 
                         parts, difficulty, repair_video_url, symptom_detail_url
        
        Args:
            file_path: Path to repairs CSV file
            appliance_type: 'refrigerator' or 'dishwasher'
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {appliance_type} repairs.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('symptom'):
                        print(f"Warning: Skipping repair row {idx} - missing symptom")
                        continue
                    
                    # Create problem-focused page content
                    video_text = "Yes" if row.get('repair_video_url') else "No"
                    
                    page_content = f"""Problem: {row.get('symptom', 'N/A')}
Appliance: {row.get('Product', appliance_type)}
Description: {row.get('description', 'N/A')}
Reported by: {row.get('percentage', 'N/A')}% of users
Difficulty: {row.get('difficulty', 'N/A')}
Parts needed: {row.get('parts', 'N/A')}
Video guide available: {video_text}"""
                    
                    # Create metadata
                    metadata = {
                        "source": "repairs",
                        "type": "repair",
                        "appliance": appliance_type,
                        "symptom": row.get('symptom', 'N/A'),
                        "difficulty": row.get('difficulty', 'N/A'),
                        "parts_needed": row.get('parts', 'N/A'),
                        "video_url": row.get('repair_video_url', ''),
                        "detail_url": row.get('symptom_detail_url', ''),
                        "percentage": row.get('percentage', '0'),
                        "doc_id": f"repair_{appliance_type}_{idx}"
                    }
                    
                    documents.append(Document(
                        page_content=page_content,
                        metadata=metadata
                    ))
            
            print(f"Loaded {len(documents)} {appliance_type} repair documents from {file_path}")
            
        except Exception as e:
            print(f"Error loading repairs CSV: {e}")
            return []
        
        return documents
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all CSV files and return combined list of documents.
        
        Returns:
            List of all Document objects from all sources
        """
        all_documents = []
        
        print("\n=== Loading all documents ===")
        
        # Try different file naming patterns (test fixtures or real data)
        blog_files = ["test_blogs.csv", "partselect_blogs_test.csv", "partselect_blogs.csv"]
        for filename in blog_files:
            blogs_path = os.path.join(self.data_dir, filename)
            if os.path.exists(blogs_path):
                all_documents.extend(self.load_blogs_csv(blogs_path))
                break
        
        # Load refrigerator parts
        part_files = ["test_parts.csv", "refrigerator_parts_test.csv", "refrigerator_parts.csv"]
        for filename in part_files:
            fridge_parts_path = os.path.join(self.data_dir, filename)
            if os.path.exists(fridge_parts_path):
                all_documents.extend(self.load_parts_csv(fridge_parts_path, "refrigerator"))
                break
        
        # Load dishwasher parts (optional)
        dish_parts_path = os.path.join(self.data_dir, "dishwasher_parts_test.csv")
        if os.path.exists(dish_parts_path):
            all_documents.extend(self.load_parts_csv(dish_parts_path, "dishwasher"))
        
        # Load refrigerator repairs
        repair_files = ["test_repairs.csv", "refrigerator_repairs_test.csv", "refrigerator_repairs.csv"]
        for filename in repair_files:
            fridge_repairs_path = os.path.join(self.data_dir, filename)
            if os.path.exists(fridge_repairs_path):
                all_documents.extend(self.load_repairs_csv(fridge_repairs_path, "refrigerator"))
                break
        
        # Load dishwasher repairs (optional)
        dish_repairs_path = os.path.join(self.data_dir, "dishwasher_repairs_test.csv")
        if os.path.exists(dish_repairs_path):
            all_documents.extend(self.load_repairs_csv(dish_repairs_path, "dishwasher"))
        
        print(f"\n=== Total documents loaded: {len(all_documents)} ===\n")
        
        return all_documents


# Convenience function for quick usage
def load_documents(data_dir: str = "data/raw") -> List[Document]:
    """
    Convenience function to load all documents.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of all Document objects
    """
    loader = DocumentLoader(data_dir)
    return loader.load_all_documents()


if __name__ == "__main__":
    # Test the loader
    print("Testing Document Loader...")
    documents = load_documents()
    
    if documents:
        print(f"\nSample document:")
        print(f"Content: {documents[0].page_content[:200]}...")
        print(f"Metadata: {documents[0].metadata}")
    else:
        print("No documents loaded.")