# should take data from the data/raw folder and process it and stores it into data/processed foleder. 

"""
Document Loader Service
Loads CSV files and converts them into LangChain Document objects with metadata.
"""

import os
import csv
from typing import List, Dict, Any
from langchain_core.documents import Document
from utils.logger import setup_logger, log_success, log_warning, log_error

logger = setup_logger(__name__)


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
            log_warning(logger, f"{file_path} not found, skipping blogs.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('title') or not row.get('url'):
                        log_warning(logger, f"Skipping blog row {idx} - missing title or url")
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
            
            log_success(logger, f"Loaded {len(documents)} blog documents")
            
        except Exception as e:
            log_error(logger, f"Error loading blogs CSV: {e}")
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
            log_warning(logger, f"{file_path} not found, skipping {appliance_type} parts.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                skipped_count = 0
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('part_name'):
                        skipped_count += 1
                        # Only log first few skipped rows, then summarize
                        if skipped_count <= 3:
                            logger.debug(f"Skipping part row {idx} - missing part_name")
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
                        "install_time": row.get('install_time', 'N/A'),
                        "symptoms": row.get('symptoms', 'N/A'),
                        "product_types": row.get('product_types', 'N/A'),
                        "replace_parts": row.get('replace_parts', 'N/A'),
                        "availability": row.get('availability', 'N/A'),
                        "install_video_url": row.get('install_video_url', ''),
                        "product_url": row.get('product_url', 'N/A'),
                        "doc_id": f"part_{appliance_type}_{row.get('part_id', idx)}"
                    }
                    
                    documents.append(Document(
                        page_content=page_content,
                        metadata=metadata
                    ))
            
            # Log summary of skipped rows
            if skipped_count > 0:
                if skipped_count <= 3:
                    logger.info(f"Skipped {skipped_count} rows with missing part_name")
                else:
                    logger.warning(f"Skipped {skipped_count} rows with missing part_name (dataset issue)")
            
            log_success(logger, f"Loaded {len(documents)} {appliance_type} part documents")
            
        except Exception as e:
            log_error(logger, f"Error loading parts CSV: {e}")
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
            log_warning(logger, f"{file_path} not found, skipping {appliance_type} repairs.")
            return []
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    # Skip rows with missing critical data
                    if not row.get('symptom'):
                        log_warning(logger, f"Skipping repair row {idx} - missing symptom")
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
            
            log_success(logger, f"Loaded {len(documents)} {appliance_type} repair documents")
            
        except Exception as e:
            log_error(logger, f"Error loading repairs CSV: {e}")
            return []
        
        return documents
    
    def load_all_documents(
        self,
        blog_files: List[str] = None,
        parts_files: Dict[str, str] = None,
        repairs_files: Dict[str, str] = None
    ) -> List[Document]:
        """
        Load all CSV files and return combined list of documents.
        
        Args:
            blog_files: List of blog CSV filenames. If None, auto-discovers.
            parts_files: Dict of {appliance_type: filename}. If None, auto-discovers.
            repairs_files: Dict of {appliance_type: filename}. If None, auto-discovers.
        
        Returns:
            List of all Document objects from all sources
        """
        import glob
        
        all_documents = []
        logger.info("\n=== Loading all documents ===")
        
        # === LOAD BLOGS ===
        if blog_files is None:
            blog_pattern = os.path.join(self.data_dir, "*blog*.csv")
            blog_files = [os.path.basename(f) for f in glob.glob(blog_pattern)]
        
        for blog_file in blog_files:
            file_path = os.path.join(self.data_dir, blog_file)
            if os.path.exists(file_path):
                logger.info(f"📰 Loading blog: {blog_file}")
                all_documents.extend(self.load_blogs_csv(file_path))
        
        # === LOAD PARTS ===
        if parts_files is None:
            parts_pattern = os.path.join(self.data_dir, "*part*.csv")
            discovered_parts = glob.glob(parts_pattern)
            
            parts_files = {}
            for file_path in discovered_parts:
                filename = os.path.basename(file_path).lower()
                if "refrigerator" in filename or "fridge" in filename:
                    parts_files["refrigerator"] = os.path.basename(file_path)
                elif "dishwasher" in filename or "dish" in filename:
                    parts_files["dishwasher"] = os.path.basename(file_path)
                else:
                    appliance_type = os.path.splitext(os.path.basename(file_path))[0].replace("_parts", "").replace("_part", "")
                    parts_files[appliance_type] = os.path.basename(file_path)
        
        for appliance_type, filename in parts_files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"🔧 Loading {appliance_type} parts: {filename}")
                all_documents.extend(self.load_parts_csv(file_path, appliance_type))
        
        # === LOAD REPAIRS ===
        if repairs_files is None:
            repairs_pattern = os.path.join(self.data_dir, "*repair*.csv")
            discovered_repairs = glob.glob(repairs_pattern)
            
            repairs_files = {}
            for file_path in discovered_repairs:
                filename = os.path.basename(file_path).lower()
                if "refrigerator" in filename or "fridge" in filename:
                    repairs_files["refrigerator"] = os.path.basename(file_path)
                elif "dishwasher" in filename or "dish" in filename:
                    repairs_files["dishwasher"] = os.path.basename(file_path)
                else:
                    appliance_type = os.path.splitext(os.path.basename(file_path))[0].replace("_repairs", "").replace("_repair", "")
                    repairs_files[appliance_type] = os.path.basename(file_path)
        
        for appliance_type, filename in repairs_files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"🔨 Loading {appliance_type} repairs: {filename}")
                all_documents.extend(self.load_repairs_csv(file_path, appliance_type))
        
        log_success(logger, f"Total documents loaded: {len(all_documents)}")
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
    logger.info("Testing Document Loader...")
    documents = load_documents()
    
    if documents:
        logger.info(f"\nSample document:")
        logger.info(f"Content: {documents[0].page_content[:200]}...")
        logger.info(f"Metadata: {documents[0].metadata}")
    else:
        log_warning(logger, "No documents loaded.")