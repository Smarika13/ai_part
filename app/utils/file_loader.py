import json
import os
from typing import List, Dict
from langchain.docstore.document import Document

class JSONFileLoader:
    """Load and process JSON files for RAG"""
    
    def __init__(self, data_dir: str = "data/raw/wildlife"):
        self.data_dir = data_dir
    
    def load_json_file(self, filepath: str) -> List[Dict]:
        """Load a single JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    
    def load_all_json_files(self) -> List[Document]:
        """Load all JSON files and convert to LangChain documents"""
        documents = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    try:
                        json_data = self.load_json_file(filepath)
                        docs = self._json_to_documents(json_data, filepath)
                        documents.extend(docs)
                        print(f"✓ Loaded: {filepath} ({len(docs)} entries)")
                    except Exception as e:
                        print(f"✗ Error loading {filepath}: {e}")
        
        return documents
    
    def _json_to_documents(self, json_data: List[Dict], source: str) -> List[Document]:
        """Convert JSON data to LangChain Document objects"""
        documents = []
        
        for idx, item in enumerate(json_data):
            # Convert the entire JSON object to a readable text format
            content = self._format_json_item(item)
            
            metadata = {
                "source": source,
                "index": idx,
                "type": os.path.basename(source).replace('.json', '')
            }
            
            # Add any ID field from the JSON as metadata
            if 'id' in item:
                metadata['id'] = item['id']
            if 'name' in item:
                metadata['name'] = item['name']
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _format_json_item(self, item: Dict) -> str:
        """Format JSON item into readable text"""
        lines = []
        
        for key, value in item.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2)
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines)