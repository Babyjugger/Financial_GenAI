import os
import hashlib
import pickle
from typing import Dict, Optional, Any

class DocumentCacheManager:
    """Manages document caching to avoid reprocessing the same content."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_document_hash(self, content: str) -> str:
        """Generate a unique hash for document content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Try to get cached results for a document by its hash."""
        cache_file = os.path.join(self.cache_dir, f"{document_hash}.pickle")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                print(f"Cache hit for document hash {document_hash[:8]}")
                return result
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {str(e)}")
                # Delete corrupt cache file
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        return None
    
    def save_result_to_cache(self, document_hash: str, result: Dict[str, Any]) -> None:
        """Save processing results to cache."""
        cache_file = os.path.join(self.cache_dir, f"{document_hash}.pickle")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Result saved to cache as {document_hash[:8]}")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pickle'):
                os.remove(os.path.join(self.cache_dir, filename))
        print(f"Cache cleared from {self.cache_dir}")