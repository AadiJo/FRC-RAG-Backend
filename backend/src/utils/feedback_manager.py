import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FeedbackManager:
    """
    Manages user feedback storage and retrieval.
    Stores feedback in a JSONL file.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "feedback.jsonl")
        os.makedirs(data_dir, exist_ok=True)
        
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback to the JSONL file.
        
        Args:
            feedback_data: Dictionary containing feedback details
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure timestamp exists
            if "timestamp" not in feedback_data:
                feedback_data["timestamp"] = datetime.now().isoformat()
                
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback_data) + "\n")
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False
            
    def get_feedback(self, feedback_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve feedback entries, optionally filtered by type.
        
        Args:
            feedback_type: 'good', 'bad', or 'redo' (optional)
            limit: Maximum number of entries to return (most recent first)
            
        Returns:
            List of feedback entries
        """
        entries = []
        try:
            if not os.path.exists(self.feedback_file):
                return []
                
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if feedback_type and entry.get("feedback_type") != feedback_type:
                            continue
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
                        
            # Return most recent first
            return sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            return []

    def get_good_examples(self) -> List[Dict[str, str]]:
        """
        Get a list of (query, response) pairs from 'good' feedback.
        """
        good_feedback = self.get_feedback(feedback_type="good", limit=1000)
        return [
            {"query": f["query"], "response": f.get("response_full", f.get("response_preview", ""))} 
            for f in good_feedback 
            if "query" in f and ("response_full" in f or "response_preview" in f)
        ]
