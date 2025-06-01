from pymongo import MongoClient
import logging
import os
from pathlib import Path
from datetime import datetime
import json
import glob

logger = logging.getLogger(__name__)

def save_model_metadata(db, metadata):
    """Save model metadata to MongoDB."""
    if db is not None:  # Changed from 'if db:' to 'if db is not None:'
        db.models.insert_one(metadata)
    else:
        # Fallback to JSON file if MongoDB is not available
        models_file = Path('models_metadata.json')
        
        # Load existing data
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    models = json.load(f)
            except:
                models = []
        else:
            models = []
        
        # Convert datetime to string
        if 'upload_date' in metadata and isinstance(metadata['upload_date'], datetime):
            metadata['upload_date'] = metadata['upload_date'].isoformat()
            
        models.append(metadata)
        
        # Save back
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)

def get_all_models(db):
    """Get all models from MongoDB."""
    if db is not None:  # Changed from 'if db:' to 'if db is not None:'
        return list(db.models.find())
    else:
        # Fallback to JSON file if MongoDB is not available
        models_file = Path('models_metadata.json')
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    models = json.load(f)
                    
                # Convert string dates back to datetime
                for model in models:
                    if 'upload_date' in model and isinstance(model['upload_date'], str):
                        try:
                            model['upload_date'] = datetime.fromisoformat(model['upload_date'])
                        except:
                            model['upload_date'] = datetime.now()
                            
                return models
            except:
                return []
        return []

def get_models_from_filesystem(models_dir, docs_dir):
    """Discover models from filesystem as a fallback."""
    models = []
    
    # Find all pickle files in models directory
    model_files = glob.glob(os.path.join(models_dir, "*_best.pkl"))
    
    for model_file in model_files:
        filename = os.path.basename(model_file)
        # Infer model_id from filename (assumes format: name_id_best.pkl)
        parts = filename.split('_')
        if len(parts) >= 2:
            model_id = parts[-2]  # Get the part before "_best.pkl"
            name = '_'.join(parts[:-2])  # Name is everything before model_id
            
            # Look for matching documentation
            doc_file = os.path.join(docs_dir, f"{model_id}.pdf")
            doc_exists = os.path.exists(doc_file)
            
            models.append({
                'model_id': model_id,
                'name': name,
                'model_file_path': model_file,
                'document_file_path': doc_file if doc_exists else None,
                'upload_date': datetime.fromtimestamp(os.path.getctime(model_file)),
                'like_count': 0  # No way to track likes without DB
            })
    
    return models

def save_like(db, model_id, user_ip):
    """Save a like to MongoDB."""
    if db is not None:  # Changed from 'if db:' to 'if db is not None:'
        db.likes.insert_one({
            'model_id': model_id,
            'user_ip': user_ip,
            'timestamp': datetime.utcnow()
        })
    else:
        # Fallback to file if MongoDB is not available
        likes_file = Path('likes.json')
        
        # Load existing data
        if likes_file.exists():
            try:
                with open(likes_file, 'r') as f:
                    likes = json.load(f)
            except:
                likes = []
        else:
            likes = []
            
        likes.append({
            'model_id': model_id,
            'user_ip': user_ip,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Save back
        with open(likes_file, 'w') as f:
            json.dump(likes, f, indent=2)
            
        # Update model like count
        models_file = Path('models_metadata.json')
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    models = json.load(f)
                    
                for model in models:
                    if model['model_id'] == model_id:
                        model['like_count'] = model.get('like_count', 0) + 1
                        break
                        
                with open(models_file, 'w') as f:
                    json.dump(models, f, indent=2)
            except:
                pass

def has_user_liked(db, model_id, user_ip):
    """Check if a user has already liked a model."""
    if db is not None:  # Changed from 'if db:' to 'if db is not None:'
        return db.likes.find_one({'model_id': model_id, 'user_ip': user_ip}) is not None
    else:
        # Fallback to file if MongoDB is not available
        likes_file = Path('likes.json')
        
        # Load existing data
        if likes_file.exists():
            try:
                with open(likes_file, 'r') as f:
                    likes = json.load(f)
                    
                for like in likes:
                    if like['model_id'] == model_id and like['user_ip'] == user_ip:
                        return True
            except:
                pass
        
        return False