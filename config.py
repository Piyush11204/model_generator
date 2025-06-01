import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Basic Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    
    # MongoDB configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    MONGODB_DB = os.getenv('MONGODB_DB', 'enhanced_model_generator')
    
    # File storage paths
    BASE_DIR = Path(os.getenv('BASE_DIR', 'ml_workspace'))
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = BASE_DIR / 'Models'
    PREPROCESSORS_FOLDER = BASE_DIR / 'Preprocessors'
    DOC_FOLDER = 'docs'
    
    # Upload limits
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Ensure directories exist
    @classmethod
    def init_app(cls, app):
        # Create necessary directories
        for folder in [cls.UPLOAD_FOLDER, cls.MODEL_FOLDER, 
                      cls.PREPROCESSORS_FOLDER, cls.DOC_FOLDER]:
            os.makedirs(folder, exist_ok=True)