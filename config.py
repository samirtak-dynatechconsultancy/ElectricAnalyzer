import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = 'uploads'
    
    # Default paths
    DEFAULT_IMAGE_PATH = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\MDB\raw"
    
    # Allowed extensions
    ALLOWED_EXTENSIONS = {'pdf', 'xml'}
    
    # Create directories
    @staticmethod
    def init_app():
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
