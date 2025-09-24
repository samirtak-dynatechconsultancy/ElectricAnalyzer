import os
from pathlib import Path

basedir = Path(__file__).parent.resolve()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = basedir / 'uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'xml'}
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app=None):
        # Create directories if they don't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(basedir / 'logs', exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + str(basedir / 'database.db')

class ProductionConfig(Config):
    DEBUG = False
    # Example: PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + str(basedir / 'database.db')

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
