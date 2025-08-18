import os
from app import create_app

# Set configuration (use 'production' for live)
config_name = os.getenv('FLASK_CONFIG') or 'production'

# Create Flask app
app = create_app(config_name)
