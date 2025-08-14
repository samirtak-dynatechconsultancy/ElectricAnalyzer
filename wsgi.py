import os
from app import create_app

# Set configuration (use 'production' for live)
config_name = os.getenv('FLASK_CONFIG') or 'production'

# Create Flask app
application = create_app(config_name)
