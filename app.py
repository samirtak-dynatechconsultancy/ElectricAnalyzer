from flask import Flask
from config import config
import os

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize configuration
    config[config_name].init_app()
    
    # Register blueprints
    from routes import main
    app.register_blueprint(main)
    
    return app
