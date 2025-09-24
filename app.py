from flask import Flask
from config import config
from models import db, Log, Report  # import your models

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize config and database
    config[config_name].init_app(app)
    db.init_app(app)

    # Create tables automatically (if not exist)
    with app.app_context():
        db.create_all()

    # Register blueprints
    from routes import main
    app.register_blueprint(main)
    
    return app
