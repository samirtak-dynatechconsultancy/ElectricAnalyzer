from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.types import Text

db = SQLAlchemy()

# ---------------------------
# Log table
# ---------------------------
class Log(db.Model):
    __tablename__ = 'log'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    file = db.Column(db.String(255), nullable=False)
    page = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # automatically stores time

# ---------------------------
# Report table
# ---------------------------
class Report(db.Model):
    __tablename__ = 'report'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    wire_no = db.Column(db.String(50), nullable=True)
    source_component = db.Column(db.String(100), nullable=True)
    source_terminal = db.Column(db.String(100), nullable=True)
    dest_component = db.Column(db.String(100), nullable=True)
    dest_terminal = db.Column(db.String(100), nullable=True)
    selected = db.Column(db.Boolean, default=False)
    reason = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # automatically stores time

