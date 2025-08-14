from app import create_app
import os

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

if __name__ == '__main__':
    print("🌟 Starting PDF Circuit Analysis Server...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
