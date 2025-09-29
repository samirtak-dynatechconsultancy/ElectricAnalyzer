import os
import json
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get PDF info
        try:
            doc = fitz.open(filepath)
            page_count = len(doc)
            doc.close()
            
            return jsonify({
                'success': True,
                'filename': filename,
                'page_count': page_count
            })
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/get_page/<filename>/<int:page_num>')
def get_page(filename, page_num):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        doc = fitz.open(filepath)
        if page_num < 0 or page_num >= len(doc):
            return jsonify({'error': 'Invalid page number'}), 400
        
        page = doc[page_num]
        
        # Get page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Render page as image with high DPI for quality
        mat = fitz.Matrix(4,4)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to base64 for web display
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        doc.close()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'page_width': page_width,
            'page_height': page_height,
            'display_width': pix.width,
            'display_height': pix.height
        })
        
    except Exception as e:
        return jsonify({'error': f'Error rendering page: {str(e)}'}), 400

@app.route('/add_white_box', methods=['POST'])
def add_white_box():
    try:
        data = request.json
        filename = data.get('filename')
        page_num = data.get('page_num')
        x = float(data.get('x'))
        y = float(data.get('y'))
        width = float(data.get('width', 50))
        height = float(data.get('height', 20))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Open PDF
        doc = fitz.open(filepath)
        page = doc[page_num]
        
        # Get page dimensions and rotation
        page_rect = page.rect
        original_rotation = page.rotation
                
        pdf_x1 = x - width/2
        pdf_y1 = y - height/2
        pdf_x2 = x + width/2
        pdf_y2 = y + height/2

        # Ensure coordinates are in correct order
        pdf_x1, pdf_x2 = min(pdf_x1, pdf_x2), max(pdf_x1, pdf_x2)
        pdf_y1, pdf_y2 = min(pdf_y1, pdf_y2), max(pdf_y1, pdf_y2)
        
        
        # Validate coordinates are within page bounds
        
        # Create rectangle
        rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
        
        # Add white rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(fill=[1, 0, 0])  # White fill  
        # annot.set_border(width=1, color=[1, 0, 0])  # Red border for debugging
        annot.update()
        
        # Save the modified PDF
        doc.save(filepath, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()
        
        return jsonify({'success': True, 'message': 'White box added successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error adding white box: {str(e)}'}), 400



@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=f'modified_{filename}')
        
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)