from flask import Blueprint, request, jsonify, render_template, current_app, Flask, redirect, url_for, flash, session, send_file

from werkzeug.utils import secure_filename
import os
from pathlib import Path
import uuid
import shutil
import base64
import cv2
from utils.file_handler import FileHandler
from utils.validation import ValidationError, validate_form_data
from analysis.circuit_analyzer import combined_circuit_analysis_improved
# from analysis.circuit_analyzer_copy import combined_circuit_analysis_improved
# from analysis.circuit_analyzer_text import combined_circuit_analysis_improved
from flask import send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler
import csv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import time
from optimize_pdf.ghostscript import optimize_pdf
import fitz
import hashlib

main = Blueprint('main', __name__)


from functools import wraps

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please login to access this page.', 'error')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function

@main.route('/home')
@login_required
def index():
    return render_template('index_.html')


ALLOWED_EXTENSIONS = {'pdf'}
box_counter = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
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

@main.route('/get_page/<filename>/<int:page_num>')
def get_page(filename, page_num):
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
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

@main.route('/add_white_box', methods=['POST'])
def add_white_box():
    try:
        data = request.json
        filename = data.get('filename')

        # Get the optimized file
        filename = filename.replace(".pdf", "_optimized.pdf")
        page_num = int(data.get('page_num'))
        x = float(data.get('x'))
        y = float(data.get('y'))
        width = float(data.get('width', 50))
        height = float(data.get('height', 20))

        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # ✅ Manage dictionary for incremental numbering
        if filename not in box_counter:
            box_counter[filename] = {}
        if page_num not in box_counter[filename]:
            box_counter[filename][page_num] = 0
        
        # Increment counter for this page
        box_counter[filename][page_num] += 1
        current_id = box_counter[filename][page_num]
        label_text = f"Φ{current_id}"  # e.g. Φ1, Φ2, Φ3
        
        # Open PDF
        doc = fitz.open(filepath)
        page = doc[page_num]
        
        page_rect = page.rect
        original_rotation = page.rotation
        
        print(f"=== DEBUG INFO ===")
        print(f"Original coordinates from frontend: x={x}, y={y}, width={width}, height={height}")
        print(f"Page rect: {page_rect}")
        print(f"Page dimensions: {page_rect.width} x {page_rect.height}")
        print(f"Page rotation: {original_rotation}")
        
        pdf_x1 = x - width/2
        pdf_y1 = y - height/2
        pdf_x2 = x + width/2
        pdf_y2 = y + height/2

        pdf_x1, pdf_x2 = min(pdf_x1, pdf_x2), max(pdf_x1, pdf_x2)
        pdf_y1, pdf_y2 = min(pdf_y1, pdf_y2), max(pdf_y1, pdf_y2)
        
        print(f"Converted PDF coordinates: x1={pdf_x1}, y1={pdf_y1}, x2={pdf_x2}, y2={pdf_y2}")
        
        if pdf_x1 < 0 or pdf_y1 < 0 or pdf_x2 > page_rect.width or pdf_y2 > page_rect.height:
            print(f"WARNING: Coordinates outside page bounds!")
        
        rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
        print(f"Final rectangle: {rect}")
        
        # Add white rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(fill=[1, 1, 1])  # White fill  
        annot.update()

        # --- ADD TEXT IN THE CENTER ---
        font_size = min(height * 0.6, width * 0.6)  # Scale font to fit box
        text_rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)

        page.insert_textbox(
            text_rect,
            label_text,
            fontname="helv",       # Helvetica
            fontsize=font_size,
            align=1,              # Center alignment
            color=(0, 0, 0)       # Black text
        )
        
        print(f"Added text '{label_text}' at {text_rect}")

        # Save modified PDF
        print(f"Saving modified PDF to {filepath}")
        doc.save(filepath, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()
        
        return jsonify({
            'success': True,
            'message': 'White box and text added successfully',
            'id': current_id,
            'label': label_text
        })
        
    except Exception as e:
        print(f"Error in add_white_box: {str(e)}")
        return jsonify({'error': f'Error adding white box: {str(e)}'}), 400


@main.route('/download_modified/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=f'modified_{filename}')
        
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 400


def parse_page_numbers(page_input: str):
    pages = []
    for part in page_input.split(","):
        part = part.strip()
        if "-" in part:  # range like 8-10
            start, end = map(int, part.split("-"))
            pages.extend(range(start, end + 1))
        elif part.isdigit():  # single page
            pages.append(int(part))
    return pages

RESULTS_FOLDER = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@main.route('/results/<unique_id>/<path:filename>')
def serve_result_file(unique_id, filename):
    """Serve saved result images."""
    user_results_folder = os.path.join(RESULTS_FOLDER, unique_id)
    return send_from_directory(user_results_folder, filename)


@main.route('/optimize', methods=['POST'])
def optimize_file():
    box_counter = {}
    file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
    files = file_handler.handle_uploads(request.files)
    pdf_path = Path(files['pdf_path'])
    output_path = pdf_path.with_name(f"{pdf_path.stem}_optimized{pdf_path.suffix}")

    # Handle file uploads

    optimize_pdf(pdf_path, output_path, quality="ebook")
    return jsonify({'success': True, 'message': 'PDF optimized successfully', 'optimized_file': os.path.basename(output_path)})

@main.route('/analyze', methods=['POST'])
def analyze_circuit():
    try:
        folder_path = os.path.join(RESULTS_FOLDER, session.get('id', 'guest'))
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Deletes folder + all files inside it
        st = time.time()
        page_input = request.form.get("page_num", "").strip()
        page_list = parse_page_numbers(page_input)

        all_wires, all_connections, all_drawn_lines, all_images = [], [], [], []
        files = FileHandler(current_app.config['UPLOAD_FOLDER']).get_files(request.files)
        pdf_path = Path(files['pdf_path'])
        output_path = pdf_path.with_name(f"{pdf_path.stem}_optimized{pdf_path.suffix}")
        form_data = validate_form_data(request)

        pdf_path = output_path

        file_stem = pdf_path.stem

        for page in page_list:
            matching_xmls = [xf for xf in form_data['xml_files'] if page in xf['pages']]
            if matching_xmls:
                for xml_entry in matching_xmls:
                    df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
                        pdf_file=pdf_path,
                        xml=xml_entry["file"],
                        page_no=page,
                        enable_network_colors=form_data['enable_network_colors']
                    )
            else:
                df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
                    pdf_file=pdf_path,
                    xml=None,
                    page_no=page,
                    enable_network_colors=form_data['enable_network_colors']
                )

            if df_wire is not None and not df_wire.empty:
                df_wire["page_number"] = page
                all_wires.append(df_wire)

            if df_connections is not None and not df_connections.empty:
                df_connections["page_number"] = page
                all_connections.append(df_connections)

            all_images.append({
                "page": page,
                "combined_canvas": combined_canvas,
                "junction_points": junction_points,
                "line_canvas": line_canvas
            })

            if drawn_lines_lst:
                for i, (wire_name, line_img) in enumerate(drawn_lines_lst):
                    all_drawn_lines.append((page, wire_name, i, line_img))

        df_wire = pd.concat(all_wires, ignore_index=True) if all_wires else pd.DataFrame()
        df_connections = pd.concat(all_connections, ignore_index=True) if all_connections else pd.DataFrame()
        df_connections["pdf_name"] = file_stem   # all rows get this value

        wire_data = df_wire.to_dict('records') if not df_wire.empty else []
        connection_data = df_connections.to_dict('records') if not df_connections.empty else []

        # ✅ Save images to disk and build URLs
        def save_img(img, filename):
            if img is None:
                return None
            user_folder = os.path.join(RESULTS_FOLDER, session.get('id', 'guest'))
            os.makedirs(user_folder, exist_ok=True)  # ✅ Ensure folder exists

            full_path = os.path.join(user_folder, filename)
            
            print("THIS IS full_path:", full_path)
            cv2.imwrite(full_path, img)
            return f"/results/{session.get('id', 'guest')}/{filename}"

        images_output = []
        for img_entry in all_images:
            page = img_entry["page"]
            images_output.append({
                "page": page,
                "combined_canvas": save_img(img_entry["combined_canvas"], secure_filename(f"{file_stem}_p{page}_combined.png")),
                "junction_points": save_img(img_entry["junction_points"], secure_filename(f"{file_stem}_p{page}_junction.png")),
                "line_canvas": save_img(img_entry["line_canvas"], secure_filename(f"{file_stem}_p{page}_lines.png")),
            })

        drawn_lines_output = []
        for page, wire_name, i, line_img in all_drawn_lines:
            filename = secure_filename(f"{file_stem}_p{page}_{wire_name}.png")
            url = save_img(line_img, filename)
            drawn_lines_output.append({
                "page": page,
                "id": i + 1,
                "name": wire_name,
                "image_url": url
            })

        result = {
            "wire_data": wire_data,
            "connection_data": connection_data,
            "images": images_output,
            "drawn_lines": drawn_lines_output
        }

        ed = time.time()
        print("Result prepared successfully")
        print("TOTAL TIME", ed - st)
        return jsonify({
            "success": True,
            "message": "Analysis completed successfully!",
            "processed_page": form_data["page_num"],
            "network_colors_enabled": form_data["enable_network_colors"],
            "result": result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    # except Exception as e:
    #     current_app.logger.error(f"Analysis error: {str(e)}")
    #     if file_handler:
    #         file_handler.cleanup()
    #     return jsonify({
    #         'success': False,
    #         'error': f'Analysis failed: {str(e)}'
    #     }), 500
    
    # finally:
    #     if file_handler:
    #         file_handler.cleanup()

@main.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(main.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@main.route('/agent')
def agent():
    return render_template('test.html')
    
@main.route('/download_csv/<data_type>')
def download_csv(data_type):
    """Endpoint to download CSV files"""
    try:
        # You would need to store the analysis results in session or database
        # For now, this is a placeholder
        if data_type == 'wire':
            # Get wire data from session/database
            pass
        elif data_type == 'connection':
            # Get connection data from session/database
            pass
        
        # Create CSV response
        # This would return the actual CSV file
        pass
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@main.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500



ALLOWED_EMAILS_FILE = 'allowed_emails.csv'
USERS_FILE = 'users.csv'

def read_csv(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(file_path, fieldnames, rows):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

@main.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'user_email' not in session or session['user_email'] != 'admin@admin.com':
        flash("Unauthorized access!", "error")
        return redirect(url_for('main.login'))

    allowed_emails = read_csv(ALLOWED_EMAILS_FILE)  # [{'useremail': '...'}]
    users = read_csv(USERS_FILE)  # [{'email':..., 'password_hash':..., 'full_name':..., 'registration_date':...}]

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_email':
            new_email = request.form.get('new_email', '').strip().lower()
            if new_email:
                if any(row['useremail'].lower() == new_email for row in allowed_emails):
                    flash(f"{new_email} is already allowed.", "warning")
                else:
                    allowed_emails.append({'useremail': new_email})
                    write_csv(ALLOWED_EMAILS_FILE, ['useremail'], allowed_emails)
                    flash(f"{new_email} added to allowed emails.", "success")
            else:
                flash("Please provide a valid email.", "error")

        elif action == 'update_password':
            user_email = request.form.get('user_email', '').strip().lower()
            new_password = request.form.get('new_password', '')
            if not new_password:
                flash("Password cannot be empty!", "error")
            else:
                updated = False
                for user in users:
                    if user['email'].lower() == user_email:
                        user['password_hash'] = generate_password_hash(new_password)
                        updated = True
                        break
                if updated:
                    write_csv(USERS_FILE, ['email', 'password_hash', 'full_name', 'registration_date'], users)
                    flash(f"Password updated for {user_email}", "success")
                else:
                    flash(f"User {user_email} not found!", "error")

        return redirect(url_for('main.admin'))

    return render_template('admin.html', users=users, allowed_emails=allowed_emails)

def load_allowed_emails():
    """Load allowed emails from CSV file"""
    allowed_emails = set()
    if os.path.exists(ALLOWED_EMAILS_FILE):
        try:
            with open(ALLOWED_EMAILS_FILE, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'useremail' in row:
                        allowed_emails.add(row['useremail'].strip().lower())
        except Exception as e:
            print(f"Error loading allowed emails: {e}")
    return allowed_emails

def initialize_users_csv():
    """Initialize users CSV file with headers if it doesn't exist"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email', 'password_hash', 'full_name', 'registration_date'])

def save_user(unique_id, email, password, full_name):
    """Save user details to CSV"""
    password_hash = generate_password_hash(password)
    registration_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(USERS_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([unique_id, email.lower(), password_hash, full_name, registration_date])

def get_user_by_email(email):
    """Get user details by email from CSV"""
    if not os.path.exists(USERS_FILE):
        return None
    
    try:
        with open(USERS_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['email'].lower() == email.lower():
                    return row
    except Exception as e:
        print(f"Error reading users file: {e}")
    
    return None

def is_email_registered(email):
    """Check if email is already registered"""
    return get_user_by_email(email) is not None

@main.route('/')
def home():
    folder_path = os.path.join(RESULTS_FOLDER, session.get('id', 'guest'))
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes folder + all files inside it

    if 'user_email' in session:
        user = get_user_by_email(session['user_email'])
        print(user)
        return render_template('index_.html', user=user)
    return render_template('index.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        
        # Validation
        if not email or not password or not full_name:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('register.html')
        
        # Check if email is in allowed list
        allowed_emails = load_allowed_emails()
        if email.lower() not in allowed_emails:
            flash('This email is not authorized to create an account!', 'error')
            return render_template('register.html')
        
        # Check if email is already registered
        if is_email_registered(email):
            flash('This email is already registered!', 'error')
            return render_template('register.html')
        
        # Save user
        unique_id = hashlib.sha256(email.lower().encode()).hexdigest()[:16]

        try:
            save_user(unique_id, email, password, full_name)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('main.login'))
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password are required!', 'error')
            return render_template('login.html')
        
        user = get_user_by_email(email)
        if user and check_password_hash(user['password_hash'], password):
            session['user_email'] = email.lower()
            session['id'] = hashlib.sha256(email.lower().encode()).hexdigest()[:16]
            flash('Login successful!', 'success')
            if email.lower() == 'admin@admin.com':
                return redirect(url_for('main.admin'))
            return redirect(url_for('main.home'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html')

@main.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('id', None)
    session.pop('session_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))

@main.route('/profile')
@login_required
def profile():
    user = get_user_by_email(session['user_email'])
    return render_template('profile.html', user=user)

# Initialize CSV files
initialize_users_csv()