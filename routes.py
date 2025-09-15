from flask import Blueprint, request, jsonify, render_template, current_app, Flask, redirect, url_for, flash, session

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

import csv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import time
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

@main.route('/analyze', methods=['POST'])
def analyze_circuit():
        st = time.time()
        file_handler = None
    # try:
        # Validate form data
        form_data = validate_form_data(request)
        
        # Handle file uploads
        file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
        files = file_handler.handle_uploads(request.files)
        page_input = request.form.get("page_num", "").strip()  # e.g. "8-10, 12, 13, 16"
        page_list = parse_page_numbers(page_input)
        
        all_wires = []
        all_connections = []
        all_drawn_lines = []
        all_images = []
        pdf_path = Path(files['pdf_path'])
        file_stem = pdf_path.stem 

        # Perform analysis for each page
        for page in page_list:

            matching_xmls = [
                xf for xf in form_data['xml_files'] if page in xf['pages']
            ]
             
            # if page == 8 and file_stem=='247_BSSUBN3D07-0105':
            #     matching_xmls = [{'file': Path(r"C:\Users\Samir.Tak\Downloads\Test-2.v2i.voc\train\247-BSSUBN3D07-01-05_8_png.rf.296084a4e9642174cab87cb120330195.xml"), 'filename': '247-BSSUBN3D07-01-05_8_png.rf.296084a4e9642174cab87cb120330195.xml', 'pages': [8]}]

            # if page == 9 and file_stem=='247_BSSUBN3D07-0105':
            #     matching_xmls = [{'file': Path(r"C:\Users\Samir.Tak\Downloads\Test-2.v2i.voc\train\247-BSSUBN3D07-01-05_9_png.rf.444b0c2d2c5b8cd24b1cd1005e23b4ef.xml"), 'filename': '247-BSSUBN3D07-01-05_9_png.rf.444b0c2d2c5b8cd24b1cd1005e23b4ef.xml', 'pages': [9]}]
            
            # if page == 8 and file_stem=='279_BSSUBN3D11-0108':
            #     matching_xmls = [{'file': Path(r"C:\Users\Samir.Tak\Downloads\Test-2.v2i.voc\train\279-BSSUBN3D11-01-08_8_png.rf.1b6241024565ce6e2b58c39c1fa03fae.xml"), 'filename': '279-BSSUBN3D11-01-08_8_png.rf.1b6241024565ce6e2b58c39c1fa03fae.xml', 'pages': [8]}]
            
            # if page == 11 and file_stem=='157_BS-APK08':
            #     matching_xmls = [{'file': Path(r"C:\Users\Samir.Tak\Downloads\Test-2.v2i.voc\train\157-BS-APK08_11_png.rf.d192ebeb5201e764d89c2389733d81ad.xml"), 'filename': '157-BS-APK08_11_png.rf.d192ebeb5201e764d89c2389733d81ad.xml', 'pages': [11]}]

            # if page == 8 and file_stem=='243_BSSUBN3C07-0106':
            #     matching_xmls = [{'file': Path(r"C:\Users\Samir.Tak\Downloads\Test-2.v2i.voc\train\243-BSSUBN3C07-01-06_8_png.rf.babb414ac6c054f2b97dc09785c33469.xml"), 'filename': '243-BSSUBN3C07-01-06_8_png.rf.babb414ac6c054f2b97dc09785c33469.xml', 'pages': [8]}]

            

            # df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
            #     pdf_file=pdf_path,
            #     xml=files.get('xml_path'),
            #     page_no=page,   # looped page number
            #     enable_network_colors=True
            # )
            if matching_xmls:
                for xml_entry in matching_xmls:
                    df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
                        pdf_file=pdf_path,
                        xml=xml_entry["file"],
                        page_no=page,
                        enable_network_colors=form_data['enable_network_colors']
                    )
            else:
                # No XML given for this page → run with None
                df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
                    pdf_file=pdf_path,
                    xml=None,
                    page_no=page,
                    enable_network_colors=form_data['enable_network_colors']
                )

            # Add page_number column
            if df_wire is not None and not df_wire.empty:
                df_wire["page_number"] = page
                all_wires.append(df_wire)

            if df_connections is not None and not df_connections.empty:
                df_connections["page_number"] = page
                all_connections.append(df_connections)

            # Save images per page
            all_images.append({
                "page": page,
                "combined_canvas": combined_canvas,
                "junction_points": junction_points,
                "line_canvas": line_canvas
            })

            # Save drawn lines per page
            if drawn_lines_lst:
                for i, (wire_name, line_img) in enumerate(drawn_lines_lst):
                    all_drawn_lines.append((page, wire_name, i, line_img))


        # ✅ Merge DataFrames from all pages
        import pandas as pd
        df_wire = pd.concat(all_wires, ignore_index=True) if all_wires else pd.DataFrame()
        df_connections = pd.concat(all_connections, ignore_index=True) if all_connections else pd.DataFrame()

        # Convert DataFrames to dictionaries
        wire_data = df_wire.to_dict('records') if not df_wire.empty else []
        connection_data = df_connections.to_dict('records') if not df_connections.empty else []


        # Convert CV2 images to base64
        def cv2_to_base64(image):
            if image is None:
                return None
            _, buffer = cv2.imencode('.png', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{img_str}"


        # Convert all images per page
        images_b64 = []
        for img_entry in all_images:
            images_b64.append({
                "page": img_entry["page"],
                "combined_canvas": cv2_to_base64(img_entry["combined_canvas"]),
                "junction_points": cv2_to_base64(img_entry["junction_points"]),
                "line_canvas": cv2_to_base64(img_entry["line_canvas"])
            })


        # Convert drawn lines to base64
        drawn_lines_b64 = []
        for page, wire_name, i, line_img in all_drawn_lines:
            line_b64 = cv2_to_base64(line_img)
            if line_b64:
                drawn_lines_b64.append({
                    "page": page,
                    "id": i + 1,
                    "name": f"{wire_name}",
                    "image": line_b64
                })


        # Prepare result data
        result = {
            "wire_data": wire_data,
            "connection_data": connection_data,
            "images": images_b64,        # 👈 now holds per-page images
            "drawn_lines": drawn_lines_b64
        }

        print("Result prepared successfully")
        ed = time.time()
        print("TOTAL TIME", ed-st)
        return jsonify({
            "success": True,
            "message": "Analysis completed successfully!",
            "processed_page": form_data["page_num"],
            "network_colors_enabled": form_data["enable_network_colors"],
            # "xml_used": files.get("xml_path") is not None,
            "result": result
        })
        
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

def save_user(email, password, full_name):
    """Save user details to CSV"""
    password_hash = generate_password_hash(password)
    registration_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(USERS_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([email.lower(), password_hash, full_name, registration_date])

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
    if 'user_email' in session:
        user = get_user_by_email(session['user_email'])
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
        try:
            save_user(email, password, full_name)
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
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))

@main.route('/profile')
@login_required
def profile():
    user = get_user_by_email(session['user_email'])
    return render_template('profile.html', user=user)

# Initialize CSV files
initialize_users_csv()
