from flask import Blueprint, request, jsonify, render_template, current_app
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
from flask import send_from_directory

main = Blueprint('main', __name__)

@main.route('/')
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
        file_handler = None
    # try:
        # Validate form data
        form_data = validate_form_data(request)
        
        # Handle file uploads
        file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
        files = file_handler.handle_uploads(request.files)
        page_input = request.form.get("page_num", "").strip()  # e.g. "8-10, 12, 13, 16"
        page_list = parse_page_numbers(page_input)

        # Prepare output directory
        # output_dir = Path(f"{os.getcwd()}/{parent_parent_name}/output/{form_data['page_num']}_{file_name}")
        # output_dir.mkdir(parents=True, exist_ok=True)
        
        all_wires = []
        all_connections = []
        all_drawn_lines = []
        all_images = []
        pdf_path = Path(files['pdf_path'])
        # Perform analysis for each page
        for page in page_list:
            matching_xmls = [
                xf for xf in form_data['xml_files'] if page in xf['pages']
            ]

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
                    "name": f"{wire_name} (Line Drawing {i + 1})",
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
