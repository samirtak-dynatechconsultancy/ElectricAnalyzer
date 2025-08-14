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

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze_circuit():
        file_handler = None
    # try:
        # Validate form data
        form_data = validate_form_data(request)
        
        # Handle file uploads
        file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
        files = file_handler.handle_uploads(request.files)
        
        # Prepare output directory
        # output_dir = Path(f"{os.getcwd()}/{parent_parent_name}/output/{form_data['page_num']}_{file_name}")
        # output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        df_wire, df_connections, combined_canvas, junction_points, line_canvas, drawn_lines_lst = combined_circuit_analysis_improved(
            pdf_file=Path(files['pdf_path']),
            xml=files.get('xml_path'),
            page_no=form_data['page_num'],
            enable_network_colors=True
        )
        
        # Convert DataFrames to dictionaries
        wire_data = df_wire.to_dict('records') if df_wire is not None else []
        connection_data = df_connections.to_dict('records') if df_connections is not None else []
        
        # Convert CV2 images to base64
        def cv2_to_base64(image):
            if image is None:
                return None
            _, buffer = cv2.imencode('.png', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        
        # Convert images to base64
        combined_canvas_b64 = cv2_to_base64(combined_canvas)
        junction_points_b64 = cv2_to_base64(junction_points)
        line_canvas_b64 = cv2_to_base64(line_canvas)
        
        # Convert drawn lines list to base64
        drawn_lines_b64 = []
        if drawn_lines_lst:
            for i, (wire_name, line_img) in enumerate(drawn_lines_lst):
                line_b64 = cv2_to_base64(line_img)
                if line_b64:
                    drawn_lines_b64.append({
                        'id': i + 1,
                        'name': f'{wire_name} (Line Drawing {i + 1})',
                        'image': line_b64
                    })
        
        # Prepare result data
        result = {
            'wire_data': wire_data,
            'connection_data': connection_data,
            'images': {
                'combined_canvas': combined_canvas_b64,
                'junction_points': junction_points_b64,
                'line_canvas': line_canvas_b64
            },
            'drawn_lines': drawn_lines_b64
        }
        print("Result prepared successfully")
        # print('\n\n\n\n\n\n\n\n\n')
        # print(result)
        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully!',
            'processed_page': form_data['page_num'],
            'network_colors_enabled': form_data['enable_network_colors'],
            'xml_used': files.get('xml_path') is not None,
            'result': result
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
