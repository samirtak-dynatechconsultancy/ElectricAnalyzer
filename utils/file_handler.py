import os
import shutil
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

class FileHandler:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
    
    def handle_uploads(self, files):
        result = {}
        
        # Handle PDF file
        if 'pdf_file' in files:
            pdf_file = files['pdf_file']
            if pdf_file and pdf_file.filename:
                pdf_filename = secure_filename(pdf_file.filename)
                pdf_path = os.path.join(self.upload_folder, pdf_filename)
                pdf_file.save(pdf_path)
                result['pdf_path'] = pdf_path
        
        # Handle XML file (optional)
        if 'xml_file' in files:
            xml_file = files['xml_file']
            if xml_file and xml_file.filename:
                xml_filename = secure_filename(xml_file.filename)
                xml_path = os.path.join(self.upload_folder, xml_filename)
                xml_file.save(xml_path)
                result['xml_path'] = xml_path
        
        return result
    
    def get_files(self, files):
        result = {}
        
        # Get PDF file path
        if 'pdf_file' in files:
            pdf_file = files['pdf_file']
            if pdf_file and pdf_file.filename:
                pdf_filename = secure_filename(pdf_file.filename)
                pdf_path = os.path.join(self.upload_folder, pdf_filename)
                result['pdf_path'] = pdf_path
        
        # Get XML file path (optional)
        if 'xml_file' in files:
            xml_file = files['xml_file']
            if xml_file and xml_file.filename:
                xml_filename = secure_filename(xml_file.filename)
                xml_path = os.path.join(self.upload_folder, xml_filename)
                result['xml_path'] = xml_path

        return result

    # def cleanup(self):
    #     try:
    #         shutil.rmtree(self.session_folder)
    #     except Exception as e:
    #         print(f"Warning: Could not cleanup {self.session_folder}: {e}")
