from pathlib import Path
import os

class PathManager:
    @staticmethod
    def ensure_directory_exists(path):
        """Create directory if it doesn't exist"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_output_directory(image_path, pdf_path, page_num):
        """Generate output directory path"""
        parent_parent_name = Path(image_path).parent.name
        file_name = Path(pdf_path).stem
        return Path(f"{os.getcwd()}/{parent_parent_name}/output/{page_num}_{file_name}")
    
    @staticmethod
    def validate_path_exists(path):
        """Check if path exists"""
        return Path(path).exists()
