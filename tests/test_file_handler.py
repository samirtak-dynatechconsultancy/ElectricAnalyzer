import unittest
import tempfile
import os
from utils.file_handler import FileHandler

class TestFileHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_handler = FileHandler(self.temp_dir)
    
    def tearDown(self):
        self.file_handler.cleanup()
    
    def test_session_folder_creation(self):
        self.assertTrue(os.path.exists(self.file_handler.session_folder))
    