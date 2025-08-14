import unittest
from app import create_app
import os

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = create_app('development')
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        self.app_context.pop()
    
    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_route_no_file(self):
        response = self.client.post('/analyze')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
