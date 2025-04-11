import unittest
import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.data_processor import process_calendar_data

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        # Sample test data
        self.sample_data = [
            {
                "subject": "Test Meeting 1",
                "start": {"dateTime": "2023-04-10T09:00:00"},
                "end": {"dateTime": "2023-04-10T10:00:00"},
                "attendees": [{"name": "John Doe"}, {"name": "Jane Smith"}]
            },
            {
                "subject": "Test Meeting 2",
                "start": {"dateTime": "2023-04-10T14:00:00"},
                "end": {"dateTime": "2023-04-10T15:30:00"},
                "attendees": []
            }
        ]
    
    def test_date_processing(self):
        """Test that dates are correctly processed"""
        result = process_calendar_data(self.sample_data)
        
        # Check if start_time and end_time are datetime objects
        self.assertIsInstance(result.iloc[0]['start_time'], datetime)
        self.assertIsInstance(result.iloc[0]['end_time'], datetime)
        
        # Check if duration is calculated correctly
        self.assertEqual(result.iloc[0]['duration_hours'], 1.0)
        self.assertEqual(result.iloc[1]['duration_hours'], 1.5)
    
    def test_empty_attendees(self):
        """Test handling of empty attendees"""
        result = process_calendar_data(self.sample_data)
        
        # Second meeting has no attendees
        self.assertTrue('Unknown' in result.iloc[1]['personnel'])
        
if __name__ == '__main__':
    unittest.main()