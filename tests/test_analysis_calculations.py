import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from analysis calculations
from functions.analysis_calculations import filter_data, calculate_workload_summary, get_unknown_assignments

class TestAnalysisCalculations(unittest.TestCase):
    
    def setUp(self):
        # Create test DataFrame
        self.df = pd.DataFrame({
            'uid': ['1', '2', '3', '4', '5'],
            'subject': ['Meeting 1', 'Meeting 2', 'Meeting 3', 'Meeting 4', 'Meeting 5'],
            'start_time': [
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=3),
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=2)
            ],
            'end_time': [
                datetime.now() - timedelta(days=5) + timedelta(hours=1),
                datetime.now() - timedelta(days=3) + timedelta(hours=2),
                datetime.now() - timedelta(days=1) + timedelta(hours=1.5),
                datetime.now() - timedelta(days=10) + timedelta(hours=3),
                datetime.now() - timedelta(days=2) + timedelta(hours=1)
            ],
            'duration_hours': [1.0, 2.0, 1.5, 3.0, 1.0],
            'personnel': ['John', 'John', 'Jane', 'Unknown', 'Jane'],
            'role': ['Doctor', 'Doctor', 'Nurse', 'Unknown', 'Nurse']
        })
        
        # Add date column for filtering
        self.df['date'] = self.df['start_time'].dt.date
    
    def test_filter_data(self):
        """Test filtering data by dates, personnel, and roles"""
        # Test date filtering
        start_date = datetime.now() - timedelta(days=6)
        end_date = datetime.now()
        filtered_df = filter_data(
            self.df,
            start_date=start_date.date(),
            end_date=end_date.date()
        )
        self.assertEqual(len(filtered_df), 4)  # Should exclude the meeting from 10 days ago
        
        # Test personnel filtering
        filtered_df = filter_data(
            self.df,
            start_date=start_date.date(),
            end_date=end_date.date(),
            selected_personnel=['John']
        )
        self.assertEqual(len(filtered_df), 2)  # Only meetings with John
        
        # Test role filtering
        filtered_df = filter_data(
            self.df,
            start_date=start_date.date(),
            end_date=end_date.date(),
            selected_roles=['Nurse']
        )
        self.assertEqual(len(filtered_df), 2)  # Only meetings with Nurses
    
    def test_workload_calculation(self):
        """Test workload summary calculation"""
        workload = calculate_workload_summary(self.df)
        
        # Check if workload is calculated correctly
        self.assertEqual(workload.loc[workload['personnel'] == 'John', 'total_duration_hours'].iloc[0], 3.0)
        self.assertEqual(workload.loc[workload['personnel'] == 'Jane', 'total_duration_hours'].iloc[0], 2.5)
        
        # Check if event counts are correct
        self.assertEqual(workload.loc[workload['personnel'] == 'John', 'total_events'].iloc[0], 2)
        self.assertEqual(workload.loc[workload['personnel'] == 'Jane', 'total_events'].iloc[0], 2)

    def test_unknown_assignments(self):
        """Test unknown assignment identification"""
        unknown_df = get_unknown_assignments(self.df)
        
        # Should contain one unknown assignment
        self.assertEqual(len(unknown_df), 1)
        self.assertEqual(unknown_df.iloc[0]['personnel'], 'Unknown')

if __name__ == '__main__':
    unittest.main()