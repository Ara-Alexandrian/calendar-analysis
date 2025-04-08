# visualization.py
"""
Generates plots for workload analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_workload_duration(workload_df: pd.DataFrame, save_path: str | None = None):
    """Generates and shows/saves a bar plot of total duration per physicist."""
    if workload_df is None or workload_df.empty:
        logging.warning("Workload data is empty. Skipping duration plot.")
        return

    if 'total_duration_hours' not in workload_df.columns or 'physicist' not in workload_df.columns:
        logging.error("Workload DataFrame missing required columns for duration plot.")
        return

    plt.figure(figsize=(12, max(6, len(workload_df) * 0.4))) # Adjust height based on number of physicists
    sns.barplot(data=workload_df.sort_values('total_duration_hours', ascending=False),
                x='total_duration_hours', y='physicist', palette='viridis')
    plt.title('Total Assigned Clinical Duration per Physicist (Hours)')
    plt.xlabel('Total Duration (Hours)')
    plt.ylabel('Physicist')
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(f"{save_path}_duration.png", dpi=300)
            logging.info(f"Duration plot saved to {save_path}_duration.png")
        except Exception as e:
            logging.error(f"Failed to save duration plot: {e}")
    else:
        plt.show()
    plt.close() # Close the plot figure


def plot_workload_events(workload_df: pd.DataFrame, save_path: str | None = None):
    """Generates and shows/saves a bar plot of total events per physicist."""
    if workload_df is None or workload_df.empty:
        logging.warning("Workload data is empty. Skipping event count plot.")
        return

    if 'total_events' not in workload_df.columns or 'physicist' not in workload_df.columns:
         logging.error("Workload DataFrame missing required columns for event count plot.")
         return

    plt.figure(figsize=(12, max(6, len(workload_df) * 0.4))) # Adjust height
    sns.barplot(data=workload_df.sort_values('total_events', ascending=False),
                x='total_events', y='physicist', palette='magma')
    plt.title('Total Number of Assigned Events per Physicist')
    plt.xlabel('Number of Events')
    plt.ylabel('Physicist')
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(f"{save_path}_events.png", dpi=300)
            logging.info(f"Event count plot saved to {save_path}_events.png")
        except Exception as e:
            logging.error(f"Failed to save event count plot: {e}")
    else:
        plt.show()
    plt.close() # Close the plot figure