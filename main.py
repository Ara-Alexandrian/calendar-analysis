# main.py
"""
Main script to run the calendar workload analysis pipeline using Ollama.
"""
import logging
import pandas as pd
from tqdm import tqdm # For progress bar

# Import project modules
import config
import data_processor
import llm_extractor
import analysis
import visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """Executes the full analysis pipeline."""
    logging.info("Starting calendar analysis pipeline...")

    # 1. Load Data
    logging.info(f"Loading data from: {config.JSON_FILE_PATH}")
    df = data_processor.load_data(config.JSON_FILE_PATH)
    if df is None or df.empty:
        logging.error("Failed to load data. Exiting.")
        return

    # 2. Preprocess Data (Dates, Duration)
    logging.info("Preprocessing data (parsing dates, calculating duration)...")
    df_processed = data_processor.preprocess_data(df)
    if df_processed.empty:
        logging.error("Preprocessing failed or resulted in empty DataFrame. Exiting.")
        return

    # 3. Extract Physicists using LLM
    # Updated logging message to include Ollama host
    logging.info(f"Extracting physicists using LLM ({config.LLM_PROVIDER} - {config.LLM_MODEL} at {config.OLLAMA_BASE_URL})...")

    # Updated check for Ollama client readiness
    if config.LLM_PROVIDER == "ollama" and llm_extractor.llm_client is None:
         logging.error(f"Ollama Client not available or failed to connect to {config.OLLAMA_BASE_URL}. Cannot proceed with LLM extraction.")
         # Updated error message for Ollama context
         print(f"\nPlease ensure the Ollama server is running at {config.OLLAMA_BASE_URL} and the model '{config.LLM_MODEL}' is available ('ollama list').")
         return # Exit if Ollama not ready
    elif config.LLM_PROVIDER != "ollama":
        # Handle cases where provider is something else but client setup failed
        # This part is less likely needed now, but kept for robustness
        logging.error(f"LLM_PROVIDER is set to '{config.LLM_PROVIDER}' but the corresponding client in llm_extractor.py might not be configured.")
        return

    # Apply LLM extraction with progress bar
    tqdm.pandas(desc="LLM Extraction Progress")
    # Ensure 'summary' column exists and handle potential NaNs before applying
    if 'summary' not in df_processed.columns:
        logging.error("'summary' column not found in processed data. Cannot extract physicists.")
        return
    df_processed['summary'] = df_processed['summary'].fillna('') # Replace NaN with empty string
    df_processed['assigned_physicists'] = df_processed['summary'].progress_apply(llm_extractor.extract_physicists_llm)

    # 4. Explode DataFrame
    logging.info("Exploding DataFrame by assigned physicist...")
    df_exploded = data_processor.explode_by_physicist(df_processed)

    # 5. Analyze Workload
    logging.info("Analyzing workload...")
    workload_summary = analysis.analyze_workload(df_exploded)

    # 6. Output Results
    if workload_summary is not None and not workload_summary.empty:
        print("\n--- Physicist Workload Summary ---")
        # Use to_string to prevent truncation in console output
        print(workload_summary.to_string(index=False))
    else:
        logging.warning("Workload analysis did not produce results (perhaps no known physicists were assigned).")

    # Report on 'Unknown' assignments
    analysis.report_unknown_assignments(df_exploded)

    # 7. Visualize Results
    logging.info("Generating visualizations...")
    if workload_summary is not None and not workload_summary.empty:
        # Define a base filename for saving plots
        plot_save_path = "workload_analysis_ollama"
        visualization.plot_workload_duration(workload_summary, save_path=plot_save_path)
        visualization.plot_workload_events(workload_summary, save_path=plot_save_path)
        logging.info(f"Visualizations generated (or shown). Check for '{plot_save_path}_*.png' files.")
    else:
        logging.warning("Skipping visualization generation due to empty or non-existent workload summary.")

    logging.info("Pipeline finished.")

if __name__ == "__main__":
    # No need for dotenv loading for local Ollama unless server requires auth tokens passed via env vars
    run_pipeline()