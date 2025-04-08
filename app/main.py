# main.py
"""
Main script to run the calendar workload analysis pipeline using Ollama
with parallel LLM extraction.
"""
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed # Import necessary components

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

    # --- 3. Extract Physicists using LLM (PARALLELIZED) ---
    logging.info(f"Extracting physicists using LLM ({config.LLM_PROVIDER} - {config.LLM_MODEL} at {config.OLLAMA_BASE_URL}) with {config.LLM_MAX_WORKERS} workers...")

    # Check LLM client readiness (same as before)
    if config.LLM_PROVIDER == "ollama" and llm_extractor.llm_client is None:
         logging.error(f"Ollama Client not available or failed to connect to {config.OLLAMA_BASE_URL}. Cannot proceed.")
         print(f"\nPlease ensure the Ollama server is running at {config.OLLAMA_BASE_URL} and the model '{config.LLM_MODEL}' is available ('ollama list').")
         return
    elif config.LLM_PROVIDER != "ollama":
        logging.error(f"LLM_PROVIDER is set to '{config.LLM_PROVIDER}' but the corresponding client isn't configured.")
        return

    # Ensure 'summary' column exists and handle potential NaNs
    if 'summary' not in df_processed.columns:
        logging.error("'summary' column not found in processed data. Cannot extract physicists.")
        return
    df_processed['summary'] = df_processed['summary'].fillna('')

    # Get the summaries to process
    summaries = df_processed['summary'].tolist()
    results = [None] * len(summaries) # Pre-allocate results list

    # Use ThreadPoolExecutor for parallel processing
    # The `with` statement ensures threads are cleaned up properly
    logging.info(f"Submitting {len(summaries)} summaries to {config.LLM_MAX_WORKERS} worker threads...")
    try:
        with ThreadPoolExecutor(max_workers=config.LLM_MAX_WORKERS) as executor:
            # Create a dictionary to map future objects back to their original index
            future_to_index = {executor.submit(llm_extractor.extract_physicists_llm, summary): i for i, summary in enumerate(summaries)}

            # Process futures as they complete, using tqdm for progress
            for future in tqdm(as_completed(future_to_index), total=len(summaries), desc="LLM Extraction Progress"):
                index = future_to_index[future]
                try:
                    # Get the result from the future (this raises exceptions if the thread failed)
                    result = future.result()
                    results[index] = result
                except Exception as exc:
                    logging.error(f"Summary index {index} generated an exception: {exc}")
                    # Handle error - assign 'Unknown' or a specific error marker
                    results[index] = ["Unknown_Error"] # Or stick with ["Unknown"]

        # Assign the collected results back to the DataFrame
        df_processed['assigned_physicists'] = results
        logging.info("LLM extraction completed.")

    except Exception as e:
        logging.error(f"An error occurred during parallel LLM extraction: {e}")
        # Decide how to proceed: exit, or continue with potentially incomplete data?
        return # Exit for now

    # --- End of Parallelized Step 3 ---

    # 4. Explode DataFrame
    logging.info("Exploding DataFrame by assigned physicist...")
    # Ensure the explode function can handle the ['Unknown_Error'] case if you added it
    df_exploded = data_processor.explode_by_physicist(df_processed)

    # 5. Analyze Workload
    logging.info("Analyzing workload...")
    workload_summary = analysis.analyze_workload(df_exploded) # Make sure this filters Unknown_Error too if needed

    # 6. Output Results
    if workload_summary is not None and not workload_summary.empty:
        print("\n--- Physicist Workload Summary ---")
        print(workload_summary.to_string(index=False))
    else:
        logging.warning("Workload analysis did not produce results.")

    # Report on 'Unknown' assignments (May include Unknown_Error now)
    analysis.report_unknown_assignments(df_exploded)

    # 7. Visualize Results
    logging.info("Generating visualizations...")
    if workload_summary is not None and not workload_summary.empty:
        plot_save_path = "workload_analysis_ollama"
        visualization.plot_workload_duration(workload_summary, save_path=plot_save_path)
        visualization.plot_workload_events(workload_summary, save_path=plot_save_path)
        logging.info(f"Visualizations generated. Check for '{plot_save_path}_*.png' files.")
    else:
        logging.warning("Skipping visualization generation due to empty workload summary.")

    logging.info("Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()