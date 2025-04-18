# Calendar Workload Analyzer

A Streamlit application designed to analyze calendar data to understand workload distribution across teams, particularly in medical/clinical settings.

## Project Overview

This application helps teams analyze their calendar data (`.json` format) to better understand workload distribution, identify scheduling patterns, and optimize resource allocation. The system uses natural language processing (via LLMs, defaulting to Ollama) to extract personnel mentions and event types from calendar event summaries and provides various visualizations and metrics for workload analysis.

## Key Features

- **Calendar Data Upload**: Upload JSON-formatted calendar data for analysis.
- **Smart Personnel & Event Type Extraction**: Uses LLM technology to identify personnel and classify event types from event descriptions.
- **Personnel Configuration Management**: Define team members and their variations (aliases, nicknames) via the UI or JSON file.
- **Comprehensive Analytics**: View workload distribution by person, role, event type, and time period.
- **Interactive Visualizations**: Analyze patterns with charts and heatmaps (Plotly).
- **Database Integration**: Optionally store and retrieve processed data and file metadata from PostgreSQL (configurable).

## Project Structure

```
calendar-analysis/
│
├── app/                        # Streamlit application
│   ├── streamlit_app.py        # Main entry point
│   └── pages/                  # Streamlit multipage app structure
│       ├── 1_📁_Upload_Process.py  # Data upload and processing pipeline (incl. LLM calls)
│       ├── 2_📊_Analysis.py        # Data visualization and analysis dashboard
│       ├── 3_⚙️_Admin.py           # Personnel configuration management UI
│       └── 4_🔍_Database_Viewer.py # Database exploration (when DB enabled)
│
├── config/                     # Configuration files
│   ├── settings.py             # Static application settings (DB, LLM provider, etc.)
│   └── personnel_config.json   # User-editable personnel configuration
│
├── data/                       # Example data directory
│   └── calendar.json           # Sample calendar data
│
├── functions/                  # Core functionality modules
│   ├── analysis_calculations.py  # Data aggregation and metrics for analysis page
│   ├── config_manager.py         # Loading/saving personnel configuration
│   ├── data_processor.py         # Data loading, preprocessing, exploding functions
│   ├── db_manager.py             # Database operations (PostgreSQL)
│   ├── initialize_db.py          # (Potentially for initial DB setup - check usage)
│   ├── llm_extractor.py          # (May contain older/alternative extraction wrappers - check usage)
│   ├── visualization_plotly.py   # Chart generation functions using Plotly
│   └── llm_extraction/           # LLM integration components
│       ├── client.py             # LLM readiness checks
│       ├── direct_parallel_processor.py # (Older? Optimized for dual RTX 3090s - check usage)
│       ├── extractor.py          # Core extraction logic (_extract_single_physicist_llm)
│       ├── normalizer.py         # Name and event type normalization logic
│       ├── ollama_client.py      # Ollama LLM client setup
│       └── utils.py              # Shared utilities for LLM processing
│
├── services/                   # External services (Placeholder/Optional)
│   └── mcp/                    # Model Context Protocol server (If used)
│
├── output/                     # Application outputs (logs, exports)
│   └── calendar_analysis.log   # Application log file
│
├── tests/                      # Unit and integration tests
│   ├── test_analysis_calculations.py
│   ├── test_data_processor.py
│   └── llm_tests/              # Tests specific to LLM functionality
│
├── utils/                      # Utility scripts and documentation aids
│   ├── scripts/
│   │   └── code-concatenator.py # (Likely for bundling code for LLM context)
│   └── outline.md              # Project structure documentation aid
│
├── .gitignore                  # Git ignore file
├── README.md                   # This file
├── requirements.txt            # Python dependencies (pip)
├── run_app.py                  # Convenience script to run the Streamlit app
├── run_tests.py                # Convenience script to run tests
└── setup_env.bat               # Windows batch script for environment setup (conda)
```

## Setup and Installation

1.  **Environment Setup**:
    *   Ensure you have Python installed (>= 3.9 recommended).
    *   **Option 1: With Conda/Mamba** (Recommended)
        ```bash
        # Create environment (using setup_env.bat on Windows or manually)
        conda create --name calendar-analysis python=3.9 -y
        conda activate calendar-analysis
        pip install -r requirements.txt
        ```
    *   **Option 2: With pip/venv**
        ```bash
        python -m venv venv
        # Activate venv (e.g., source venv/bin/activate on Linux/macOS, venv\Scripts\activate on Windows)
        pip install -r requirements.txt
        ```

2.  **Configuration**:
    *   Review and edit `config/settings.py` to configure the LLM provider (default: Ollama), model name, base URL, database connection details (if using), etc.
    *   Use the Admin page (`3_⚙️_Admin.py`) in the running application to manage personnel configurations, or edit `config/personnel_config.json` directly.

3.  **Database Setup** (Optional):
    *   If `DB_ENABLED` is `True` in `config/settings.py`, ensure you have a running PostgreSQL server accessible with the credentials provided in settings.
    *   The application will attempt to create necessary tables automatically on first run or when needed.

4.  **LLM Setup** (Required for core functionality):
    *   Ensure your chosen LLM provider (e.g., Ollama) is running and accessible at the URL specified in `config/settings.py`.
    *   If using Ollama, make sure the model specified in `settings.OLLAMA_MODEL` (default: "mistral:latest") is downloaded (`ollama pull mistral:latest`).
    *   The `ollama` Python library is required (`pip install ollama`).

## Running the Application

Use the provided convenience script:

```bash
python run_app.py
```

Or run directly with Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## Usage Workflow

1.  **Configure Personnel**: Navigate to the Admin page (`3_⚙️_Admin.py`) to set up your team members and their name variations (aliases, nicknames). This is crucial for accurate LLM extraction.
2.  **Upload Data**: Go to the Upload & Process page (`1_📁_Upload_Process.py`) to upload your calendar data (`.json` format).
3.  **Process Data**: Click the "Process Data" button. The pipeline will:
    *   Preprocess events (dates, durations).
    *   Call the LLM to extract personnel and event types for each event summary.
    *   Normalize the extracted data.
    *   Prepare the data for analysis.
    *   Optionally save results to the database.
4.  **Analyze Data**: Navigate to the Analysis page (`2_📊_Analysis.py`) to explore workload distribution, filter data, and view visualizations.
5.  **View Database** (Optional): Use the Database Viewer page (`4_🔍_Database_Viewer.py`) to inspect raw data stored in PostgreSQL.
6.  **Export Results**: Download analysis results in various formats (CSV, Excel, JSON) from the Analysis page.

## LLM Integration Details

-   **Extraction**: The core logic resides in `_extract_single_physicist_llm` within `functions/llm_extraction/extractor.py`. This function is called iteratively for each event summary during the processing step on the Upload page (`app/pages/1_📁_Upload_Process.py`).
-   **Normalization**: After extraction, `normalize_extracted_personnel` and `normalize_event_type` from `functions/llm_extraction/normalizer.py` are used to map extracted names/types to canonical forms defined in the personnel config and settings.
-   **Configuration**: LLM provider, model, and endpoint are configured in `config/settings.py`.
-   **Event Types**: The LLM also attempts to classify the type of event (e.g., Meeting, Clinical Duty, Admin). The mapping from raw LLM output to standardized types is defined in `EVENT_TYPE_MAPPING` within `config/settings.py`.
-   **Recent Fix**: Resolved an issue where the extracted event type column was inconsistently named (`event_type` vs. `extracted_event_type`) between processing and analysis/database steps. The correct name `extracted_event_type` is now used consistently.

## Dependencies

-   Streamlit: Web application framework
-   Pandas: Data manipulation and analysis
-   Plotly: Interactive visualizations
-   Ollama: LLM integration client library
-   Psycopg2: PostgreSQL database adapter (required if DB_ENABLED=True)

See `requirements.txt` for specific versions.

## Troubleshooting

-   **LLM Errors**: Ensure the LLM service (Ollama) is running, accessible at the configured URL, and the specified model is available. Check the Streamlit console and `output/calendar_analysis.log` for detailed errors. If the `ollama` library is missing, install it (`pip install ollama`).
-   **Database Errors**: Verify PostgreSQL server status and connection details in `config/settings.py`. Ensure the database user has permissions to create tables and read/write data.
-   **Personnel/Event Type Issues**: Check the personnel configuration on the Admin page. Ensure the `EVENT_TYPE_MAPPING` in `config/settings.py` covers expected outputs from the LLM. Examine the `extracted_event_type` column in the raw data view on the Analysis page or in the database to see what the LLM is actually extracting.
-   **Missing Data/Columns**: If analysis fails due to missing columns (like the previous `extracted_event_type` issue), review the processing steps in `app/pages/1_📁_Upload_Process.py` and the data loading in `app/pages/2_📊_Analysis.py`.

## LLM Context Summary (For Future Sessions)

-   **Project Goal:** Analyze workload distribution from calendar `.json` files using a Streamlit app.
-   **Core Workflow:** Upload -> Preprocess -> LLM Extract (Personnel & Event Type) -> Normalize -> Explode -> Analyze/Visualize. Optional DB persistence (PostgreSQL).
-   **Key Files:**
    -   `app/pages/1_📁_Upload_Process.py`: Upload UI & main processing pipeline driver. Calls LLM iteratively.
    -   `app/pages/2_📊_Analysis.py`: Analysis dashboard, filtering, visualizations. Loads processed data.
    -   `functions/llm_extraction/extractor.py`: Contains `_extract_single_physicist_llm` (core extraction logic).
    -   `functions/llm_extraction/normalizer.py`: `normalize_extracted_personnel`, `normalize_event_type`.
    -   `functions/data_processor.py`: `load_raw_calendar_data`, `preprocess_data`, `explode_by_personnel`.
    -   `functions/db_manager.py`: PostgreSQL interactions (saving/loading processed data, file metadata).
    -   `config/settings.py`: LLM endpoint, DB connection, event type mapping (`EVENT_TYPE_MAPPING`).
    -   `config/personnel_config.json`: Canonical names, roles, variations for personnel.
-   **Recent Activity (as of ~2025-04-16 20:25 CST):**
    -   Fixed bug: Renamed internal `event_type` column to `extracted_event_type` in `app/pages/1_📁_Upload_Process.py` for consistency with database schema and analysis page expectations.
-   **Current State:** The app should correctly process calendar data, extract personnel and event types using the configured LLM, allow analysis via the UI, and optionally persist data to PostgreSQL.

## Contributing

Contributions are welcome! Please ensure your code passes the test suite (if applicable) and follows existing style conventions before submitting pull requests.

## License

This project is licensed under the MIT License.
