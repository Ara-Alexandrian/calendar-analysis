CALENDAR-ANALYSIS/

├── config/                     # NEW: For configuration files

│   ├── settings.py             # Static settings (LLM URL, paths, etc.)

│   └── personnel_config.json   # DYNAMIC personnel list (editable by Admin)

├── data/

│   └── calendar.json           # Example input data

├── functions/

│   ├── __init__.py

│   ├── analysis_calculations.py # NEW: Functions for aggregation/metrics

│   ├── config_manager.py       # NEW: Load/save personnel_config.json

│   ├── data_processor.py       # MODIFIED: Handle BytesIO input

│   ├── llm_extractor.py        # MODIFIED: Streamlit integration, normalization

│   └── visualization_plotly.py # NEW: Plotly-based visualizations

├── pages/                      # Streamlit pages

│   ├── 1_📁_Upload_&_Process.py

│   ├── 2_📊_Analysis.py

│   └── 3_⚙️_Admin.py

├── output/                     # Logs, exported data (optional)

│   └── calendar_analysis.log

├── streamlit_app.py            # Main Streamlit entry point

├── requirements.txt            # Updated dependencies

└── README.md                   # Updated instructions
