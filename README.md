# Calendar Workload Analyzer

A Streamlit application designed to analyze calendar data to understand workload distribution across teams, particularly in medical/clinical settings.

## Project Overview

This application helps teams analyze their calendar data to better understand workload distribution, identify scheduling patterns, and optimize resource allocation. The system uses natural language processing (via LLMs) to extract personnel mentions from calendar event summaries and provides various visualizations and metrics for workload analysis.

## Key Features

- **Calendar Data Upload**: Upload JSON-formatted calendar data for analysis
- **Smart Personnel Extraction**: Uses LLM technology to identify personnel from event descriptions
- **Personnel Configuration Management**: Define team members and their variations (aliases, nicknames)
- **Comprehensive Analytics**: View workload distribution by person, role, time period
- **Interactive Visualizations**: Analyze patterns with charts and heatmaps
- **Database Integration**: Optionally store and retrieve data from PostgreSQL (configurable)

## Project Structure

```
calendar-analysis/
│
├── app/                        # Streamlit application
│   ├── streamlit_app.py        # Main entry point
│   └── pages/                  # Streamlit multipage app structure
│       ├── 1_📁_Upload_Process.py  # Data upload and processing
│       ├── 2_📊_Analysis.py        # Data visualization and analysis
│       ├── 3_⚙️_Admin.py           # Personnel configuration management
│       └── 4_🔍_Database_Viewer.py # Database exploration (when enabled)
│
├── config/                     # Configuration files
│   ├── settings.py             # Static application settings
│   └── personnel_config.json   # User-editable personnel configuration
│
├── data/                       # Example data directory
│   └── calendar.json           # Sample calendar data
│
├── functions/                  # Core functionality modules
│   ├── analysis_calculations.py  # Data aggregation and metrics
│   ├── config_manager.py         # Configuration management
│   ├── data_processor.py         # Data preprocessing functions
│   ├── db_manager.py             # Database operations
│   ├── llm_extractor.py          # Personnel extraction wrapper
│   ├── visualization_plotly.py   # Chart generation with Plotly
│   └── llm_extraction/           # LLM integration components
│       ├── extractor.py          # Core extraction logic
│       ├── normalizer.py         # Name normalization logic
│       ├── ollama_client.py      # Ollama LLM client
│       └── smart_router.py       # Batch processing router
│
├── services/                   # External services
│   └── mcp/                    # Model Context Protocol server
│       ├── server.py           # MCP server implementation
│       └── requirements.txt    # MCP server dependencies
│
├── output/                     # Application outputs (logs, exports)
│   └── calendar_analysis.log   # Application log
│
├── tests/                      # Unit tests
│   ├── test_analysis_calculations.py
│   └── test_data_processor.py
│
├── utils/                      # Utility scripts
│   ├── scripts/
│   │   └── code-concatenator.py # Code compilation utility for LLM integration
│   └── outline.md              # Project structure documentation
│
├── environment.yml             # Conda environment specification
└── requirements.txt            # Python dependencies
```

## Setup and Installation

1. **Environment Setup**:
   ```
   # Option 1: With Conda
   conda env create -f environment.yml
   conda activate calendar-analysis

   # Option 2: With pip
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Edit `config/settings.py` to configure LLM provider, database settings, etc.
   - Use the Admin page to manage personnel configurations

3. **Database Setup** (Optional):
   - Install and configure PostgreSQL if you want to use database features
   - The application will work without a database, storing data in session state

4. **LLM Setup**:
   - Make sure Ollama is running at the specified URL in settings
   - Default model is "mistral:latest" but can be configured in settings or environment variables

## Running the Application

Run the Streamlit application:

```bash
cd app
streamlit run streamlit_app.py
```

## Usage Workflow

1. **Upload Data**: Navigate to the Upload & Process page to upload your calendar data (JSON format)
2. **Configure Personnel**: Use the Admin page to set up your team members and their aliases
3. **Analyze Data**: Go to the Analysis page to explore workload distribution and patterns
4. **Export Results**: Download analysis results in various formats (CSV, Excel, JSON)

## LLM Integration

The application uses Large Language Models to:
- Extract personnel mentions from calendar event summaries
- Handle variations in naming (nicknames, abbreviations, etc.)

By default, it uses Ollama with the Mistral model, but this can be configured.

## Dependencies

- Streamlit: Web application framework
- Pandas: Data manipulation
- Plotly: Interactive visualizations
- Ollama: LLM integration
- PostgreSQL (optional): Database persistence

## Troubleshooting

- If LLM extraction is not working, ensure Ollama is running and accessible
- Check the logs in the `output` directory for detailed error information
- Use the Database Viewer to check if data is being stored correctly (when enabled)

## Contributing

Contributions are welcome! Please ensure your code passes the test suite before submitting pull requests.

## License

This project is licensed under the MIT License.
