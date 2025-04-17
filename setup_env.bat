@echo off
set VENV_DIR=.venv\SPC
set REQUIREMENTS_FILE=requirements.txt
set MCP_REQUIREMENTS_FILE=MCP\requirements.txt
set PYTHON_EXE=python

echo Checking for virtual environment in %VENV_DIR%...

REM Check if the python executable exists within the venv dir as a proxy for existence
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment not found. Creating...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Exiting.
        exit /b %errorlevel%
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment found.
)

echo Installing/updating requirements from %REQUIREMENTS_FILE%...

REM Use the pip from the virtual environment
"%VENV_DIR%\Scripts\pip" install -r %REQUIREMENTS_FILE%
if %errorlevel% neq 0 (
    echo Failed to install requirements. Exiting.
    exit /b %errorlevel%
)

echo Installing MCP server requirements...
"%VENV_DIR%\Scripts\pip" install -r %MCP_REQUIREMENTS_FILE%
if %errorlevel% neq 0 (
    echo Failed to install MCP requirements, but continuing...
)

echo Upgrading pip in the virtual environment...
"%VENV_DIR%\Scripts\python" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip, but continuing...
)

echo Environment setup complete.
echo You can activate the environment manually using: %VENV_DIR%\Scripts\activate
