import os
import sys

# --- Configuration ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes script is saved in project root
OUTPUT_FILENAME = "combined_project_output.txt" # File to save the output

# Directories to exclude completely
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".vscode",
    ".idea",
    "venv", 
    "env", 
    ".env", 
    "output", 
}

# File extensions or specific filenames to exclude
EXCLUDE_FILES = {
    ".pyc",
    ".DS_Store",
    OUTPUT_FILENAME, # Don't include the output file itself
}

# Extensions for files to sample (first 100 lines)
SAMPLE_EXTENSIONS = {
    ".json",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".csv",
    ".env", 
    ".ics", 
}

MAX_SAMPLE_LINES = 100
# --- End Configuration ---

def combine_files(root_dir):
    """Walks through the directory and combines file contents."""
    combined_content = []
    processed_files = set()

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()
            if filename in EXCLUDE_FILES or file_ext in EXCLUDE_FILES:
                continue

            full_path = os.path.join(dirpath, filename)
            absolute_path = os.path.abspath(full_path)
            
            if absolute_path in processed_files:
                continue
            processed_files.add(absolute_path)

            relative_path = os.path.relpath(full_path, root_dir).replace(os.sep, '/')
            
            header = f"--- START OF FILE: {relative_path} ---\n"
            footer = f"--- END OF FILE: {relative_path} ---\n\n"
            content = ""
            is_sample = False

            try:
                if file_ext == ".py":
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                elif file_ext in SAMPLE_EXTENSIONS:
                    is_sample = True
                    lines = []
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if i >= MAX_SAMPLE_LINES:
                                break
                            lines.append(line)
                    content = "".join(lines)
                    if len(lines) >= MAX_SAMPLE_LINES:
                         header = f"--- START OF FILE: {relative_path} (First {MAX_SAMPLE_LINES} lines) ---\n"
                else:
                    continue 
                    
                combined_content.append(header)
                combined_content.append(content)
                combined_content.append(footer)

            except FileNotFoundError:
                 combined_content.append(f"--- ERROR: Could not find file {relative_path} during processing ---\n\n")
            except Exception as e:
                 combined_content.append(f"--- ERROR reading file {relative_path}: {e} ---\n\n")

    return "".join(combined_content)

if __name__ == "__main__":
    print(f"Starting file combination process in directory: {PROJECT_ROOT_DIR}")
    print(f"Excluding Dirs: {EXCLUDE_DIRS}")
    print(f"Excluding Files/Exts: {EXCLUDE_FILES}")
    print(f"Sampling Exts (up to {MAX_SAMPLE_LINES} lines): {SAMPLE_EXTENSIONS}")
    print("-" * 30)
    
    full_output = combine_files(PROJECT_ROOT_DIR)
    
    # --- Save to File ---
    try:
        output_file_path = os.path.join(PROJECT_ROOT_DIR, OUTPUT_FILENAME)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"\nSuccessfully combined content saved to: {output_file_path}") # Confirmation message
    except Exception as e:
        print(f"\nError saving combined content to file '{OUTPUT_FILENAME}': {e}", file=sys.stderr)
    # --- End Save to File ---

    # --- Console printing commented out ---
    # print("\n*** Combined Project Content Start ***\n")
    # print(full_output)
    # print("\n*** Combined Project Content End ***\n")
    # --- End Console printing ---

    print("Process finished.")
    print(f"\n>>> IMPORTANT: Please carefully REVIEW the generated file ('{OUTPUT_FILENAME}') for any sensitive information (API keys, passwords, personal data in logs/samples) before sharing! <<<")