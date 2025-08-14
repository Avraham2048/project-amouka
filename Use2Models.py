# usemodels.py
import subprocess
import sys
import os

# --- CONFIGURATION ---
OCR_SCRIPT_PATH = "useocr.py"
NIKUD_SCRIPT_PATH = "use_diac.py"
NIKUD_MODEL_PATH = "diac.pt"  
PYTHON_EXECUTABLE = sys.executable

def run_process(command, env_overrides=None):
    """
    Runs command (list). Captures stdout/stderr, decodes using utf-8.
    Exits the program on failure with helpful messages.
    """
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if env_overrides:
        env.update(env_overrides)

    # Show the command in a readable way (do not run via shell)
    #print(f"▶️  Running command: {' '.join(command)}")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,
            env=env
        )
        return completed.stdout.strip()
    except FileNotFoundError:
        print(f"❌ ERROR: File not found. Is the path '{command[0]}' correct?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Try to produce a readable script name for the error message
        script_name = os.path.basename(e.cmd[0]) if isinstance(e.cmd, (list, tuple)) else str(e.cmd)
        print(f"❌ ERROR: The script '{script_name}' failed with exit code {e.returncode}.")
        print("--- Script's Error Output ---")
        print(e.stderr)
        print("-----------------------------")
        sys.exit(1)

def main():
    if NIKUD_MODEL_PATH == "path/to/your/model.pt":
        print("❌ ERROR: Please update the NIKUD_MODEL_PATH variable in this script before running.")
        sys.exit(1)
    if not os.path.exists(NIKUD_MODEL_PATH):
        print(f"❌ ERROR: Nikud model not found at '{NIKUD_MODEL_PATH}'. Please check the path.")
        sys.exit(1)

    # 1. Run OCR Script
    print("\n--- Step 1: Starting OCR Process ---")
    ocr_command = [PYTHON_EXECUTABLE, OCR_SCRIPT_PATH]
    ocr_text = run_process(ocr_command)

    if not ocr_text:
        print("⚠️  Warning: OCR script ran successfully but produced no text output.")
        sys.exit(0)

    # show a short preview (avoid dumping huge text)
    preview = ocr_text if len(ocr_text) <= 200 else ocr_text[:200] + "…"
    print(f"✅ OCR Output Received: \"{preview}\"")

    # 2. Run Nikud Script with OCR text as a positional argument
    #    This matches: python use_diac.py diac.pt "שלום עולם"
    print("\n--- Step 2: Starting Nikud (Diacritization) Process ---")
    nikud_command = [PYTHON_EXECUTABLE, NIKUD_SCRIPT_PATH, NIKUD_MODEL_PATH, ocr_text]

    final_text = run_process(nikud_command)

    # 3. Display Final Result
    print("\n" + "="*40)
    print("          ✨ Final Result ✨")
    print("="*40)
    print(f"Text from OCR : {ocr_text}")
    print(f"Punctuated Text: {final_text}")
    print("="*40)
    print("\n✨ Process finished successfully.")

if __name__ == "__main__":
    main()
