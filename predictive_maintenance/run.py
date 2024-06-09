import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(f"{script_name} completed successfully.")
        print(result.stdout)

if __name__ == "__main__":
    scripts = [
        'src/data_preprocessing.py',
        'src/model_training.py',
        'src/real_time_processing.py',
        'src/visualization.py'
    ]

    for script in scripts:
        run_script(script)
