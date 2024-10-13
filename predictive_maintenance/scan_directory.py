import os


def scan_directory(root_dir, output_file):
    exclude_dirs = {".venv", "maintenance_env",".git",".idea"}  # Directories to exclude

    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude the unwanted directories
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

            # Write the current directory to the file
            level = dirpath.replace(root_dir, "").count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")

            # Write the files in the current directory
            subindent = ' ' * 4 * (level + 1)
            for filename in filenames:
                f.write(f"{subindent}{filename}\n")


# Path to the root directory (adjust this path based on your system)
root_dir = 'C:/Users/neora/Desktop/Final_project/-maintenance-system'
output_file = "directory_structure.txt"

scan_directory(root_dir, output_file)
