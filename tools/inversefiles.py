import os

def reverse_names_in_folder(folder_path):
    # Walk through the folder structure from deepest first
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Rename files (keeping extension in place)
        for filename in files:
            old_path = os.path.join(root, filename)
            name, ext = os.path.splitext(filename)
            new_filename = name[::-1] + ext
            new_path = os.path.join(root, new_filename)
            os.rename(old_path, new_path)

        # Rename folders (whole name reversed)
        for dirname in dirs:
            old_path = os.path.join(root, dirname)
            new_dirname = dirname[::-1]
            new_path = os.path.join(root, new_dirname)
            os.rename(old_path, new_path)

if __name__ == "__main__":
    folder = input("Enter the full path of the folder: ").strip()
    if os.path.isdir(folder):
        reverse_names_in_folder(folder)
        print("✅ All file and folder names reversed (extensions preserved).")
    else:
        print("❌ The provided path is not a valid folder.")
