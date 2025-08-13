import os
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

def create_dataset_from_folders(root_dir, output_path):
    """
    Scans folders of images, labels them based on the folder name,
    and saves the dataset as a .parquet file.

    Args:
        root_dir (str): The path to the directory containing numbered subfolders of images
                        (e.g., 'dataocr/train').
        output_path (str): The path where the output .parquet file will be saved
                           (e.g., 'dataocr/train.parquet').
    """
    # --- Hebrew Alphabet Mapping (for verification and clarity) ---
    # 27 characters including final forms (sofiot)
    # The index of the letter in this list corresponds to the folder name.
    hebrew_letters = [
        'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ך', 'ל', 'מ', 'ם', 'נ', 'ן',
        'ס', 'ע', 'פ', 'ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת', ','
    ]

    image_data = []

    # Get a sorted list of directories (0, 1, 2, ...) to ensure consistent ordering.
    try:
        # We assume folder names are numbers, so we sort them numerically.
        dir_list = sorted(os.listdir(root_dir), key=int)
    except ValueError:
        # Fallback for non-numeric folder names, though the setup requires numbers.
        dir_list = sorted(os.listdir(root_dir))
        print("Warning: Could not sort folder names numerically. Using alphabetical sort.")


    print(f"Found {len(dir_list)} directories. Processing...")

    # Use tqdm for a nice progress bar
    for label_str in tqdm(dir_list, desc="Processing Folders"):
        folder_path = os.path.join(root_dir, label_str)

        # Ensure it's actually a directory
        if not os.path.isdir(folder_path):
            continue

        try:
            # The label is the integer value of the folder name
            label = int(label_str)
            
            # Optional: Print the letter we're currently processing
            if label < len(hebrew_letters):
                print(f"\nProcessing folder '{label}' which corresponds to letter: '{hebrew_letters[label]}'")
            else:
                print(f"\nProcessing folder '{label}' (No corresponding letter in map)")


        except ValueError:
            print(f"Skipping non-numeric directory: {label_str}")
            continue

        # Loop through all files in the current folder
        for filename in os.listdir(folder_path):
            # Process only image files (e.g., .png, .jpg)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, filename)

                try:
                    # Open the image using Pillow
                    with Image.open(file_path) as img:
                        # Convert the image to bytes. Your OCR script expects the raw bytes.
                        # We'll save it in memory first.
                        with io.BytesIO() as byte_io:
                            # Save the image to the in-memory byte stream in PNG format
                            img.save(byte_io, format='PNG')
                            # Get the byte value
                            image_bytes = byte_io.getvalue()

                        # Append the data as a dictionary to our list.
                        # This matches the structure your ocr.py script expects.
                        image_data.append({
                            'image': {'bytes': image_bytes},
                            'label': label
                        })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if not image_data:
        print("No image data was processed. Please check the `root_dir` path and folder structure.")
        return

    # --- Create and Save the DataFrame ---
    print("\nConverting data to pandas DataFrame...")
    df = pd.DataFrame(image_data)

    # Shuffle the dataset to ensure randomness when training
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"DataFrame created with {len(df)} samples.")
    print("DataFrame Info:")
    df.info()
    print("\nLabel Distribution:")
    print(df['label'].value_counts().sort_index())

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSaving DataFrame to .parquet file at: {output_path}")
    df.to_parquet(output_path)
    print("Successfully created the .parquet file!")


if __name__ == '__main__':
    # --- Configuration ---
    # Set the path to your training images folder
    TRAIN_DATA_DIR = 'dataocr/train'
    # Set the desired output path for the parquet file
    OUTPUT_PARQUET_PATH = 'dataocr/train2.parquet'

    # Run the function
    create_dataset_from_folders(TRAIN_DATA_DIR, OUTPUT_PARQUET_PATH)

    # You can uncomment this section to process a validation set as well
    # print("\n--- Processing Validation Data ---")
    # VAL_DATA_DIR = 'dataocr/val'
    # VAL_OUTPUT_PARQUET_PATH = 'dataocr/val.parquet'
    # create_dataset_from_folders(VAL_DATA_DIR, VAL_OUTPUT_PARQUET_PATH)


