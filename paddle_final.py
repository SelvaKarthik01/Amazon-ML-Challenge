import os
import requests
import pandas as pd
import numpy as np
import signal
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR
from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to your CSV file
csv_file_path = r'train.csv'

# Output folder for the OCR processed images and CSV
output_folder = r'paddle_ocr_output'
os.makedirs(output_folder, exist_ok=True)
# Read the CSV (limit to 100 rows for this example)
df = pd.read_csv(csv_file_path).head(1000)
processed_data = []
chunk_size = 50  # Save the CSV after every 5 rows
chunk_counter = 0
def process_row(row):
    img_url = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name']
    entity_value = row['entity_value']

    ocr_result_text = ""

    try:
        # Download the image from the URL
        response = requests.get(img_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Convert the image to a NumPy array
            img_np = np.array(img)

            # Perform OCR on the image
            result = ocr.ocr(img_np, cls=True)

            if result is not None:
                # Concatenate all detected texts with ";" delimiter
                ocr_result_text = "; ".join([line[1][0] for res in result for line in res])
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")

    # Return the processed row as a dictionary
    return {
        'group_id': group_id,
        'text': ocr_result_text.strip(),  # Clean up trailing spaces
        'entity_name': entity_name,
        'entity_value': entity_value
    }
def save_chunk(processed_data, chunk_counter):
    if processed_data:
        output_csv_path = os.path.join(output_folder, f'ocr_results_chunk_{chunk_counter}.csv')
        pd.DataFrame(processed_data).to_csv(output_csv_path, index=False)
        print(f"Chunk saved to {output_csv_path}")
        processed_data.clear()  # Clear the buffer after saving
        chunk_counter += 1
    return chunk_counter
try:
    for idx, row in df.iterrows():
        processed_data.append(process_row(row))

        # Save if we have enough rows
        if len(processed_data) >= chunk_size:
            chunk_counter = save_chunk(processed_data, chunk_counter)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected! Saving current progress...")

finally:
    # Save any remaining rows in the buffer if present
    if processed_data:
        chunk_counter = save_chunk(processed_data, chunk_counter)

    print("OCR processing complete. All data saved.")