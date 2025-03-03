import os
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from paddleocr import PaddleOCR

# Initialize PaddleOCR (you only need to do this once)
ocr = PaddleOCR(use_angle_cls=True, lang='en') 

# Path to your CSV file
csv_file_path = 'student_resource3/dataset/train.csv'

# Output folder for the OCR processed images and CSV
output_folder = 'paddle_ocr_output'
os.makedirs(output_folder, exist_ok=True)

# Read the CSV (limit to 100 rows for this example)
df = pd.read_csv(csv_file_path).head(100)
print(df)

# List to store OCR results for the CSV
ocr_data = []

# Iterate over each row in the DataFrame
for idx, row in df.iterrows():
    img_url = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name']
    entity_value = row['entity_value']
    
    try:
        # Download the image from the URL
        response = requests.get(img_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Convert the image to a NumPy array
            img_np = np.array(img)

            # Perform OCR on the NumPy array
            result = ocr.ocr(img_np, cls=True)

            if result is not None:
                # Open the image for drawing OCR results
                # draw = ImageDraw.Draw(img)

                # Iterate over the OCR result and draw boxes with text
                for res in result:
                    if res is not None:
                        for line in res:
                            # Get bounding box, text, and score
                            box = line[0]
                            text = line[1][0]
                            score = line[1][1]

                            # Flatten the box coordinates into a single list
                            box = [tuple(coord) for coord in box]

                            # Draw the bounding box (use line for drawing the polygon)
                            # draw.line(box + [box[0]], fill='red', width=2)

                            # Draw the text along with the confidence score
                            # draw.text((box[0][0], box[0][1] - 10), f'{text} ({score:.2f})', fill='red')

                            # Save OCR results in the list
                            ocr_data.append({
                                'filename': group_id,
                                'text': text,
                                'score': score,
                                'box': box,
                                'entity_name': entity_name,
                                'entity_value': entity_value
                            })

                # Save only the OCR-processed image with bounding boxes and labels
                # output_image_path = os.path.join(output_folder, f'{group_id}_paddleocr.jpg')
                # img.save(output_image_path)

    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        continue  # Skip the current image and move to the next one

# Save the OCR results to a CSV file
ocr_df = pd.DataFrame(ocr_data)
output_csv_path = os.path.join(output_folder, 'ocr_results.csv')
ocr_df.to_csv(output_csv_path, index=False)

print(f"OCR results saved to {output_csv_path}")
