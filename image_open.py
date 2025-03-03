
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv('student_resource3/dataset/train.csv')

df = df.head(50)
# Empty list to store results
results = []

def get_image_from_url(url):
    """Fetch the image from the provided URL and return it as a numpy array."""
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

def add_text_to_image(image, entity_name, entity_value):
    """Add text annotations to the image."""
    image_with_text = image.copy()
    # Set text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (255, 255, 255)  # White color
    thickness = 3
    line_type = cv2.LINE_AA

    # Text to add
    text = f"{entity_name}: {entity_value}"
    
    # Calculate text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    text_x = max(10, image.shape[1] - text_width - 10)
    text_y = image.shape[0] - 10

    # Add text to image
    cv2.putText(image_with_text, text, (text_x, text_y), font, font_scale, color, thickness, line_type)

    return image_with_text

def save_results():
    """Save the current results to a CSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv('image_classifications.csv', index=False)
    print("Progress saved to 'image_classifications.csv'.")

def on_key(event, row):
    """Handle key press events."""
    global current_index, results

    # Mapping key presses to classification results
    if event.key == '0':
        result = 'False'
    elif event.key == '1':
        result = 'True'
    elif event.key == '2':
        result = 'Unknown'
    elif event.key == 'q':
        # If 'q' is pressed, save and quit
        print("Exit requested. Saving progress...")
        save_results()
        plt.close('all')  # Close all open figures
        exit(0)  # Terminate the program
    else:
        print("Invalid key pressed. Please press '0', '1', '2', or 'q'.")
        return  # Ignore invalid key presses

    # Append result to the list if a valid key is pressed
    if event.key in ['0', '1', '2']:
        results.append({
            'image_link': row['image_link'],
            'entity_name': row['entity_name'],
            'entity_value': row['entity_value'],
            'classification': result
        })

        # Close the current figure to move to the next image
        plt.close()

        # Increment the current index
        current_index += 1

        # Show the next image if available
        if current_index < len(df):
            display_next_image()

def display_next_image():
    """Display the next image in the list."""
    row = df.iloc[current_index]

    # Fetch the image
    image = get_image_from_url(row['image_link'])
    if image is not None:
        # Add text to image
        image_with_text = add_text_to_image(image, row['entity_name'], row['entity_value'])

        # Display image with title
        fig, ax = plt.subplots()
        ax.imshow(image_with_text)
        plt.title(f"Group ID: {row['group_id']},entity_name: {row['entity_name']},entity_value : {row['entity_value']}")
        plt.axis('off')  # Hide axis

        # Set the key press event handler
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, row))

        # Show the image and wait for a key press
        plt.show()

# Initialize the current index
current_index = 0

# Show the first image
display_next_image()
