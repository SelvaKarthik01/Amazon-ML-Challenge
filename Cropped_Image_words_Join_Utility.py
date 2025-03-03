import pandas as pd

# Load the CSV file
csv_file = "predicted_words.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)

# Create a dictionary to store image_name and combined predicted words
image_text_dict = {}

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    image_name = row['image_name']
    predicted_word = str(row['predicted_word'])
    
    # Extract the base image name (remove _1, _2, etc.)
    base_image_name = image_name.rsplit('_', 1)[0]
    
    # Append the predicted word to the corresponding base image
    if base_image_name in image_text_dict:
        image_text_dict[base_image_name] += ' ' + predicted_word
    else:
        image_text_dict[base_image_name] = predicted_word

# Print the results and prepare them for the new CSV
output_data = []
for image_name, combined_text in image_text_dict.items():
    print(f"Image: {image_name}.jpg")
    print(f"Combined Text: {combined_text}\n")
    output_data.append([f"{image_name}.jpg", combined_text])

# Save the results to a new CSV file
output_df = pd.DataFrame(output_data, columns=["image_name", "combined_text"])
output_file = "combined_predicted_words.csv"  # Path to save the new CSV file
output_df.to_csv(output_file, index=False)

print(f"Combined words saved to {output_file}")
