import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('DR_Master_Dataset/master_metadata.csv')

# 1. Create the splits
# Stratify ensures the distribution of 'dr_grade' is the same in all sets
frontend_df, model_df = train_test_split(
    df, test_size=0.99, stratify=df['dr_grade'], random_state=42
)

train_df, val_df = train_test_split(
    model_df, test_size=0.20, stratify=model_df['dr_grade'], random_state=42
)

def move_and_save_metadata(dataframe, folder_name):
    # Path for the images
    dest_path = f'drDataset/{folder_name}'
    os.makedirs(dest_path, exist_ok=True)
    
    # --- NEW: Save the CSV for this specific split ---
    # This saves ALL columns (image_id, dr_grade, etc.) present in the original metadata
    csv_path = f'drDataset/{folder_name}_metadata.csv'
    dataframe.to_csv(csv_path, index=False)
    print(f"Created: {csv_path} ({len(dataframe)} rows)")
    
    # Move the actual image files
    for _, row in dataframe.iterrows():
        filename = row['image_id'] 
        source = os.path.join('DR_Master_Dataset/images', filename)
        destination = os.path.join(dest_path, filename)
        
        if os.path.exists(source):
            shutil.copy(source, destination)
        else:
            # Helpful if some images from APTOS/DDR are missing extensions
            print(f"Warning: {filename} not found!")

# Run the process for all three sets
move_and_save_metadata(frontend_df, 'frontend_demo')
move_and_save_metadata(train_df, 'train')
move_and_save_metadata(val_df, 'val')