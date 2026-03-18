import pandas as pd
import os
import shutil

# --- CONFIGURATION ---
BASE_PATH = "data" 
OUTPUT_DIR = "DR_Master_Dataset"
IMAGE_OUT = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_OUT, exist_ok=True)

all_metadata = []

def move_and_record(old_path, new_name, label, source):
    if os.path.exists(old_path):
        target_path = os.path.join(IMAGE_OUT, new_name)
        shutil.copy(old_path, target_path)
        all_metadata.append({'image_id': new_name, 'dr_grade': int(label), 'source': source})
    else:
        # This will tell you exactly which file path is wrong
        print(f"  [!] File not found: {old_path}")

# --- 1. PROCESS APTOS ---
print("Processing APTOS...")
try:
    aptos_csv = pd.read_csv(os.path.join(BASE_PATH, 'aptos', 'train.csv'))
    for _, row in aptos_csv.iterrows():
        img_id = row['id_code']
        # Double check if folder is 'train_images' or just 'train'
        old = os.path.join(BASE_PATH, 'aptos', 'train_images', f"{img_id}.png")
        move_and_record(old, f"AP_{img_id}.png", row['diagnosis'], 'APTOS')
except Exception as e: print(f"APTOS Error: {e}")

# --- 2. PROCESS DDR ---
print("\nProcessing DDR...")
try:
    # Check filename: is it 'DR_grading.csv' or 'dr_grading.csv'?
    csv_path = os.path.join(BASE_PATH, 'ddr','DR_grading.csv')
    if not os.path.exists(csv_path):
        print(f"  [!] CSV Missing at: {csv_path}")
    else:
        ddr_csv = pd.read_csv(csv_path)
        for _, row in ddr_csv.iterrows():
            # Your path: data/ddr/DR_grading/DR_grading/ID.jpg
            old = os.path.join(BASE_PATH, 'ddr', 'DR_grading', 'DR_grading', f"{row['id_code']}")
            move_and_record(old, f"DDR_{row['id_code']}", row['diagnosis'], 'DDR')
except Exception as e: print(f"DDR Error: {e}")

# --- 3. PROCESS MESSIDOR-2 ---
print("\nProcessing Messidor-2...")
try:
    m_csv_path = os.path.join(BASE_PATH,  'messidor','messidor_data.csv') 
    if os.path.exists(m_csv_path):
        mess_csv = pd.read_csv(m_csv_path)
        
        # Filter: Only keep images that doctors could read
        mess_gradable = mess_csv[mess_csv['adjudicated_gradable'] == 1]
        
        for _, row in mess_gradable.iterrows():
            img_id = row['id_code']
            label = row['diagnosis']  # Using 'diagnosis' as per your data
            
            # Updated path based on your folder structure
            old = os.path.join(BASE_PATH, 'messidor', 'messidor-2', 'messidor-2', 'preprocess', img_id)
            
            # Note: img_id already contains .png or .JPG, so we just use MS_ prefix
            move_and_record(old, f"MS_{img_id}", label, 'Messidor2')
    else:
        print(f"  [!] Messidor CSV not found at: {m_csv_path}")
except Exception as e: 
    print(f"Messidor Error: {e}")

# --- 4. PROCESS EYEPACS ---
print("\nProcessing EyePACS...")
# Check spelling: release-crop or release_crop?
ep_nrg_path = os.path.join(BASE_PATH, 'eyepacs', 'release-crop', 'release-crop', 'train', 'NRG')
if os.path.exists(ep_nrg_path):
    ep_files = os.listdir(ep_nrg_path)[:2000]
    for img_name in ep_files:
        old = os.path.join(ep_nrg_path, img_name)
        move_and_record(old, f"EP_{img_name}", 0, 'EyePACS')
else:
    print(f"  [!] EyePACS folder missing at: {ep_nrg_path}")

# --- SAVE ---
if all_metadata:
    master_df = pd.DataFrame(all_metadata)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "master_metadata.csv"), index=False)
    print(f"\nSUCCESS! Total images collected: {len(master_df)}")
    print(master_df['dr_grade'].value_counts().sort_index())
else:
    print("\n[!!!] NO IMAGES COLLECTED. Check 'File not found' errors above.")