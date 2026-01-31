import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_DIR = 'archive_resized'  # Your folder name
IMG_SIZE = 64                 # 64x64 pixels

# --- 1. DEFINE CATEGORIES (EXACT SPELLING MATTERS) ---
# We hardcode this list to ensure the order stays the same forever.
CATEGORIES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

print(f"Looking for data in: {os.path.abspath(DATA_DIR)}")

# Check if main folder exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: The folder '{DATA_DIR}' was not found.")
    exit()

data = []
labels = []

print("Step 1: Loading images from all 4 folders...")

# --- 2. LOAD & FLATTEN IMAGES ---
for category_idx, category_name in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category_name)
    
    # Check if this specific sub-folder exists
    if not os.path.exists(folder_path):
        print(f"WARNING: Folder '{category_name}' not found! Skipping it.")
        continue

    image_count = 0
    print(f"   -> Reading folder: {category_name}...")

    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = imread(img_path)
            
            if img is None:
                continue
                
            # Resize
            img_resized = resize(img, (IMG_SIZE, IMG_SIZE, 3))
            
            # Flatten
            flat_data = img_resized.flatten()
            
            data.append(flat_data)
            labels.append(category_idx)
            image_count += 1
            
        except Exception as e:
            pass
            
    print(f"      Loaded {image_count} images.")

# Convert to NumPy
data = np.array(data)
labels = np.array(labels)

print(f"Total Images: {len(data)}")

# --- 3. TRAINING ---
# Split data (80% Train, 20% Test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)

print("Step 2: Training the Random Forest... (This might take a bit longer now)")
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# --- 4. TESTING ---
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 5. SAVING ---
print("Step 3: Saving model.pkl...")
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("SUCCESS! 'model.pkl' is ready.")