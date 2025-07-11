import os
import pickle
import mediapipe as mp
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Configure the Hands model for static image processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the dataset directory
DATA_DIR = './data'

# Initialize lists to hold processed data and labels
data = []
labels = []

# Process each directory (representing a class) in the dataset
for dir_ in os.listdir(DATA_DIR):
    print(f"Processing class: {dir_}")  # Output the class being processed
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read and convert the image to RGB for MediaPipe
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the Hands model
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Process hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through landmarks to collect x and y coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Calculate relative coordinates in a single loop
                min_x = min(x_)
                min_y = min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            # Add the processed data and the corresponding label
            data.append(data_aux)
            labels.append(dir_)

            # Output the processed data for this image
            print(f"Processed {img_path} from class {dir_}")
            print(f"Landmark data (first 10 values): {data_aux[:10]}")  # Display the first 10 values for brevity
        else:
            print(f"No hand landmarks detected in {img_path}")

# Save the processed data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Release the MediaPipe Hands model
hands.close()

# Output final summary
print(f"\nTotal images processed: {len(labels)}")
print(f"Classes: {set(labels)}")
