# Import necessary libraries
from keras.models import model_from_json
import cv2
import numpy as np

# Load the trained model architecture from JSON file
json_file = open("signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()

# Create model from JSON
model = model_from_json(model_json)

# Load the trained weights
model.load_weights("signlanguagedetectionmodel48x48.h5")

print("Model loaded successfully!")

# Function to extract and preprocess features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define labels for the 6 classes the model was trained on
# Note: This model only recognizes these 6 classes, not full A-Z
label = ['A', 'M', 'N', 'S', 'T', 'blank']

print("Starting real-time detection...")
print("Press 'q' to quit")

while True:
    # Capture frame from webcam
    _,frame = cap.read()
    
    # Draw ROI rectangle on frame
    cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
    
    # Extract ROI
    cropframe=frame[40:300,0:300]
    
    # Convert to grayscale
    cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    cropframe = cv2.resize(cropframe,(48,48))
    
    # Extract features
    cropframe = extract_features(cropframe)
    
    # Make prediction
    pred = model.predict(cropframe, verbose=0)
    prediction_label = label[pred.argmax()]
    
    # Draw prediction text background
    cv2.rectangle(frame, (0,0), (300, 40), (0, 165, 255), -1)
    
    # Display prediction
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
    
    # Show output
    cv2.imshow("Sign Language Detection",frame)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print("Closing...")
cap.release()
cv2.destroyAllWindows()
