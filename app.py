import cv2
import mysql.connector
import os
from datetime import datetime

# MySQL connection setup
db_connection = mysql.connector.connect(
    host="localhost",  # Your MySQL server address (e.g., "localhost")
    user="root",       # Your MySQL username
    password="",  # Your MySQL password
    database="tickettrack"  # The database where the table is created
)
cursor = db_connection.cursor()

# Function to store user data in MySQL
def store_user_data(image_path, location, status):
    query = "INSERT INTO passengers (profile, source, status) VALUES (%s, %s, %s)"
    cursor.execute(query, (image_path, location, status))
    db_connection.commit()
    print(f"User data saved successfully at {image_path}")

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(1)

# Directory to store images
image_dir = "uploads"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected
    if len(faces) > 0:
        # Draw bounding boxes around each face detected
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the captured image to the server directory
        image_filename = os.path.join(image_dir, f'face_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        cv2.imwrite(image_filename, frame)

        # Get the current location (you can replace this with actual location data)
        location = "Dummy Location"  # Replace with actual location
        status = "unverified" # Get the current date and time

        # Store the image path in MySQL
        store_user_data(image_filename, location, status)

        # Display the frame with detected face(s) and bounding box
        cv2.imshow('Face Detection', frame)

        # Wait for the user to press any key before closing the window
        cv2.waitKey(0)

        # Break the loop after saving the picture and displaying it
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Close the database connection
cursor.close()
db_connection.close()
