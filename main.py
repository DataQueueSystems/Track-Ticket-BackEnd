from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
from datetime import datetime
import requests  # For making API calls
import mysql.connector
import numpy as np

app = Flask(__name__)

# MySQL connection setup
db_connection = mysql.connector.connect(
    host="localhost",  # Your MySQL server address (e.g., "localhost")
    user="root",       # Your MySQL username
    password="",  # Your MySQL password
    database="tickettrack"  # The database where the table is created
)
cursor = db_connection.cursor()

# Function to store user data in MySQL
def store_user_data(image_path, location, status="unverified"):
    query = "INSERT INTO passengers (profile, source,status,destination,fare,mode) VALUES (%s,  %s,%s, %s, %s,%s)"
    cursor.execute(query, (image_path, location, status,"",0,""))
    db_connection.commit()
    print(f"User data saved successfully: {image_path}, {location}, {status}")

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

captured = False  # Flag to indicate if an image has been captured
location_name = ""  # Variable to store location name

# Geocoding API (using Nominatim)
def get_location_name(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {"User-Agent": "your_application_name"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('display_name', "Unknown Location")
    return "Unknown Location"

def generate_frames():
    global captured, location_name
    camera = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Save the frame when a face is detected
                if not captured:  # Ensure only one image is captured
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = os.path.join(UPLOAD_FOLDER, f"face_{timestamp}.jpg")
                    cv2.imwrite(file_path, frame)

                    # Save to database
                    store_user_data(file_path, location_name, "unverified")
                    captured = True  # Set flag to true

            # Encode frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global captured
    captured = False  # Reset captured flag when starting feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_capture', methods=['POST'])
def check_capture():
    global captured, location_name
    if captured:
        # Get location details from request
        data = request.json
        lat = data.get('latitude')
        lon = data.get('longitude')
        if lat and lon:
            location_name = get_location_name(lat, lon)
        return jsonify({'status': 'captured', 'location_name': location_name})
    else:
        return jsonify({'status': 'not_captured'})

@app.route('/face_match', methods=['POST'])
def face_match():
    """Compare uploaded face image with database images."""
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'})

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file'})

    # Save the uploaded file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, "temp_face.jpg")
    uploaded_file.save(temp_path)

    # Load uploaded image
    uploaded_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    uploaded_image = cv2.resize(uploaded_image, (100, 100))  # Resize for uniformity

    # Query database for images
    query = "SELECT p_id,profile, source, status FROM passengers where status='unverified'"
    cursor.execute(query)
    results = cursor.fetchall()

    # Compare uploaded image with each image in the database
    for p_id,profile, source, status in results:
        db_image = cv2.imread(profile, cv2.IMREAD_GRAYSCALE)
        if db_image is None:
            continue
        db_image = cv2.resize(db_image, (100, 100))
        
        # Compare using Mean Squared Error (MSE)
        mse = np.mean((uploaded_image - db_image) ** 2)
        if mse < 200:  # Adjust threshold as needed
            os.remove(temp_path)  # Clean up temporary file
            return jsonify({'status': 'match_found','id':p_id, 'location': source, 'profile': profile, 'status': status})

    os.remove(temp_path)  # Clean up temporary file
    return jsonify({'status': 'no_match'})

@app.route('/update_passenger/<int:p_id>', methods=['PUT'])
def update_passenger(p_id):
    """
    Update passenger information based on p_id.
    Updates destination, fare, mode, and sets status to 'verified'.
    """
    # Parse request JSON
    data = request.json
    destination = data.get('destination')
    fare = data.get('fare')
    mode = data.get('mode')

    # Check for missing fields
    if not all([destination, fare, mode]):
        return jsonify({'status': 'error', 'message': 'Missing required fields: destination, fare, mode'}), 400

    # Update the database
    query = """
        UPDATE passengers
        SET destination = %s, fare = %s, mode = %s, status = 'verified'
        WHERE p_id = %s
    """
    try:
        cursor.execute(query, (destination, fare, mode, p_id))
        db_connection.commit()
        return jsonify({'status': 'success', 'message': f'Passenger with ID {p_id} updated successfully'})
    except mysql.connector.Error as err:
        return jsonify({'status': 'error', 'message': str(err)}), 500




@app.route('/exit', methods=['POST'])
def exit_and_verify():
    """
    Capture a photo, verify the person, and respond accordingly.
    """
    global captured, location_name
    captured = False  # Reset captured flag for this action

    camera = cv2.VideoCapture(0)  # Open the webcam
    success, frame = camera.read()  # Capture a single frame
    camera.release()  # Release the camera

    if not success:
        return jsonify({'status': 'error', 'message': 'Could not capture image'}), 500

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces are detected
    if len(faces) == 0:
        return jsonify({'status': 'error', 'message': 'No face detected'}), 400

    # Save the captured frame temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, "exit_capture.jpg")
    cv2.imwrite(temp_path, frame)

    # Load uploaded image
    uploaded_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    uploaded_image = cv2.resize(uploaded_image, (100, 100))  # Resize for uniformity

    # Query database for images
    query = "SELECT p_id, profile, source, status FROM passengers where status!='completed'"
    cursor.execute(query)
    results = cursor.fetchall()

    match_found = False
    matched_record = None

    # Compare uploaded image with each image in the database
    for p_id, profile, source, status in results:
        db_image = cv2.imread(profile, cv2.IMREAD_GRAYSCALE)
        if db_image is None:
            continue
        db_image = cv2.resize(db_image, (100, 100))

        # Compare using Mean Squared Error (MSE)
        mse = np.mean((uploaded_image - db_image) ** 2)
        if mse < 200:  # Adjust threshold as needed
            match_found = True
            matched_record = (p_id, source, profile, status)
            break

    # os.remove(temp_path)  # Clean up temporary file


    

    if match_found:
        p_id, source, profile, status = matched_record
  
        if status == "verified":
            try:
                update_query = "UPDATE passengers SET status = 'completed' WHERE p_id = %s"
                cursor.execute(update_query, (p_id,))
                db_connection.commit()  # Commit the transaction to save changes
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Error updating status: {str(e)}'}), 500
            return jsonify({'status': 'success', 'message': 'Person verified successfully, door opens.'})
        else:
            # Insert a notification into the notifications table
            notification_message = f'Person with image {profile} is not verified.'
            insert_notification(notification_message)
            return jsonify({'status': 'not_verified', 'message': 'Person not verified, notification logged.'})
    else:
        return jsonify({'status': 'no_match', 'message': 'No matching person found in the database.'})

def insert_notification(message):
    """Insert a notification into the notifications table."""
    query = "INSERT INTO notifications (message, status) VALUES (%s, %s)"
    cursor.execute(query, (message,'sent'))
    db_connection.commit()

def generate_exit_frames():
    global captured
    camera = cv2.VideoCapture(0)  # Open the webcam for exit verification
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Save the frame if required for processing
                if not captured:  # Ensure only one image is captured
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = os.path.join(UPLOAD_FOLDER, f"exit_face_{timestamp}.jpg")
                    cv2.imwrite(file_path, frame)

                    # Optional: You can add processing logic here if needed
                    captured = True

            # Encode frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/exit_video_feed')
def exit_video_feed():
    global captured
    captured = False  # Reset captured flag when starting exit feed
    return Response(generate_exit_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')







if __name__ == '__main__':
    # Ensure the server is running in debug mode
    app.run(debug=True,host='0.0.0.0', port=5000)