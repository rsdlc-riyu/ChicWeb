import os
import wave
import numpy as np
import tensorflow as tf
import uuid
import threading
import io
import librosa
import serial
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, send
from flask_socketio import emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO

# Create Flask app
app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins='*')
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = 'b880774a78a36ec4afe090493af57ddac1934ea1c7d9b1b4223f6afc33c5f931'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history_logs.db'
db = SQLAlchemy(app)

# Serial communication settings
serial_port = 'COM4'  # Update this with your IoT device's serial port
baud_rate = 115200

# Load the trained model
model_path = 'ChicDistressVocalizations.h5'
model = tf.keras.models.load_model(model_path)

# Define the target classes
classes = {
    0: "Normal",
    1: "Alarm Call",
    2: "Brooding",
    3: "Cackling",
    4: "Clucking",
    5: "Crying",
    6: "Squawking"
}

# HistoryLog model
class HistoryLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10))
    time = db.Column(db.String(10))
    distress_type = db.Column(db.String(50))
    average_frequency = db.Column(db.Float)
    average_magnitude = db.Column(db.Float)
    max_duration = db.Column(db.Float)

    def __repr__(self):
        return f"<HistoryLog(id={self.id}, date={self.date}, time={self.time}, distress_type={self.distress_type}, " \
               f"average_frequency={self.average_frequency}, average_magnitude={self.average_magnitude}, " \
               f"max_duration={self.max_duration})>"

# Global variable to store the latest distress type and audio data
latest_distress_type = "Unknown"
latest_audio_data = None
max_duration = 0.0

# Function to update the global variable with the latest distress type and audio data
def update_distress_type_and_audio_data(distress_type, audio_data):
    global latest_distress_type, latest_audio_data, max_duration
    latest_distress_type = distress_type
    latest_audio_data = audio_data

    # Update the maximum duration
    max_duration = len(audio_data) / 48000.0  # Assuming sample rate is 48000 (adjust if different)

# Function to read and process raw audio data from the serial port
def read_and_process_audio():
    distress_type = "Unknown"  # Initialize the distress type as Unknown
    audio_data = None

    with serial.Serial(serial_port, baud_rate) as ser:
        while True:
            try:
                raw_audio_data = ser.readline().strip().decode('utf-8', 'ignore')
                if raw_audio_data and ',' in raw_audio_data:  # Check if the received data has a comma
                    print(f"Raw Audio Data: {raw_audio_data}")

                    # Step 1: Display received data from IoT in the terminal
                    print(f"Received data from IoT: {raw_audio_data}")

                    # Step 2: Convert raw audio data to PCM WAV
                    audio_data, sample_rate = preprocess_audio(raw_audio_data)
                    if audio_data is None or sample_rate is None:
                        print("Error processing audio data.")
                        continue

                    # Step 4: Extract MFCC features from the PCM WAV
                    mfcc_features = extract_features(audio_data)

                    # Step 5: Classify using the model
                    distress_type = classify_audio(mfcc_features)

                    # Update the global variables
                    update_distress_type_and_audio_data(distress_type, mfcc_features)

                    # Step 6: Send the MFCC data to the frontend
                    send_mfcc_data_to_frontend(mfcc_features.tolist())  # Convert to a list for JSON serialization

                    # Step 7: Send the classification result to the frontend
                    send_classification_result_to_frontend(distress_type)

                    # Step 8: Calculate and display frequency, amplitude, and time
                    calculate_and_display_frequency_amplitude_time(audio_data, sample_rate)

                    # Step 9: Send the frequency and magnitude spectrum data to the frontend
                    frequencies, magnitude_spectrum = calculate_frequency(audio_data, sample_rate)
                    send_spectrum_data_to_frontend(frequencies, magnitude_spectrum)

                    # Save history log to the database
                    save_history_log(distress_type, frequencies, magnitude_spectrum, max_duration)

                    print(f"Distress Type: {distress_type}")

                else:
                    print("Invalid or empty raw audio data received:", raw_audio_data)
            except Exception as e:
                print(f"Error: {e}")
                break

# Function to save the history log to the database
def save_history_log(distress_type, frequencies, magnitude_spectrum, max_duration):
    if latest_audio_data is not None:
        # Calculate average frequency and average magnitude
        average_frequency = np.mean(frequencies)
        average_magnitude = np.mean(magnitude_spectrum)

        # Get the current date and time
        now = datetime.now()
        date_str = now.strftime("%m-%d-%Y")
        time_str = now.strftime("%H:%M:%S")

        # Create a new history log entry and add it to the database
        log_entry = HistoryLog(date=date_str, time=time_str, distress_type=distress_type,
                               average_frequency=average_frequency, average_magnitude=average_magnitude,
                               max_duration=max_duration)
        db.session.add(log_entry)
        db.session.commit()

# Function to fetch history logs from the database
def fetch_history_logs():
    with app.app_context():  # Set up the application context
        logs = HistoryLog.query.all()
        history_logs = [
            {
                'date': log.date,
                'time': log.time,
                'distress_type': log.distress_type,
                'average_frequency': log.average_frequency,
                'average_magnitude': log.average_magnitude,
                'max_duration': log.max_duration
            }
            for log in logs
        ]
        return history_logs

# Backend route to get history logs from the database
@app.route('/get_history_logs')
def get_history_logs():
    history_logs = fetch_history_logs()
    return jsonify(history_logs)

# Function to calculate frequency information from audio data
def calculate_frequency(audio_data, sample_rate):
    # Perform the Fast Fourier Transform (FFT) on the audio data
    fft_result = np.fft.fft(audio_data)

    # Get the magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)

    # Calculate the frequencies corresponding to each element in the magnitude spectrum
    frequencies = np.fft.fftfreq(len(audio_data), d=1/sample_rate)

    return frequencies, magnitude_spectrum

# Function to calculate and display frequency, amplitude, and time
def calculate_and_display_frequency_amplitude_time(audio_data, sample_rate):
    # Calculate the frequency, amplitude, and time
    frequencies, magnitude_spectrum = calculate_frequency(audio_data, sample_rate)
    amplitude = np.max(np.abs(audio_data))
    time = len(audio_data) / sample_rate

    # Print amplitude, frequency, and time
    print("Amplitude:", amplitude)
    print("Frequency Spectrum:", frequencies)
    print("Magnitude Spectrum:", magnitude_spectrum)
    print("Time (seconds):", time)

    # Send the maximum duration to the frontend
    socketio.emit('max_duration', {'duration': max_duration}, namespace='/')

# Function to convert raw audio data to PCM WAV format
# (Step 2 in the data flow)
def preprocess_audio(raw_audio_data):
    try:
        # Split the raw audio data by comma and convert to integers
        audio_data = [int(value.strip()) for value in raw_audio_data.split(',')]
    except ValueError:
        # Handle invalid values or empty strings (you may log or raise an exception here)
        return None, None

    print("Received Raw Audio Data:", audio_data)  # Add this line to log the raw audio data

    # Assuming the sample rate is 44100 (adjust if it's different for your setup)
    sample_rate = 48000

    # Normalize the audio data to the range [-1, 1]
    audio_data = np.array(audio_data) / 16384.0

    print("Converted Audio Data to PCM WAV format:", audio_data)  # Add this line to log the PCM WAV audio data

    return audio_data, sample_rate


# Function to extract MFCC features from the audio data
# (Step 3 in the data flow)
def extract_features(audio_data):
    # Convert the audio data to a numpy array
    audio_data = np.array(audio_data)

    # Scale the audio data to the range [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Create a directory for temporary files
    temp_dir = 'temp_files'
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique temporary WAV file path
    temp_path = os.path.join(temp_dir, f'temp_{uuid.uuid4().hex}.wav')

    # Save the audio data as a temporary WAV file
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_data.tobytes())

    # Read the temporary WAV file using librosa
    audio, _ = librosa.load(temp_path, sr=48000)

    # Remove the temporary WAV file
    os.remove(temp_path)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=20, n_fft=1024)  # Reduced n_fft value

    # Pad or truncate the MFCC features to a fixed length of 38
    MAX_FRAMES = 38
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Normalize the MFCCs
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Reshape the MFCCs into a 4D array
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Return the reshaped MFCCs as the feature
    return mfcc

# Function to classify the audio using the model
# (Step 4 in the data flow)
def classify_audio(mfcc_features):
    if mfcc_features is None:
        return "Unknown"

    # Perform the prediction
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction)
    predicted_class = classes[predicted_label]

    return predicted_class

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Function to send the result of classification and waveform to the frontend
def send_classification_result_to_frontend(distress_type):
    global latest_audio_data

    # Step 6: Use the latest PCM WAV for the MFCC graph
    if latest_audio_data is not None:
        mfcc_features = extract_features(latest_audio_data)
        # Send the MFCC features to the client for the graph
        socketio.emit('mfcc_features', {'mfcc_features': mfcc_features.tolist()}, namespace='/')

    # Step 7: Send the result of the classification to the client
    socketio.emit('classification_result', {'distress_type': distress_type}, namespace='/')

# Function to send frequency and magnitude spectrum data to the frontend
def send_spectrum_data_to_frontend(frequencies, magnitude_spectrum):
    # Convert the arrays to lists for JSON serialization
    frequencies_list = frequencies.tolist()
    magnitude_spectrum_list = magnitude_spectrum.tolist()

    # Send the frequency and magnitude spectrum data to the client
    socketio.emit('spectrum_data', {'frequencies': frequencies_list, 'magnitude_spectrum': magnitude_spectrum_list}, namespace='/')

# Function to send MFCC data to the frontend
def send_mfcc_data_to_frontend(mfcc_data):
    socketio.emit('mfcc_data', {'mfcc_data': mfcc_data}, namespace='/')

if __name__ == '__main__':
    # Create the database tables
    with app.app_context():
        db.create_all()

    # Start a new thread to read and process audio data in real-time
    audio_thread = threading.Thread(target=read_and_process_audio)
    audio_thread.daemon = True
    audio_thread.start()

    # Use socketio.run() to run the app with WebSocket support
    socketio.run(app, host='127.0.0.1', port=5000)
