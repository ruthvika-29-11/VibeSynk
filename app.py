# Importing additional module for hashing passwords
# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import hashlib

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User credentials (hardcoded for simplicity; consider a database for production)
USER_CREDENTIALS = {
    "user1": hash_password("password1"),
    "user2": hash_password("password2")
}

# Login function
def login():
    # Login function
        st.title("Login to VibeSynk")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == hash_password(password):
                st.session_state["authenticated"] = True
                st.success("Login successful! Redirecting...")
                st.query_params.update({"authenticated": "true"})  # Set query parameter
            else:
                st.error("Invalid username or password.")


# Main app
# Main app
# Main app

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

    # Check query parameters for authentication
if st.query_params.get("authenticated") == "true":
    st.session_state["authenticated"] = True

if not st.session_state["authenticated"]:
    login()
else:
    st.sidebar.title("VibeSynk")
    logout = st.sidebar.button("Logout")
    if logout:
        st.session_state["authenticated"] = False
        st.query_params.update({"authenticated": "false"})  # Reset query parameter
        st.stop()  # Stops the script execution and refreshes the app
    # Refresh the app
    df = pd.read_csv("muse_v3.csv")
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']

    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    # Split data into emotion categories
    df_sad = df[:18000]
    df_fear = df[18000:36000]
    df_angry = df[36000:54000]
    df_neutral = df[54000:72000]
    df_happy = df[72000:]


    # Helper function to generate recommended songs
    def recommend_songs(emotions):
        data = pd.DataFrame()
        emotion_map = {
            'Neutral': df_neutral,
            'Angry': df_angry,
            'Fearful': df_fear,
            'Happy': df_happy,
            'Sad': df_sad
        }
        sample_sizes = [30, 20, 15, 10, 5]

        for emotion, size in zip(emotions, sample_sizes[:len(emotions)]):
            if emotion in emotion_map:
                # Exclude songs containing 'api.spotify' in the link
                filtered_data = emotion_map[emotion][
                    ~emotion_map[emotion]['link'].str.contains('api.spotify', na=False)]
                data = pd.concat([data, filtered_data.sample(n=size, replace=True)], ignore_index=True)
        return data


    # Preprocess detected emotions
    def process_emotions(emotion_list):
        emotion_counts = Counter(emotion_list)
        unique_emotions = list(emotion_counts.keys())
        return unique_emotions


    # Build and load the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights('model.h5')

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Original app code starts here
    # Adding a background to the webpage
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://cdn.pixabay.com/photo/2016/03/27/20/55/abstract-1283872_1280.jpg');
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }
        .sidebar .sidebar-content {
            background: rgba(0, 0, 0, 0.7);
        }
        h1, h4, h5 {
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center;'><b>VibeSynk: Where Emotions and Music Collide</b></h1>",
                unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>Click on a song name to open it on Last.fm</b></h5>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    detected_emotions = []

    # Webcam integration
    # Webcam integration
    if col2.button('SCAN EMOTION (Click here)'):
        cap = cv2.VideoCapture(0)
        detected_emotions.clear()
        st.write("Initializing camera...")

        # Warm-up the camera by skipping the first few frames
        for _ in range(10):  # Skip 10 frames
            ret, _ = cap.read()

        st.write("Scanning emotions...")

        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

                prediction = model.predict(roi_gray)
                max_index = int(np.argmax(prediction))
                detected_emotions.append(emotion_dict[max_index])

                # Display the detected emotion on the frame
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
            st.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
        ec = Counter(detected_emotions)
        emotion = ec.most_common(1)[0][0]
        detected_emotions = process_emotions(detected_emotions)
        st.success(f"Emotion: {emotion} detected successfully!")

    # Display recommended songs
    recommended_songs = recommend_songs(detected_emotions)
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs</b></h5>", unsafe_allow_html=True)
    st.write("--------------------------------------------------")

    for _, row in recommended_songs.iterrows():
        st.markdown(
          f"""
                <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px; text-align: center; background-color: rgba(255, 255, 255, 0.8);">
                    <h4><a href='{row['link']}' style="text-decoration: none; color: #333;">{row['name']} by {row['artist']}</a></h4>
                </div>
                """,
          unsafe_allow_html=True
        )
        st.write("--------------------------------------------------")
