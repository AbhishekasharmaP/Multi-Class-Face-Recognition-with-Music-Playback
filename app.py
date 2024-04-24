import cv2
import pandas as pd
import numpy as np
import os
import pickle
import threading
import vlc
import time
import mediapipe as mp
from scipy.spatial.distance import euclidean
from collections import Counter, defaultdict
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Model
from pytube import YouTube
from googleapiclient.discovery import build

# Load EfficientNet model
model = load_model("D:\\ABHI\\Project final year\\final\\New folder (2)\\efficient_sciemese_greay_1000.h5")

# Load gesture recognizer model and class names
ges_model = load_model('mp_hand_gesture')
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load feature vectors
features_save_path = "features.pkl"
with open(features_save_path, 'rb') as f:
    all_features = pickle.load(f)

# Load music player instance
instance = vlc.Instance('--no-xlib')

# Initialize variables for gesture recognition
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize variables for controlling music playback
flag = 0
flag_ges = 0
action = ''

update_interval = 3
last_update_time = time.time()

# for gesture recognition
update_interval_ges = 3
last_update_time_ges = time.time()

# YouTube Data API key
API_KEY = "AIzaSyC6IakWhNg5qlvoWHUKP8zNhxmsyrz_2RU"

# Initialize the YouTube Data API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Queue to store predicted classes
predicted_queue = []

df= pd.read_csv("D:\\ABHI\\Project final year\\final\\ratings.csv")

df.set_index('Unnamed: 0', inplace =True)

p1=" "

def similary_score(predicted_labels):
  #df = pd.DataFrame(data['Ratings'], columns=data['Music_Category'], index=data['person'])
  #print(df)
  #print()
  length = len(predicted_labels)
  if length == 1:
    user1 = df.loc[predicted_labels[-1]].values.reshape(1, -1)
    #print(user1)
    r= np.argmax(user1)
    category = df.columns[r]
    return category
  elif length == 2:
    user1_ratings = df.loc[predicted_labels[-1]].values.reshape(1, -1)
    user2_ratings = df.loc[predicted_labels[-2]].values.reshape(1, -1)
    #print(user1_ratings)
    #print(user2_ratings)
    mean = []
    for i in range(len(user1_ratings[0])):
      score = np.mean((user1_ratings[:, i].reshape(1, -1), user2_ratings[:, i].reshape(1, -1)))
      mean.append(score)
    #print(mean)
    max=np.argmax(mean)
    r = df.columns[max]
    return r
  elif length >= 3:
    user1 = df.loc[predicted_labels[-1]].values.reshape(1, -1);print(user1)
    user2 = df.loc[predicted_labels[-2]].values.reshape(1, -1);print(user2)
    user3 = df.loc[predicted_labels[-3]].values.reshape(1, -1);print(user3)
    mean = []
    for i in range(len(user1[0])):
      score = np.mean((user1[:, i].reshape(1, -1), user2[:, i].reshape(1, -1) , user3[:, i].reshape(1, -1)))
      mean.append(score)
    #print(mean)
    max=np.argmax(mean)
    r = df.columns[max]
    return r
  else:
    print("Invalid!!")

# Function to preprocess image
def preprocess_image(image_path):
    gray_img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (224, 224))
    rgb_img = np.stack((resized_img,) * 3, axis=-1)
    normalized_img = rgb_img.astype('float32') / 255.0
    img_batch = np.expand_dims(normalized_img, axis=0)
    return img_batch

# Function to extract features
def extract_features(image):
    return model.predict(image)

# Function to compute similarity using k-NN
def compute_similarity(input_features, all_features, k=5):
    all_distances = []
    for label, class_features in all_features.items():
        for img_features_flat in class_features:
            distance = euclidean(input_features.flatten(), img_features_flat)
            if distance == 0:
                distance = 1e-6
            weight = 1 / (distance**2)
            all_distances.append((label, distance, weight))
    all_distances.sort(key=lambda x: x[1])
    top_k_labels_distances_weights = all_distances[:k]

    total_similarity = sum([1 / (d**2) for _, d, _ in top_k_labels_distances_weights])
    class_contributions = {}
    for label, distance, weight in top_k_labels_distances_weights:
        normalized_weight = weight / total_similarity
        class_contributions[label] = class_contributions.get(label, 0) + normalized_weight
    most_common_label = max(class_contributions, key=class_contributions.get)
    return most_common_label

# Function to play music
def play_music(predicted_class):
    global flag
    global flag_ges
    global action

    music_folder = 'music'

    if predicted_class in os.listdir(music_folder):
        playlist_folder = os.path.join(music_folder, predicted_class)
        music_files = [os.path.join(playlist_folder, f) for f in os.listdir(playlist_folder) if f.endswith(".mp3")]

        if music_files:
            media_list = instance.media_list_new()
            for music_file in music_files:
                media_list.add_media(instance.media_new(music_file))

            player = instance.media_list_player_new()
            player.set_media_list(media_list)

            player.play()
            song_index = 0

            while song_index < len(music_files):
                if action == '1':  # pause the song(stop)
                    action = ''
                    player.get_media_player().pause()

                if action == '4':  # play the song(fist)
                    action = ''
                    player.get_media_player().play()

                if action == '2':  # next song(thumbs up||call me)
                    action = ''
                    song_index += 1
                    if song_index < len(music_files):
                        player.get_media_player().set_media(instance.media_new(music_files[song_index]))
                        player.play()
                        while player.get_state() != vlc.State.Playing:
                            pass
                if action == '3':  # previous song(thumbs down)
                    action = ''
                    song_index -= 1
                    if song_index < len(music_files):
                        player.get_media_player().set_media(instance.media_new(music_files[song_index]))
                        player.play()
                        while player.get_state() != vlc.State.Playing:
                            pass
                if action == '6':  # stop music palying function(rock)
                    action = ''
                    flag = 0
                    flag_ges = 0
                    break
                if player.get_state() == vlc.State.Ended:
                    flag = 0
                    flag_ges = 0
                    break

            player.get_media_player().stop()
            player.release()

# Function to start final function and music threads
def start_final_function_and_music(predicted_class):
    global flag
    global flag_ges

    if not flag:
        flag = 1
        flag_ges = 1
        '''
        final_thread = threading.Thread(target=avgsim, args=(predicted_class[0],))
        final_thread.start()
        '''
        print(predicted_class)
        if len(predicted_class)>1:
            similar_music=similary_score(predicted_class)
            print(f'\nMost similar music for{predicted_class} is: ',similar_music)
            multi_person_music_thread = threading.Thread(target=play_music_multiple_person, args=(similar_music,))
            #print('\n',predicted_class)
            multi_person_music_thread.start()
        if len(predicted_class)==1:
            music_thread = threading.Thread(target=play_music, args=(predicted_class[0],))
            #print('\n',predicted_class[0])
            music_thread.start()

# Function to compute gesture
def compute_gesture(frame):
    global flag_ges
    global action

    x, y, c = frame.shape
    frame1 = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            prediction = ges_model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

    return className


# Function to process images from the camera
def process_camera_images():
    global flag
    global flag_ges
    global action
    global update_interval
    global last_update_time
    global last_update_time_ges
    global predicted_queue

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if time.time() - last_update_time >= update_interval:
                #predicted_queue = []  # Clear predicted queue
                for (x, y, w, h) in faces:
                    roi = frame[y:y + h, x:x + w]
                    predicted_class = avgsim(roi)
                    if predicted_class not in predicted_queue:
                        predicted_queue.append(predicted_class)
                '''
                if len(predicted_queue) > 1:
                    query = " innashtu bekenna".join(predicted_queue)   # Construct query from multiple predictions
                    print("Predicted Faces:", predicted_queue)
                    play_music_multiple_person(query)
                else:
                '''
                start_final_function_and_music(predicted_queue)

                last_update_time = time.time()

            gesture = compute_gesture(frame)

            if time.time() - last_update_time_ges >= update_interval_ges:
                last_update_time_ges = time.time()
                action = gesture
                # print('\n\n', action)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to compute similarity and return predicted class
def avgsim(test_img):
    global all_features

    test_image = preprocess_image(test_img)
    test_features = extract_features(test_image)
    most_similar_class = compute_similarity(test_features.flatten(), all_features)
    #print('\n', most_similar_class)
    return most_similar_class

# Function to search videos
def search_videos(query, max_results=20):
    search_response = youtube.search().list(
        q=query,
        part='id',
        maxResults=max_results,
        type='video'
    ).execute()

    videos = []
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append(search_result['id']['videoId'])
    return videos

# Function to extract audio
def extract_audio(video_id):
    try:
        video_info = youtube.videos().list(
            part="snippet,contentDetails",
            id=video_id
        ).execute()

        if not video_info['items']:
            print(f"Video {video_id} not found.")
            return None

        video_item = video_info['items'][0]

        if 'liveBroadcastContent' in video_item['snippet']:
            if video_item['snippet']['liveBroadcastContent'] == 'live':
                print(f"Skipping live stream video: {video_id}")
                return None

        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        stream = yt.streams.filter(only_audio=True).first()
        return stream.url
    except Exception as e:
        print(f"Error extracting audio for video {video_id}: {e}")
        return None

# Function to play audio
def play_audio(url, player):
    global p1
    if url is None:
        print("Error: Unable to extract audio URL.")
        player=p1
        return player

    if player:
        player.stop()  # Stop current player if exists

    Instance = vlc.Instance()
    player = Instance.media_player_new()
    Media = Instance.media_new(url)
    Media.get_mrl()
    player.set_media(Media)
    player.play()
    return player


# Function to play music for multiple persons
def play_music_multiple_person(song_query):
    global flag
    global flag_ges
    global action

    videos = search_videos(song_query)
    if videos:
        player = None  # Initialize player object
        current_video_index = 0
        while True:

            video_id = videos[current_video_index]
            audio_url=extract_audio(video_id)

            print(f"Playing video {current_video_index + 1} of {len(videos)}")
            player = play_audio(audio_url, player)  # Pass player object

            while True:

                if action == '2':  # Next song   thumbs up
                    action = ''
                    current_video_index = (current_video_index + 1) % len(videos)
                    break
                elif action == '3':  # Previous song    thumbs down
                    action = ''
                    current_video_index = (current_video_index - 1) % len(videos)
                    break
                elif action == '6':  # Quit   rock(thumb folded toghether with middle and ring finger)
                    flag = 0
                    player.stop()
                    flag_ges = 0
                    break
                elif action == '###':  # Stop
                    action = ''
                    player.stop()
                    break
                elif action == '1':  # Toggle play/pause    high five
                    action = ''
                    player.pause()
                elif action == '4':  # Resume    fist
                    action = ''
                    player.play()


            if current_video_index < 0:
                    current_video_index = 0
            elif current_video_index >= len(videos):
                    current_video_index = len(videos) - 1
            if player.get_state() == vlc.State.Ended:
                flag = 0
                flag_ges = 0
                break

            #player.stop()
            #player.release()

    else:
        print("No videos found.")

# Start processing images from the camera
process_camera_images()
