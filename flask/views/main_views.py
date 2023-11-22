from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


<<<<<<< HEAD
model = load_model("./model/videoyoga1.h5",compile=False)

if model:
    print("모델이 성공적으로 로드되었습니다.")
else:
    print("모델을 로드하지 못했습니다.")


=======
# bp = Blueprint('main', __name__, template_folder="templates")


# mp_drawing = mp.solutions.drawing_utils # Visualizing our poses
# mp_pose = mp.solutions.pose # Importing our pose estimation model (ex)hand,
>>>>>>> 8b6e1b162d6b15bd8561b2a89ee897f807730ab8

bp = Blueprint("main", __name__, template_folder="templates")

<<<<<<< HEAD
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2

app = Flask(__name__)
=======
# @bp.route('/')
# def home():
#     return render_template('home.html')

# @bp.route('/calendar')
# def calendar():
#     return render_template('calendar.html')

# @bp.route('/login')
# def login():
#     return render_template('login.html')


# @bp.route('/register')
# def register():
#     return render_template('register.html')


# @bp.route('/contact')
# def contact():
#     return render_template('contact.html')
>>>>>>> 8b6e1b162d6b15bd8561b2a89ee897f807730ab8




@bp.route('/detect')
def detect():
    return render_template("detect.html")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def feed():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    actions = np.array(['TreePose','WarriorPose1','WarriorPose2'])
    cap= cv2.VideoCapture(0)
# Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()



@bp.route('/video_feed')
def video_feed():
    return Response(feed(), mimetype='multipart/x-mixed-replace; boundary=frame')