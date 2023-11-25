from flask import Flask, request, redirect, url_for, render_template, session, flash, Response
import pandas as pd
import pymysql


#from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO,emit
import cv2
import mediapipe as mp
import numpy as np
import datetime
#from scipy import spatial
import math
from flask import jsonify
#from tensorflow.keras.models import load_model
#from views.main_views import bp 
from views.login_views import log
from db_config import db  # db_config 모듈에서 db 불러오기


#from tensorflow.keras.models import load_model

#model = load_model("./model/videoyoga10.h5",compile=False)

#if model:
#    print("모델이 성공적으로 로드되었습니다.")
#else:
#    print("모델을 로드하지 못했습니다.")



app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'your_secret_key_here'

#app.register_blueprint(bp, url_prefix='/')
app.register_blueprint(log,url_prefix='/')



# 홈 페이지
@app.route('/')
def home():
    return render_template('home.html')

# 기록 페이지
@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

# 운동 후 페이지
@app.route('/after')
def after():
    hits = 10  # Replace with actual logic to get this value
    total = 14  # Replace with actual logic to get this value

    def calculate_grade(hits, total):
        if total == 0:
            return 'N/A', '#555555'  # Dark gray for undefined grade

        percentage = (hits / total) * 100

        if percentage >= 90:
            return 'A', '#4c9a2a'  # Dark green
        elif percentage >= 80:
            return 'B', '#2a77ad'  # Dark blue
        elif percentage >= 70:
            return 'C', '#ebcb2d'  # Dark yellow
        elif percentage >= 60:
            return 'D', '#ad6c2a'  # Dark orange
        else:
            return 'F', '#9a2a2a'  # Dark red

    grade, color = calculate_grade(hits, total)

    # Pass the grade, color, hits, and total to your template
    return render_template('after.html', grade=grade, color=color, hits=hits, total=total)

# 연락처 페이지
@app.route('/contact')
def contact():
    return render_template('contact.html')

# 카메라 조정 페이지
@app.route('/prepare')
def prepare():
    return render_template('prepare.html')


# 안내 페이지
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')



mp_drawing = mp.solutions.drawing_utils # Visualizing our poses
mp_pose = mp.solutions.pose 

seconds_old = 0
dem = 0
z = 1

def count_time(time):
    global seconds_old, dem, z
    now = datetime.datetime.now()
    seconds_new = now.strftime("%S")
    
    print(f"count_time called - now: {seconds_new}, seconds_old: {seconds_old}, dem: {dem}, z: {z}")

    if seconds_new != seconds_old:
        seconds_old = seconds_new
        dem += 1
        if dem == time + 1:
            dem = 0
            z += 1
            if z == 5:
                z = 1
    return dem, z
def send_time(time):
    socketio.emit('time', {'time': time})


@app.route('/test4')
def test4():
    return render_template('test4.html')

@app.route('/test_count_time')
def test_count_time():
    time, new_z = count_time(5)
    return {"time": time, "z": new_z}




@app.route('/detect')
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
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw pose connections
    
def draw_styled_landmarks(image, results):
    # Draw pose connections
     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    return np.concatenate([pose])



def calculateAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians =  np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0]- b[0])
    angle = np.abs ( radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360 - angle
        
    return angle

def feed():
    sequence = []
    predictions = []
    threshold = 0.5
    new_frame_width = 640
    new_frame_height = 480
    global seconds_old, dem,z
    actions = np.array(['Downdog','Warrior1','Warrior2'])
    cap= cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            ret,frame= cap.read()
            frame = cv2.flip(frame, 1) 
            # Resize frame to new dimensions
            frame = cv2.resize(frame, (new_frame_width, new_frame_height))
            
            # Make detections
            image, results = mediapipe_detection(frame, pose)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Draw landmarks
            #draw_styled_landmarks(image, results)
            
                    
            actions = np.array(['Downdog','Warrior1','Warrior2'])
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                detectedLabel = str(actions[np.argmax(res)])
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility*100, 2)]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility*100, 2)]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility*100, 2)]
                
                angle_point = []
                
                #각도 포인트
                
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                angle_point.append(right_elbow)
                
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                angle_point.append(left_elbow)
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                angle_point.append(right_shoulder)
                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                angle_point.append(left_shoulder)
                
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                angle_point.append(right_hip)
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                angle_point.append(left_hip)
                
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                angle_point.append(right_knee)
                
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                angle_point.append(left_knee)
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
     
                
                # p_score = dif_compare(keypoints, point_target)      
                
                #각도 계산
                angle = []
                
                angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                angle.append(int(angle1))
                angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                angle.append(int(angle2))
                angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                angle.append(int(angle3))
                angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                angle.append(int(angle4))
                angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                angle.append(int(angle5))
                angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                angle.append(int(angle6))
                angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                angle.append(int(angle7))
                angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                angle.append(int(angle8))

                
                #print(detectedLabel)
            

                if z==1:
                    label = "Warrior2"
                    print("이건 실행되니?1")
                    cv2.rectangle(image, (0, 450), (350, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(image, str(detectedLabel) ,(10,230),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                    time, z = count_time(5)
                    send_time(time)
                    cv2.putText(image, f"TIME: {int(time)}s", (10,250),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    
                elif z==2:
                    labels = "Warrior2"
                    
                elif z==3:
                    labels = "Warrior2"    
                    
            except:
                pass

            # Draw landmarks
            draw_styled_landmarks(image, results)
                
                
           
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(feed(), mimetype='multipart/x-mixed-replace; boundary=frame')







if __name__ == '__main__':
    socketio.run(app, Debug =True)





