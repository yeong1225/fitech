from flask import Flask, request, redirect, send_file, url_for, render_template, session, flash, Response
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
from tensorflow.keras.models import load_model
#from views.main_views import bp 
from views.login_views import log
from views.calendar_views import log
from db_config import db  # db_config 모듈에서 db 불러오기


from tensorflow.keras.models import load_model

model = load_model("./model/videoyoga10.h5",compile=False)

if model:
    print("모델이 성공적으로 로드되었습니다.")
else:
    print("모델을 로드하지 못했습니다.")



app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'your_secret_key_here'

#app.register_blueprint(bp, url_prefix='/')
app.register_blueprint(log,url_prefix='/')



# 홈 페이지
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/static/video/warrior2.mp4')
def get_video():
    video_path = './static/video/warrior2.mp4'
    return send_file(video_path, as_attachment=True)

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

# 운동 페이지
@app.route('/exercise')
def exercise():
    return render_template('exercise.html')




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
    
# def send_feedback(feedback):
#     socketio.emit('feedback', {'feedback': feedback})
    
def send_feedback1(feedback1):
    socketio.emit('feedback1', {'feedback1': feedback1})

def send_feedback2(feedback2):
    socketio.emit('feedback2', {'feedback2': feedback2})
    
def send_feedback3(feedback3):
    socketio.emit('feedback3', {'feedback3': feedback3})
    
def send_feedback4(feedback4):
    socketio.emit('feedback4', {'feedback4': feedback4})
    
def send_feedback5(feedback5):
    socketio.emit('feedback5', {'feedback5': feedback5})

def send_feedback6(feedback6):
    socketio.emit('feedback6', {'feedback6': feedback6})

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

def diff_compare_angle(x,y):
    new_x = []
    for i,j in zip(range(len(x)),range(len(y))):
        z = np.abs(x[i] - y[j])/((x[i]+ y[j])/2)
        new_x.append(z)
        #print(new_x[i])
    return Average(new_x)

def Average(lst):
    return Percente(sum(lst) / len(lst))

def Percente(x):
    return int((1 - x)*100)


    

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
                
                angle9= abs(angle4 - angle6) #왼쪽으로 기울어짐
                angle.append(int(angle9))
                angle10 = abs(angle3 - angle5) #오른쪽으로 기울어짐
                angle.append(int(angle10))
                #print(detectedLabel)
            

                if z==1:
                    label = "Warrior2"
                    # cv2.rectangle(image, (0, 450), (350, 300), (0, 255, 0), cv2.FILLED)
                    # cv2.putText(image, str(detectedLabel) ,(10,230),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    angle_target = np.array([180, 180, 90, 90, 140, 98, 175, 100])
                    a_score = diff_compare_angle(angle,angle_target)
                    compare_pose(image,angle_point,angle,label)
                    if a_score >=75:
                        time, z = count_time(5)
                        send_time(time)
                        cv2.putText(image, f"TIME: {int(time)}s", (10,250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    
                elif z==2:
                    label = "Warrior1"
                    angle_target = np.array([176, 173, 165, 170, 122, 130, 175, 105])
                    cv2.putText(image, str(detectedLabel) ,(10,230),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    a_score = diff_compare_angle(angle,angle_target)
                    compare_pose(image,angle_point,angle,label)
                    if detectedLabel == label:
                        time, z = count_time(30)
                        send_time(time)
                        cv2.putText(image, f"TIME: {int(time)}s", (10,250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    
                elif z==3:
                    label = "Downdog"
                    cv2.rectangle(image, (0, 450), (350, 300), (0, 255, 0), cv2.FILLED)
                    cv2.putText(image, str(detectedLabel) ,(10,230),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    compare_pose(image,angle_point,angle,label)
                    if detectedLabel == label:
                        time, z = count_time(30)
                        send_time(time)
                        cv2.putText(image, f"TIME: {int(time)}s", (10,250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    
            except:
                pass

            # Draw landmarks
            draw_styled_landmarks(image, results)
                
                
           
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        
def compare_pose(image,angle_point,angle_user,labels):
    angle_user = np.array(angle_user)
    #angle_target = np.array(angle_target)
    #angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    # cv2.rectangle(image,(0,0), (370,40), (255,255,255), -1)
    # cv2.rectangle(image,(0,40), (370,370), (255,255,255), -1)
    # cv2.putText(image, str("Score:"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
    height, width, _ = image.shape   
    
    #labels = "Warrior2"
    #angle_target = np.array([180, 180, 90, 90, 140, 98, 175, 90])
    #왼쪽 무릎으로 하는 warrior2
    if labels == "Warrior2":
        angle_target = np.array([180, 180, 90, 90, 140, 98, 175, 115])
        if angle_user[2] < (angle_target[2] - 15):
            feedback1 = "오른쪽 팔을 더 드세요"
            # cv2.putText(image, str("Lift your right arm"), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)
        else:
            feedback1 = ""
        send_feedback1(feedback1)
        if angle_user[2] > (angle_target[2] + 15):
            feedback2 = "오른쪽 팔을 더 내리세요"
            send_feedback2(feedback2)
            # cv2.putText(image, str("Put your arm down a little"), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)
        else:
            feedback2 = ""
        send_feedback2(feedback2)
        if angle_user[3] < (angle_target[3] - 15):
            feedback3 = "왼쪽 팔을 더 드세요"
            send_feedback3(feedback3)
            stage = stage + 1
            cv2.putText(image, str("Lift your left arm"), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)
        else:
            feedback3 = ""
        send_feedback3(feedback3)
        if angle_user[3] > (angle_target[3] + 15):
            feedback4 = "왼쪽 팔을 더 내리세요"
            send_feedback4(feedback4)
            stage = stage + 1
            cv2.putText(image, str("Put your arm down a little"), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)
        else:
            feedback4 = ""
        send_feedback4(feedback4)
        # if angle_user[7] < (angle_target[7] - 15):
        #     stage = stage + 1
        #     feedback5 = "왼쪽 무릎을 더 세우세요"
        #     cv2.putText(image, str("Extend the angle at left knee"), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        #     cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
        # else:
        #     feedback5 = ""
        # send_feedback5(feedback5)

        if angle_user[7] > (angle_target[7] + 30):
            stage = stage + 1
            feedback6 = "오른쪽 무릎을 더 굽히세요"
            cv2.putText(image, str("Reduce the angle at left knee"), (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
        else:
            feedback6 =""
        send_feedback6(feedback6)
        
        if angle_user[9] > 10:
            #기울어짐 체크
        
  
      #labels = "DownDog"
    #angle_target = np.array([178, 177, 161, 160, 62, 62, 173, 175])
    if labels == "Downdog":
        angle_target = np.array([178, 177, 161, 160, 62, 62, 173, 175])
        user_elbow = (angle_user[0] + angle_user[1])/2  
        target_elbow = (angle_target[0] + angle_target[1])/2  
        if user_elbow < (target_elbow - 15):
            feedback1 = "팔꿈치를 펴세요"
            stage = stage + 1
            cv2.putText(image, str("Extend the right arm at elbow"), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5)
            cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)
        else:
            feedback1 = ""
        send_feedback1(feedback1)
        
        user_knee =  (angle_user[6]+ angle_user[7])/2
        target_knee = (angle_target[6] + angle_target[7])/2
        if user_knee < target_knee:
            stage = stage + 1
            feedback2 = "무릎을 펴세요"
            cv2.putText(image, str("Extend the angle of right knee"), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)  
        else:
             feedback2 = ""
        send_feedback2(feedback2)
        
    # 7 -> 오른쪽 무릎 (6)
    # 8 -> 왼쪽 (7)
    #오른쪽
    #[156, 160, 161, 162, 92, 127, 90, 175]
    #왼쪽
    #[176, 173, 165, 170, 122, 130, 175, 90]
    #[177, 175, 169, 171, 126, 145, 175, 90]
    if labels == "Warrior1":
        angle_target = np.array([176, 173, 165, 170, 122, 130, 175, 115])
        if angle_user[6] < (angle_target[6] - 15):
            #print("Extend the angle of right knee")
            stage = stage + 1
            feedback1 = "왼쪽 무릎을 세우세요."
            cv2.putText(image, str("Extend the angle of right knee"), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)
        else:
            feedback1 = ""
        send_feedback1(feedback1)

        if angle_user[7] < (angle_target[7] - 15):
            stage = stage + 1
            feedback5 = "오른쪽 무릎을 더 세우세요"
            cv2.putText(image, str("Extend the angle at left knee"), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
        else:
            feedback5 = ""
        send_feedback5(feedback5)

        if angle_user[7] > (angle_target[7] + 30):
            stage = stage + 1
            feedback6 = "오른쪽 무릎을 더 굽히세요"
            cv2.putText(image, str("Reduce the angle at left knee"), (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
        else:
            feedback6 =""
        send_feedback6(feedback6)
        
       
        
             


    if stage!=0:
        #print("FIGHTING!")
        cv2.putText(image, str("FIGHTING!"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        pass
    if stage == 0:
        cv2.putText(image, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        



@app.route('/video_feed')
def video_feed():
    return Response(feed(), mimetype='multipart/x-mixed-replace; boundary=frame')







if __name__ == '__main__':
    socketio.run(app, Debug =True)





