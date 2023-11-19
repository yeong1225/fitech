from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from scipy import spatial
import math

# 모듈 import
#import pymysql






bp = Blueprint('main', __name__, template_folder="templates")

# Assuming calculateAngle is defined in .func and used somewhere in your code
from .func import calculateAngle
from .func import compare_pose
from .func import Average
from .func import Percente
from .func import diff_compare_angle
from .func import count_time

mp_drawing = mp.solutions.drawing_utils # Visualizing our poses
mp_pose = mp.solutions.pose # Importing our pose estimation model (ex)hand,body...)

#db = pymysql.connect(host='database-1.cdha8hslaur6.ap-northeast-2.rds.amazonaws.com', user='admin', password='vltxpzm1234', db='Fit', charset='utf8')


#cursor = db.cursor()

# @bp.route('/db')
# def show_db():
#     # SQL query 작성
#     sql = "SELECT * FROM users"
    
#     # SQL query 실행
#     cursor.execute(sql)

#     # 데이터 가져오기
#     users = cursor.fetchall()

#     # 템플릿 렌더링
#     return render_template('db.html', users=users)


@bp.route('/')
def home():
    return render_template('home.html')

@bp.route('/calendar')
def calendar():
    return render_template('calendar.html')

@bp.route('/login')
def login():
    return render_template('login.html')


@bp.route('/register')
def register():
    return render_template('register.html')


@bp.route('/contact')
def contact():
    return render_template('contact.html')


a_score = 0  # 실제 a_score 값으로 대체
labels = "Warrior Pose"  # 실제 labels 값으로 대체

@bp.route('/warrior')
def warrior():
    return render_template('warrior.html',a_score=a_score, labels=labels)


def generate_frames():
    seconds_old =0
    dem =  0
    z =1 
    
    cap = cv2.VideoCapture(0) 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape
            image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))

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
                
                
                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                        })
                
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
                
                #print(angle)
                #비교 이미지의 앵글과 앵글포인트 앵글 , 앵글 타겟...
                if z==1:
                    labels = "Mountain"
                    angle_target = np.array([173, 175, 11, 12, 172, 172, 178, 178])
                    compare_pose(image,angle_point,angle,angle_target,labels)
                    a_score = diff_compare_angle(angle,angle_target)
                    cv2.putText(image, str(int(a_score)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    cv2.putText(image, str(labels), (500,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    if a_score >= 70:
                        time_hen, z = count_time(5)
                        cv2.rectangle(image, (0, 450), (350, 720), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f"TIME: {int(time_hen) }" +"s", (10, 600),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                elif z==2:
                    labels = "Warrior2"
                    angle_target = np.array([175, 171, 99, 107, 140, 98, 175, 103])
                    compare_pose(image,angle_point,angle,angle_target,labels)
                    a_score = diff_compare_angle(angle,angle_target)
                    cv2.putText(image, str(int(a_score)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    if a_score >= 70:
                        time_hen, z = count_time(5)
                        cv2.rectangle(image, (0, 450), (350, 720), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f"TIME: {int(time_hen) }" +"s", (10, 600),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                elif z==3:
                    labels = "Warrior2"
                    angle_target = np.array([175, 171, 99, 107, 140, 98, 175, 103])
                    compare_pose(image,angle_point,angle,angle_target,labels)
                    a_score = diff_compare_angle(angle,angle_target)
                    cv2.putText(image, str(int(a_score)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    if a_score >= 70:
                        time_hen, z = count_time(5)
                        cv2.rectangle(image, (0, 450), (350, 720), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f"TIME: {int(time_hen) }" +"s", (10, 600),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)        
                
                #a_score = diff_compare_angle(angle,angle_target)
                
                #if (p_score >= a_score):
                # cv2.putText(image, str(int((1 - a_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

            # else:
                #   cv2.putText(image, str(int((1 - p_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
    
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 4),
                                 mp_drawing.DrawingSpec(color = (0,255,0),thickness = 3, circle_radius = 3)
                                  )


            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@bp.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/score')
def score():
    return 