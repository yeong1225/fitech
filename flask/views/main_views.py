from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

# 모듈 import
#import pymysql






bp = Blueprint('main', __name__, template_folder="templates")

# Assuming calculateAngle is defined in .func and used somewhere in your code
from .func import calculateAngle

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




@bp.route('/warrior')
def warrior():
    return render_template('warrior.html')

def compare_pose(image,angle_point,angle_user):
    #compare_pose(angle_point,angle)
    angle_user = np.array(angle_user)
    angle_target1 = np.array([180, 180, 90, 90,0,0,0,0])
    angle_target = np.array([180, 180, 90, 90,0,0,90,0])
    angle_point = np.array(angle_point)
    stage = 0
    cv2.rectangle(image,(0,0), (370,40), (255,255,255), -1)
    cv2.rectangle(image,(0,40), (370,370), (255,255,255), -1)

    #cv2.putText(image, str("Score:"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
    height, width, _ = image.shape   
    


    COLOR = (0,0,0) # 흰색 
    FONT_SIZE = 30
    if angle_target[0] != 0:
        if angle_user[0] < (angle_target[0] - 15):
            #print("Extend the right arm at elbow")
            stage = stage + 1
            cv2.putText(image, str("Stretch out your right elbow"), (10,60), cv2.FONT_HERSHEY, 0.7, [0,0,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5) 
    if angle_target[0] != 0:   
        if angle_user[0] > (angle_target[0] + 15):
            #print("Fold the right arm at elbow")
            stage = stage + 1
            cv2.putText(image, str("Fold your right elbow"), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5)

    if angle_target[1] != 0:
        if angle_user[1] < (angle_target[1] -15):
                #print("Extend the left arm at elbow")
                stage = stage + 1
                cv2.putText(image, str("Stretch out your left elbow"), (10,100), cv2.FONT_HERSHEY, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)
        
    if(angle_target[1] != 0):
        if angle_user[1] >(angle_target[1] + 15):
                #print("Fold the left arm at elbow")
                stage = stage + 1
                cv2.putText(image, str("Fold your left elbow"), (10,120), cv2.FONT_HERSHEY, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)

    if(angle_target[2] != 0):
        if angle_user[2] < (angle_target[2] - 15):
                #print("Lift your right arm")
                stage = stage + 1
                cv2.putText(image, str("Lift your right arm"), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)
    if(angle_target[2] != 0):
        if angle_user[2] > (angle_target[2] + 15):
                #print("Put your arm down a little")
                stage = stage + 1
                cv2.putText(image, str("Put your arm down a little"), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)
    if(angle_target[3] != 0):
        if angle_user[3] < (angle_target[3] - 15):
                #print("Lift your left arm")
                stage = stage + 1
                cv2.putText(image, str("Lift your left arm"), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)
    if(angle_target[3] != 0):
        if angle_user[3] > (angle_target[3] + 15):
                #print("Put your arm down a little")
                stage = stage + 1
                cv2.putText(image, str("Put your arm down a little"), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)
    if(angle_target[4] != 0):
        if angle_user[4] < (angle_target[4] - 15):
                #print("Extend the angle at right hip")
                stage = stage + 1
                cv2.putText(image, str("Extend the angle at right hip"), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)
    if(angle_target[4] != 0):
        if angle_user[4] > (angle_target[4] + 15):
                #print("Reduce the angle at right hip")
                stage = stage + 1
                cv2.putText(image, str("Reduce the angle of at right hip"), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
                cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)
    if(angle_target[5] != 0):
        if angle_user[5] < (angle_target[5] - 15):
            #print("Extend the angle at left hip")
            stage = stage + 1
            cv2.putText(image, str("Extend the angle at left hip"), (10,260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)

    if(angle_target[5] != 0):
        if angle_user[5] > (angle_target[5] + 15):
            #print("Reduce the angle at left hip")
            stage = stage + 1
            cv2.putText(image, str("Reduce the angle at left hip"), (10,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)
    if(angle_target[6] != 0):
        if angle_user[6] < (angle_target[6] - 15):
            #print("Extend the angle of right knee")
            stage = stage + 1
            cv2.putText(image, str("Extend the angle of right knee"), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)
        
    if(angle_target[6] != 0):
        if angle_user[6] > (angle_target[6] + 15):
            #print("Reduce the angle of right knee")
            stage = stage + 1
            cv2.putText(image, str("Reduce the angle at right knee"), (10,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)

    if(angle_target[7] != 0):
        if angle_user[7] < (angle_target[7] - 15):
            #print("Extend the angle at left knee")
            stage = stage + 1
            cv2.putText(image, str("Extend the angle at left knee"), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
    if(angle_target[7] != 0):
        if angle_user[7] > (angle_target[7] + 15):
            #print("Reduce the angle at left knee")
            stage = stage + 1
            cv2.putText(image, str("Reduce the angle at left knee"), (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
        
        
    if stage!=0:
        cv2.putText(image, str("FIGHTING!"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        
        pass
    else:
        #print("PERFECT")
        cv2.putText(image, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        
def generate_frames():
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
                compare_pose(image,angle_point,angle)
                
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


