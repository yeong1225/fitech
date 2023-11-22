from flask import Flask, render_template,Response
from flask_socketio import SocketIO,emit
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from scipy import spatial
import math
from flask import jsonify
from tensorflow.keras.models import load_model
from views.main_views import bp 
model = load_model("./model/videoyoga1.h5")

if model:
    print("모델이 성공적으로 로드되었습니다.")
else:
    print("모델을 로드하지 못했습니다.")

mp_drawing = mp.solutions.drawing_utils # Visualizing our poses
mp_pose = mp.solutions.pose # Importing our pose estimation model (ex)hand,body...)
seconds_old =0
dem =  0
z =1 
app = Flask(__name__)
socketio = SocketIO(app)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)




# Blueprint를 /search 경로에 등록
app.register_blueprint(bp, url_prefix='/')




@app.route('/test')
def test():
    return 'hi'

def send_new_score(score):
    socketio.emit('new_score', {'score': score})
    
def send_time_hen(time_hen):
    socketio.emit('time_hen', {'time_hen': time_hen})

@app.route('/warrior')
def warrior():
    return render_template('warrior.html')

def generate_frames():
    global a_score
    seconds_old =0
    dem =  0
    z =1 
    new_frame_width = 640
    new_frame_height = 480
    
    cap = cv2.VideoCapture(0) 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (new_frame_width, new_frame_height))

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
                    send_new_score(a_score)
                    cv2.putText(image, str(int(a_score)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    cv2.putText(image, str(labels), (500,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    if a_score >= 70:
                        time_hen, z = count_time(10)
                        send_time_hen(time_hen)
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
                        time_hen, z = count_time(10)
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
                        time_hen, z = count_time(10)
                        cv2.rectangle(image, (0, 450), (350, 720), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, f"TIME: {int(time_hen) }" +"s", (10, 600),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                     
    
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
    
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    



def calculateAngle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def compare_pose(image,angle_point,angle_user,angle_target,labels):
    angle_user = np.array(angle_user)
    #angle_target = np.array(angle_target)
    angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    cv2.rectangle(image,(0,0), (370,40), (255,255,255), -1)
    cv2.rectangle(image,(0,40), (370,370), (255,255,255), -1)
    cv2.putText(image, str("Score:"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
    height, width, _ = image.shape   
    
    if angle_user[0] < (angle_target[0] - 15):
        #print("Extend the right arm at elbow")
        stage = stage + 1
        cv2.putText(image, str("Extend the right arm at elbow"), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5) 
        #text = "Extend the right arm at elbow"
        #text_to_speech(text)
        
    if angle_user[0] > (angle_target[0] + 15):
        #print("Fold the right arm at elbow")
        stage = stage + 1
        cv2.putText(image, str("Fold the right arm at elbow"), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5)

        
    if angle_user[1] < (angle_target[1] -15):
        #print("Extend the left arm at elbow")
        stage = stage + 1
        cv2.putText(image, str("Extend the left arm at elbow"), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)
        
    if angle_user[1] >(angle_target[1] + 15):
        #print("Fold the left arm at elbow")
        stage = stage + 1
        cv2.putText(image, str("Fold the left arm at elbow"), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)

        
    if angle_user[2] < (angle_target[2] - 15):
        #print("Lift your right arm")
        stage = stage + 1
        cv2.putText(image, str("Lift your right arm"), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)

    if angle_user[2] > (angle_target[2] + 15):
        #print("Put your arm down a little")
        stage = stage + 1
        cv2.putText(image, str("Put your arm down a little"), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)

    if angle_user[3] < (angle_target[3] - 15):
        #print("Lift your left arm")
        stage = stage + 1
        cv2.putText(image, str("Lift your left arm"), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)

    if angle_user[3] > (angle_target[3] + 15):
        #print("Put your arm down a little")
        stage = stage + 1
        cv2.putText(image, str("Put your arm down a little"), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)

    if angle_user[4] < (angle_target[4] - 15):
        #print("Extend the angle at right hip")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at right hip"), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)

    if angle_user[4] > (angle_target[4] + 15):
        #print("Reduce the angle at right hip")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle of at right hip"), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)

    if angle_user[5] < (angle_target[5] - 15):
        #print("Extend the angle at left hip")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at left hip"), (10,260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)
        

    if angle_user[5] > (angle_target[5] + 15):
        #print("Reduce the angle at left hip")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at left hip"), (10,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)

    if angle_user[6] < (angle_target[6] - 15):
        #print("Extend the angle of right knee")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle of right knee"), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)
        

    if angle_user[6] > (angle_target[6] + 15):
        #print("Reduce the angle of right knee")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at right knee"), (10,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)


    if angle_user[7] < (angle_target[7] - 15):
        #print("Extend the angle at left knee")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at left knee"), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)

    if angle_user[7] > (angle_target[7] + 15):
        #print("Reduce the angle at left knee")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at left knee"), (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
        cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
    
    if stage!=0:
        #print("FIGHTING!")
        cv2.putText(image, str("FIGHTING!"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        
        pass
    else:
        #print("PERFECT")
        cv2.putText(image, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
  
      
def Average(lst):
    return Percente(sum(lst) / len(lst))

def Percente(x):
    return int((1 - x)*100)

def diff_compare_angle(x,y):
    new_x = []
    for i,j in zip(range(len(x)),range(len(y))):
        z = np.abs(x[i] - y[j])/((x[i]+ y[j])/2)
        new_x.append(z)
        #print(new_x[i])
    return Average(new_x)

def count_time(time):
    global seconds_old, dem,z
    now = datetime.now()
    seconds_new = now.strftime("%S")
    if seconds_new != seconds_old:
        seconds_old = seconds_new
        dem = dem + 1
        if dem == time+1:
            dem =0
            z +=1
            if z==5:
                z=1
    return dem, z



if __name__ == '__main__':
    socketio.run(app, debug=True)






