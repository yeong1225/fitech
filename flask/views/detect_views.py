from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp


dbp = Blueprint("detect", __name__, template_folder="templates")





# # Flask 앱 라우트 정의
# @dbp.route('/detect')
# def detect():
#     return render_template('detect.html')

# # 웹캠에서 영상을 읽기 위한 OpenCV VideoCapture 객체 생성
# cap = cv2.VideoCapture(0)

# # Mediapipe Hand 객체 생성
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hand()

# # 브라우저로 영상 전송하기 위한 제너레이터 함수
# def generate_frames():
#     while True:
#         # 웹캠에서 프레임 읽기
#         success, frame = cap.read()
#         if not success:
#             break

#         # Mediapipe Hand를 이용하여 손 인식 수행
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         # 손 인식 결과를 화면에 표시
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for landmark in hand_landmarks.landmark:
#                     h, w, c = frame.shape
#                     cx, cy = int(landmark.x * w), int(landmark.y * h)
#                     cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

#         # 프레임을 JPEG 형식으로 인코딩하여 전송
#         ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# @dbp.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')