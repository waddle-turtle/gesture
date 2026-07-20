import cv2
import mediapipe as mp
import subprocess
import numpy as np
import threading
import time
import serial
from flask import Flask, Response

app = Flask(__name__)

# 初始化 Arduino USB 連線 (樹莓派上的 Arduino Uno 通常會抓到 ttyACM0)
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    print("成功連線至 Arduino")
except:
    arduino = None
    print("警告：未偵測到 Arduino 連線，請確認 USB 是否插好")

# 初始化 AI 模型 (MediaPipe Hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    model_complexity=0, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 全域變數
latest_frame = None
frame_lock = threading.Lock()

current_finger_count = 0      
last_finger_count = -1      # 狀態鎖變數，避免重複發送
active_motor_port = -1        
current_action_cmd = "IDLE"   

def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)  # 使用標準 OpenCV 抓相機
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            with frame_lock: 
                latest_frame = frame
        else:
            time.sleep(0.01)
    cap.release()

threading.Thread(target=camera_thread, daemon=True).start()

# 核心演算法：判斷手指伸直數量 (1~5)
def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    count = 0
    
    if lm[8].y < lm[6].y: count += 1   
    if lm[12].y < lm[10].y: count += 1  
    if lm[16].y < lm[14].y: count += 1  
    if lm[20].y < lm[18].y: count += 1  
    
    dist_tip = np.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y)
    dist_joint = np.hypot(lm[3].x - lm[17].x, lm[3].y - lm[17].y)
    if dist_tip > dist_joint:
        count += 1
        
    return count

# 操偶指法映射與空間邏輯分析 
def analyze_puppet_gestures(hand_landmarks):
    global current_finger_count, last_finger_count, active_motor_port, current_action_cmd
    
    # 更新為你指定的 PWM 孔位
    port_mapping = {
        1: 0, 
        2: 3, 
        3: 6, 
        4: 8, 
        5: 11
    }
    
    current_finger_count = count_fingers(hand_landmarks)
    
    # 狀態鎖邏輯：只有當數字發生變化時，才處理並發送訊號
    if current_finger_count != last_finger_count:
        if current_finger_count in port_mapping:
            active_motor_port = port_mapping[current_finger_count]
            current_action_cmd = f"SEND CMD: {current_finger_count} -> MOTOR: {active_motor_port}"
            
            # 透過 USB 將數字傳給 Arduino
            if arduino is not None:
                arduino.write(str(current_finger_count).encode())
        else:
            active_motor_port = -1
            current_action_cmd = "STOP / ZERO"
            
        last_finger_count = current_finger_count

# 影像即時繪製與輸出
def generate_frames():
    global latest_frame, current_finger_count, active_motor_port, current_action_cmd
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        if frame_count % 10 == 0:
            fps = round(10 / (time.time() - fps_start_time), 1)
            fps_start_time = time.time()
            
        if frame_count % 2 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    analyze_puppet_gestures(hand_landmarks)
            else:
                current_finger_count = 0
                last_finger_count = -1
                active_motor_port = -1
                current_action_cmd = "LOST_HAND"

        h, w, _ = frame.shape
        
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Fingers: {current_finger_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        cv2.putText(frame, f"CMD: {current_action_cmd}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_command')
def get_command():
    return {
        "finger_count": current_finger_count,
        "active_motor": active_motor_port,
        "command": "RUN" if active_motor_port != -1 else "STOP"
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)