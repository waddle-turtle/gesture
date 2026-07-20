import cv2
import mediapipe as mp
import subprocess
import numpy as np
import threading
import time
import serial
from flask import Flask, Response, jsonify


# =========================================================
# Flask 初始化
# =========================================================
app = Flask(__name__)


# =========================================================
# Arduino 序列連線設定
# =========================================================
ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUD_RATE = 9600

arduino = None
serial_lock = threading.Lock()


def connect_arduino():
    """
    連接 Arduino。

    Arduino 開啟序列埠時通常會重新啟動，
    因此連線後等待 2 秒再開始傳送指令。
    """
    global arduino

    try:
        arduino = serial.Serial(
            port=ARDUINO_PORT,
            baudrate=ARDUINO_BAUD_RATE,
            timeout=1,
            write_timeout=1
        )

        time.sleep(2)

        # 清除 Arduino 啟動時輸出的選單與 SYSTEM_READY
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()

        print(f"成功連線至 Arduino：{ARDUINO_PORT}")

    except serial.SerialException as error:
        arduino = None
        print(f"警告：Arduino 連線失敗：{error}")


connect_arduino()


# =========================================================
# MediaPipe Hands 初始化
# =========================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================================================
# 全域影像變數
# =========================================================
latest_camera_frame = None
latest_output_frame = None

camera_frame_lock = threading.Lock()
output_frame_lock = threading.Lock()


# =========================================================
# 手勢與系統狀態
# =========================================================
current_finger_count = 0
active_motor_port = -1
current_action_cmd = "IDLE"

# 手勢穩定判斷
candidate_finger_count = -1
candidate_frame_count = 0

# 需要連續辨識相同幀數才觸發
STABLE_FRAMES_REQUIRED = 5

# 上一次已傳送的手勢
last_sent_finger_count = -1

# 避免動作尚未結束就快速傳送下一筆
last_command_time = 0.0
COMMAND_COOLDOWN_SECONDS = 1.5

state_lock = threading.Lock()


# =========================================================
# Arduino 指令映射
# =========================================================
ACTION_MAPPING = {
    1: {
        "command": "1",
        "action_name": "WAVE_HAND",
        "motor_port": 12
    },
    2: {
        "command": "2",
        "action_name": "JUMP_BODY",
        "motor_port": 15
    }
}


# =========================================================
# 傳送指令給 Arduino
# =========================================================
def send_arduino_command(command):
    """
    將一筆完整指令傳給 Arduino。

    Arduino 使用 readStringUntil('\\n')，
    所以每筆指令最後一定要加上換行符號。
    """
    global arduino

    if arduino is None or not arduino.is_open:
        print("Arduino 未連線，無法傳送指令")
        return False

    message = f"{command}\n"

    try:
        with serial_lock:
            arduino.write(message.encode("utf-8"))
            arduino.flush()

        print(f"已傳送 Arduino 指令：{command}")
        return True

    except (serial.SerialException, serial.SerialTimeoutException) as error:
        print(f"Arduino 指令傳送失敗：{error}")
        return False


# =========================================================
# 相機背景執行緒
# =========================================================
def camera_thread():
    global latest_camera_frame

    command = [
        "rpicam-vid",
        "-t", "0",
        "--inline",
        "--codec", "mjpeg",
        "--nopreview",
        "--width", "1280",
        "--height", "720",
        "--framerate", "60",
        "-o", "-"
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

    except FileNotFoundError:
        print("錯誤：找不到 rpicam-vid，請確認相機套件已安裝")
        return

    byte_buffer = b""

    while True:
        if process.stdout is None:
            time.sleep(0.1)
            continue

        chunk = process.stdout.read(16 * 1024)

        if not chunk:
            print("警告：相機影像串流中斷")
            time.sleep(0.1)
            continue

        byte_buffer += chunk

        start_index = byte_buffer.find(b"\xff\xd8")
        end_index = byte_buffer.find(b"\xff\xd9", start_index + 2)

        if start_index == -1 or end_index == -1:
            # 避免異常狀況下緩衝區持續變大
            if len(byte_buffer) > 4 * 1024 * 1024:
                byte_buffer = b""
            continue

        jpg_data = byte_buffer[start_index:end_index + 2]
        byte_buffer = byte_buffer[end_index + 2:]

        image_array = np.frombuffer(jpg_data, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is not None:
            with camera_frame_lock:
                latest_camera_frame = frame


# =========================================================
# 計算伸直手指數量
# =========================================================
def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    count = 0

    # 食指
    if landmarks[8].y < landmarks[6].y:
        count += 1

    # 中指
    if landmarks[12].y < landmarks[10].y:
        count += 1

    # 無名指
    if landmarks[16].y < landmarks[14].y:
        count += 1

    # 小指
    if landmarks[20].y < landmarks[18].y:
        count += 1

    # 拇指
    thumb_tip_distance = np.hypot(
        landmarks[4].x - landmarks[17].x,
        landmarks[4].y - landmarks[17].y
    )

    thumb_joint_distance = np.hypot(
        landmarks[3].x - landmarks[17].x,
        landmarks[3].y - landmarks[17].y
    )

    if thumb_tip_distance > thumb_joint_distance:
        count += 1

    return count


# =========================================================
# 重設手勢觸發狀態
# =========================================================
def reset_gesture_state(status_text):
    global current_finger_count
    global candidate_finger_count
    global candidate_frame_count
    global last_sent_finger_count
    global active_motor_port
    global current_action_cmd

    with state_lock:
        current_finger_count = 0
        candidate_finger_count = -1
        candidate_frame_count = 0

        # 手離開或比出無效手勢後，
        # 允許相同手勢下次重新觸發
        last_sent_finger_count = -1

        active_motor_port = -1
        current_action_cmd = status_text


# =========================================================
# 分析手勢並觸發戲偶動作
# =========================================================
def analyze_puppet_gesture(hand_landmarks):
    global current_finger_count
    global candidate_finger_count
    global candidate_frame_count
    global last_sent_finger_count
    global active_motor_port
    global current_action_cmd
    global last_command_time

    detected_count = count_fingers(hand_landmarks)

    with state_lock:
        current_finger_count = detected_count

        # 3～5 根手指目前沒有動作，直接解除觸發鎖
        if detected_count not in ACTION_MAPPING:
            candidate_finger_count = detected_count
            candidate_frame_count = 0
            last_sent_finger_count = -1
            active_motor_port = -1
            current_action_cmd = "NO_ACTION"
            return

        # 判斷目前手勢是否連續穩定
        if detected_count == candidate_finger_count:
            candidate_frame_count += 1
        else:
            candidate_finger_count = detected_count
            candidate_frame_count = 1

        current_action_cmd = (
            f"VERIFY {detected_count}: "
            f"{candidate_frame_count}/{STABLE_FRAMES_REQUIRED}"
        )

        # 尚未達到穩定幀數
        if candidate_frame_count < STABLE_FRAMES_REQUIRED:
            return

        # 同一手勢持續維持時，只觸發一次
        if detected_count == last_sent_finger_count:
            action = ACTION_MAPPING[detected_count]
            active_motor_port = action["motor_port"]
            current_action_cmd = f"{action['action_name']} / HOLD"
            return

        current_time = time.monotonic()

        # 防止 Arduino 動作期間收到太密集的命令
        if current_time - last_command_time < COMMAND_COOLDOWN_SECONDS:
            current_action_cmd = "WAIT_MOTOR"
            return

        action = ACTION_MAPPING[detected_count]

        command = action["command"]
        action_name = action["action_name"]
        motor_port = action["motor_port"]

    # 實際序列傳送放在 state_lock 外，
    # 避免序列埠等待時阻塞畫面狀態讀取
    send_success = send_arduino_command(command)

    with state_lock:
        if send_success:
            last_sent_finger_count = detected_count
            last_command_time = time.monotonic()
            active_motor_port = motor_port
            current_action_cmd = action_name

            print(
                f"手勢觸發：{detected_count} 根手指 "
                f"→ {action_name} "
                f"→ PCA9685 通道 {motor_port}"
            )
        else:
            active_motor_port = -1
            current_action_cmd = "SERIAL_ERROR"


# =========================================================
# 影像辨識背景執行緒
# =========================================================
def vision_thread():
    global latest_output_frame

    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    hand_missing_frames = 0
    HAND_MISSING_RESET_FRAMES = 5

    while True:
        with camera_frame_lock:
            frame = (
                latest_camera_frame.copy()
                if latest_camera_frame is not None
                else None
            )

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - fps_start_time

            if elapsed_time > 0:
                fps = round(10 / elapsed_time, 1)

            fps_start_time = current_time

        # 每兩幀執行一次 MediaPipe，降低樹莓派負擔
        if frame_count % 2 == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_missing_frames = 0

                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                analyze_puppet_gesture(hand_landmarks)

            else:
                hand_missing_frames += 1

                # 避免偶爾漏掉一幀就立刻解除狀態
                if hand_missing_frames >= HAND_MISSING_RESET_FRAMES:
                    reset_gesture_state("LOST_HAND")

        with state_lock:
            finger_text = current_finger_count
            action_text = current_action_cmd
            motor_text = active_motor_port

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Fingers: {finger_text}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"CMD: {action_text}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Motor: {motor_text}",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        encode_success, encoded_frame = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )

        if encode_success:
            with output_frame_lock:
                latest_output_frame = encoded_frame.tobytes()


# =========================================================
# Flask MJPEG 串流
# =========================================================
def generate_frames():
    while True:
        with output_frame_lock:
            frame_bytes = latest_output_frame

        if frame_bytes is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.01)


# =========================================================
# Flask 路由
# =========================================================
@app.route("/")
def index():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_command")
def get_command():
    with state_lock:
        response_data = {
            "finger_count": current_finger_count,
            "active_motor": active_motor_port,
            "action": current_action_cmd,
            "arduino_connected": (
                arduino is not None and arduino.is_open
            )
        }

    return jsonify(response_data)


# =========================================================
# 啟動背景執行緒與 Flask
# =========================================================
if __name__ == "__main__":
    threading.Thread(
        target=camera_thread,
        daemon=True
    ).start()

    threading.Thread(
        target=vision_thread,
        daemon=True
    ).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )