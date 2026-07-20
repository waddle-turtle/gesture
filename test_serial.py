import time
import serial

try:
    arduino = serial.Serial(
        "/dev/ttyUSB0",
        9600,
        timeout=1,
        write_timeout=1
    )

    time.sleep(2)

    print("傳送1：揮手")
    arduino.write(b"1\n")
    arduino.flush()

    time.sleep(4)

    print("傳送2：跳躍")
    arduino.write(b"2\n")
    arduino.flush()

except serial.SerialException as error:
    print(f"Arduino連線失敗：{error}")