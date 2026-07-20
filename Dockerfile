# 使用輕量級的 Python 3.9 作為基礎基底
FROM python:3.9-slim

# 設定容器內的工作目錄
WORKDIR /app

# 安裝 OpenCV 運作所需的底層系統套件
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 將 requirements.txt 複製到容器內並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將專案的所有程式碼複製到容器內
COPY . .

# 宣告對外開放 Flask 的 5000 port
EXPOSE 5000

# 容器啟動時執行的預設指令 (指向你的主程式)
CMD ["python", "lapcamera.py"]