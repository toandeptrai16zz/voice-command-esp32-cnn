import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
import os
import serial 

MODEL_PATH = "models/my_voice_model.keras"
LABEL_PATH = "models/classes.npy"
FS = 22050        
SECONDS = 1       
THRESHOLD = 0.70  


ESP_PORT = 'COM3'
BAUD_RATE = 115200

# --- KẾT NỐI SERIAL ---
print(f" Đang kết nối ESP32 tại {ESP_PORT}...")
try:
    ser = serial.Serial(ESP_PORT, BAUD_RATE, timeout=1)
    ser.dtr = False 
    ser.rts = False
    
    time.sleep(2)
    print(" KẾT NỐI ESP32 THÀNH CÔNG!")
except Exception as e:
    print(f" Cảnh báo: Không kết nối được ESP32 ({e})")
    ser = None
except Exception as e:
    print(f" Cảnh báo: Không kết nối được ESP32 ({e})")
    print("-> Chương trình vẫn chạy nhưng sẽ không bật tắt đèn thật.")
    ser = None

# --- LOAD MODEL & NHÃN ---
print(" Đang load model AI...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(LABEL_PATH)
    print(f" Đã load xong! Các lệnh hỗ trợ: {classes}")
except:
    print(" Lỗi: Không tìm thấy model! Bạn đã chạy file train.py chưa?")
    exit()

# Biến lưu trữ buffer âm thanh
buffer = np.zeros(int(FS * SECONDS))

def audio_callback(indata, frames, time_info, status):
    global buffer
    # Đẩy dữ liệu mới vào cuối buffer, cuốn chiếu
    buffer = np.roll(buffer, -len(indata))
    buffer[-len(indata):] = indata[:, 0]

def predict_command():
    # 1. Kiểm tra độ ồn
    vol = np.max(np.abs(buffer))
    if vol < 0.05: 
        return None

    # 2. Xử lý âm thanh (MFCC)
    try:
        mfcc = librosa.feature.mfcc(y=buffer, sr=FS, n_mfcc=13)
        
        
        if mfcc.shape[1] < 44:
            pad = 44 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0), (0,pad)), mode='constant')
        else:
            mfcc = mfcc[:, :44]
            
        mfcc_input = mfcc[np.newaxis, ..., np.newaxis]

        
        prediction = model.predict(mfcc_input, verbose=0)
        idx = np.argmax(prediction)
        confidence = prediction[0][idx]
        label = classes[idx]

        return label, confidence
    except Exception as e:
        return None

# ---  CHƯƠNG TRÌNH CHÍNH ---
print("\n HỆ THỐNG ĐANG NGHE... (Nói 'Bật đèn' hoặc 'Tắt đèn')")
print(f"Ngưỡng nhận diện: {THRESHOLD * 100}%")

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=FS, blocksize=4096):
        while True:
            result = predict_command()
            
            if result:
                label, conf = result
                
                # Logic lọc: Chỉ nhận lệnh nếu tự tin > 95% và không phải 'nen'
                if conf > THRESHOLD and label != "nen":
                    print(f"PHÁT HIỆN: {label.upper()} ({conf*100:.1f}%)")
                    
                    # --- GỬI TÍN HIỆU XUỐNG ESP32 ---
                    if label == "bat_den":
                        print(">>>  LỆNH: BẬT ĐÈN <<<")
                        if ser: ser.write(b'1')  # Gửi số 1

                    elif label == "tat_den":
                        print(">>>  LỆNH: TẮT ĐÈN <<<")
                        if ser: ser.write(b'0')  # Gửi số 0
                    
                    # Ngủ 1.5 giây để tránh gửi lệnh liên tục
                    time.sleep(1.5)
                    buffer = np.zeros(int(FS * SECONDS))
            
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nĐã dừng hệ thống.")
    if ser: ser.close()
except Exception as e:
    print(f"Lỗi hệ thống: {e}")