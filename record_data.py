import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
import numpy as np

# --- CẤU HÌNH ---
FS = 22050        # Tần số lấy mẫu chuẩn
SECONDS = 1       # Độ dài file (1 giây)

def record_process():
    print("--- TOOL THU ÂM DATASET TỰ ĐỘNG ---")
    print("Lưu ý: Các nhãn nên đặt là: bat_den, tat_den, nen")
    
    # 1. Nhập nhãn (Label) muốn thu
    label_name = input(">> Nhập tên nhãn bạn muốn thu (ví dụ: bat_den): ").strip()
    if not label_name:
        print("Tên nhãn không được để trống!")
        return

    # Tạo thư mục
    save_path = os.path.join("dataset", label_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nĐã sẵn sàng lưu vào: {save_path}")
    print("Hướng dẫn: Nhấn ENTER để thu 1 file. Nhấn Ctrl+C để thoát/đổi nhãn.")
    
    existing_files = len(os.listdir(save_path))
    count = existing_files
    
    try:
        while True:
            input(f"\n[{count+1}] Nhấn Enter để bắt đầu thu...")
            print("🔴 ĐANG THU...", end="\r")
            
            # Thu âm
            myrecording = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1)
            sd.wait()
            
            # Kiểm tra nhanh âm lượng (tránh thu file câm)
            if np.max(np.abs(myrecording)) < 0.01:
                print("⚠️ Cảnh báo: Âm thanh quá nhỏ! Hãy nói to hơn.")
            
            # Lưu file
            filename = os.path.join(save_path, f"{label_name}_{count}.wav")
            write(filename, FS, myrecording)
            print(f"✅ Đã lưu: {filename}")
            count += 1
            
    except KeyboardInterrupt:
        print(f"\n\nĐã dừng thu cho nhãn '{label_name}'.")

if __name__ == "__main__":
    while True:
        try:
            record_process()
            cont = input("Bạn có muốn thu nhãn khác không? (y/n): ")
            if cont.lower() != 'y':
                break
        except Exception as e:
            print(f"Lỗi: {e}")
            break