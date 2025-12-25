# Voice Command Control System using Custom CNN & ESP32

## 1. Giới thiệu (Introduction)

Dự án này là một hệ thống IoT điều khiển bằng giọng nói (Voice Control) hoạt động hoàn toàn Offline trên máy tính, không phụ thuộc vào Google Assistant hay API bên thứ 3.

Điểm cốt lõi của dự án là một mô hình trí tuệ nhân tạo Convolutional Neural Network (CNN) được thiết kế và huấn luyện từ đầu (from scratch) để nhận diện các câu lệnh tiếng Việt cụ thể (như "Bật đèn", "Tắt đèn") và gửi tín hiệu điều khiển xuống vi điều khiển ESP32 qua giao tiếp Serial.

## 2. Kiến trúc Mô hình AI (The Core Model)

Đây là phần quan trọng nhất của dự án. Thay vì xử lý âm thanh dạng sóng thô (Raw Waveform), hệ thống chuyển đổi tín hiệu giọng nói thành hình ảnh nhiệt **MFCC (Mel-frequency cepstral coefficients)** để đưa vào mạng CNN xử lý như một bài toán phân loại ảnh.

**Quy trình xử lý (Pipeline):**

- **Input**: Âm thanh thu từ Microphone (Sample rate: 22050Hz, Duration: 1s).
- **Preprocessing**:
    - Trích xuất đặc trưng MFCC (13 coefficients).
    - Padding/Truncating để chuẩn hóa kích thước đầu vào về `(13, 44, 1)`.
- **Model Architecture (CNN)**:
    - **Layer 1**: `Conv2D` (32 filters) + `BatchNormalization` + `MaxPooling2D`.
    - **Layer 2**: `Conv2D` (64 filters) + `BatchNormalization` + `MaxPooling2D`.
    - **Classifier**: `Flatten` -> `Dense` (64 neurons, ReLU) -> `Dropout` (0.3) -> `Output` (Softmax).
- **Output**: Phân loại ra 3 nhãn: `bat_den`, `tat_den`, `nen` (tạp âm).
- **Hiệu năng**: Mô hình đạt độ tin cậy > 98% trên tập test và hoạt động ổn định trong môi trường phòng thực tế.

## 3. Yêu cầu cài đặt (Requirements)

### Phần cứng
- 1x ESP32-CAM (hoặc ESP32 thường).
- 1x Cáp Micro USB (có chức năng truyền dữ liệu).
- PC/Laptop có Microphone.

### Phần mềm & Thư viện
Cài đặt các thư viện Python cần thiết:
```shell
pip install tensorflow numpy scipy librosa sounddevice pyserial
```

## 4. Hướng dẫn sử dụng (Usage Steps)

### Bước 1: Thu thập dữ liệu giọng nói
Chạy script để tự thu âm giọng nói của chính bạn cho bộ dữ liệu (Dataset):
```shell
python record_data.py
```
- Nhập nhãn muốn thu (ví dụ: `bat_den`).
- Nhấn `Enter` để thu từng mẫu (nên thu khoảng 50-100 mẫu cho mỗi lệnh để model học tốt nhất).

### Bước 2: Huấn luyện mô hình (Training)
Sau khi đã có dữ liệu trong thư mục `dataset/`, chạy lệnh train:
```shell
python train.py
```
- Model sẽ được lưu vào: `models/my_voice_model.keras`
- Nhãn sẽ được lưu vào: `models/classes.npy`

### Bước 3: Nạp code cho ESP32
- Sử dụng PlatformIO hoặc Arduino IDE.
- Nạp firmware trong thư mục `esp32_firmware/` vào mạch.
> **Lưu ý:** Rút dây GPIO 0 ra khỏi GND sau khi nạp xong nếu dùng ESP32-CAM.

### Bước 4: Chạy hệ thống
Kết nối ESP32 với máy tính và chạy chương trình chính:
```shell
python main.py
```
- Hệ thống sẽ lắng nghe liên tục.
- Khi phát hiện lệnh "Bật đèn" (độ tin cậy > 80%), tín hiệu `1` sẽ được gửi qua Serial.

## 5. Cấu trúc thư mục (Project Structure)
```
📂 MyVoiceProject/
├──  dataset/              # Chứa dữ liệu âm thanh thu âm (.wav)
├──  models/               # Chứa file model .keras và file nhãn .npy
├──  esp32_firmware/       # Code C++ cho ESP32 (PlatformIO)
│   └── src/main.cpp
├──  record_data.py        # Tool thu thập dữ liệu
├──  train.py              # Script huấn luyện mô hình CNN
├──  main.py               # Chương trình chính (Real-time recognition)
├──  requirements.txt      # Các thư viện cần thiết
└──  README.md             # Tài liệu dự án
```

## 6. Source Code ESP32
Dưới đây là mã nguồn nạp cho vi điều khiển ESP32:
```cpp
#include <Arduino.h>
#include "driver/rtc_io.h"

#define FLASH_PIN 4       // Chân điều khiển đèn Flash (ESP32-CAM)
#define RED_LED   33      // Chân đèn LED đỏ tích hợp (thường active LOW)

void setup() {
  Serial.begin(115200);
  
  // Tắt tính năng giữ trạng thái GPIO của RTC (cần thiết cho ESP32-CAM)
  rtc_gpio_hold_dis(GPIO_NUM_4);

  pinMode(FLASH_PIN, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(FLASH_PIN, LOW);
  digitalWrite(RED_LED, HIGH); // LED đỏ tắt ở mức cao
}

void loop() {
  
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // Bỏ qua ký tự xuống dòng
    if (cmd == '\r' || cmd == '\n') return;

    if (cmd == '1') {
      // Lệnh BẬT ĐÈN
      digitalWrite(FLASH_PIN, HIGH); 
      digitalWrite(RED_LED, LOW);    
    } 
    else if (cmd == '0') {
      // Lệnh TẮT ĐÈN
      digitalWrite(FLASH_PIN, LOW);  
      digitalWrite(RED_LED, HIGH);   
    }
  }
}
```
---

### Tác giả

**Hà Quang Chương**
- **Sinh viên**: Đại học Điện Lực (EPU)
- **Khoa**: Công nghệ Điện tử & Viễn thông
- **Lớp**: D17DT&KTMT1
- **Email**: `haquangchuong28@gmail.com`

*Dự án này được phát triển nhằm mục đích học tập và nghiên cứu cá nhân. Mọi đóng góp và phản hồi xin gửi về email tác giả.*