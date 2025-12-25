import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- CẤU HÌNH ---
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "models/my_voice_model.keras"
LABEL_SAVE_PATH = "models/classes.npy"
SAMPLE_RATE = 22050
N_MFCC = 13
MAX_LEN = 44  # Tương ứng 1 giây

# Tạo folder models để lưu kết quả
os.makedirs("models", exist_ok=True)

# --- 1. HÀM XỬ LÝ ÂM THANH ---
def preprocess_audio(file_path):
    # Load file âm thanh
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Padding (nếu file ngắn hơn 1s thì chèn thêm số 0)
        if len(signal) < SAMPLE_RATE:
            padding = SAMPLE_RATE - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        else:
            signal = signal[:SAMPLE_RATE] # Cắt bớt nếu dài hơn 1s
            
        # Biến đổi thành ảnh nhiệt (MFCC)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        
        # Resize về kích thước chuẩn (13, 44)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        return mfcc
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

# --- 2. LOAD DATASET ---
print(" Đang đọc dữ liệu... ")
X = [] 
y = [] 

# Quét từng folder
for label in os.listdir(DATASET_PATH):
    label_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_dir): continue
    
    print(f"   -> Đang xử lý nhãn: {label}")
    for wav_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, wav_file)
        data = preprocess_audio(file_path)
        if data is not None:
            X.append(data)
            y.append(label)

X = np.array(X)
# Thêm 1 chiều channel (giống ảnh đen trắng) -> Shape: (Số lượng, 13, 44, 1)
X = X[..., np.newaxis] 

# Mã hóa nhãn (bat_den -> 0, tat_den -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Lưu tên nhãn lại để sau này dùng
np.save(LABEL_SAVE_PATH, le.classes_)
print(f"✅ Đã tìm thấy {len(le.classes_)} lớp: {le.classes_}")

# Chia dữ liệu: 80% học - 20% thi thử
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# --- 3. DỰNG MODEL CNN ---
print(" Đang xây dựng bộ não AI...")
model = Sequential([
    # Lớp 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, MAX_LEN, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Lớp 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    # Lớp 3
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3), # Quên bớt 30% để tránh học vẹt
    
    # Output
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAIN (HUẤN LUYỆN) ---
print(" BẮT ĐẦU TRAIN (Sẽ chạy 30 vòng)...")
history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# --- 5. LƯU KẾT QUẢ ---
model.save(MODEL_SAVE_PATH)
print(f"\n XONG! Model đã lưu tại: {MODEL_SAVE_PATH}")