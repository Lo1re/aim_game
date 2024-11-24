import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

# Завантаження моделі
model = YOLO("D:/jammer/test/yolov8n-drone.pt")

# Підключення до камери
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

# Завантаження зображення дрона
drone_image_path = "D:/jammer/test/drone.png"
drone_image = cv2.imread(drone_image_path, cv2.IMREAD_UNCHANGED)

if drone_image is None:
    print("Не вдалося завантажити зображення дрона")
    exit()

# Розділення каналів (RGBA)
drone_rgb = drone_image[:, :, :3]
drone_alpha = drone_image[:, :, 3] if drone_image.shape[2] == 4 else np.ones_like(drone_image[:,:,0])

# Нормалізація альфа-каналу
drone_alpha = cv2.normalize(drone_alpha, None, 0, 1, cv2.NORM_MINMAX)

# Масштаб дрона
drone_scale = 0.2  # Зменшив масштаб для кращого руху
drone_h, drone_w = int(drone_rgb.shape[0] * drone_scale), int(drone_rgb.shape[1] * drone_scale)
drone_rgb = cv2.resize(drone_rgb, (drone_w, drone_h))
drone_alpha = cv2.resize(drone_alpha, (drone_w, drone_h))

# Параметри руху
x_pos = 100
y_pos = 100
x_speed = 3
y_speed = 2
angle = 0
radius = 100
center_x = 300
center_y = 200

# Час початку
start_time = time.time()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Не вдалося отримати кадр")
        break

    frame_h, frame_w = frame.shape[:2]
    current_time = time.time() - start_time

    # Оновлення позиції дрона (комбінація кругового та синусоїдального руху)
    x_pos = center_x + int(radius * math.cos(current_time * 0.5)) + int(50 * math.sin(current_time * 2))
    y_pos = center_y + int(radius * math.sin(current_time * 0.5)) + int(50 * math.cos(current_time * 3))

    # Обмеження позиції в межах кадру
    x_pos = max(0, min(x_pos, frame_w - drone_w))
    y_pos = max(0, min(y_pos, frame_h - drone_h))

    # Вставка дрона в кадр
    roi = frame[y_pos:y_pos + drone_h, x_pos:x_pos + drone_w]
    
    # Накладання дрона
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - drone_alpha) + drone_rgb[:, :, c] * drone_alpha

    frame[y_pos:y_pos + drone_h, x_pos:x_pos + drone_w] = roi

    # Детекція об'єктів за допомогою YOLO
    results = model(frame)
    
    # Обробка результатів детекції
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Отримання координат обмежувального прямокутника
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            
            # Малювання прямокутника з відображенням впевненості
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Drone: {confidence:.2f}', 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)

    # Виведення кадру
    cv2.imshow("Drone Detection Game", frame)

    # Вихід за клавішею 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()