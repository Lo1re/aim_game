from ultralytics import YOLO
import cv2

# Завантаження моделі YOLO (попередньо натренована або власна)
model = YOLO("D:/jammer/myTAttemps/yolov8n-drone.pt")  # 'n' для найлегшої моделі, замініть на 'm', 'l', 'x' для більшої точності.

# Завантаження зображення
image_path = "D:/jammer/myTAttemps/test2.jpg"  # Замість цього шляху вкажіть шлях до вашого зображення.
image = cv2.imread(image_path)

# Аналіз зображення
results = model(image)

# Вивід результатів у консоль
print(results)  # Показує знайдені об'єкти та їх координати

# Візуалізація результатів
annotated_frame = results[0].plot()  # Накласти результати на зображення
cv2.imshow("YOLO Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
