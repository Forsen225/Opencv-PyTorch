
import torch
import cv2
import math

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\поток_машин2.mp4'  # Укажите путь к входному видео
output_video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\2output_video.mp4'  # Путь для сохранения видео


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_delay = 1 / fps  # Время между кадрами (в секундах)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для сохранения MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Минимальная уверенность для отображения объектов
MIN_CONFIDENCE = 0.29

# Хранение предыдущих координат центров объектов
previous_centers = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Обработка видео завершена.")
        break

   
    results = model(frame)

    
    current_centers = {}

    for _, row in results.pandas().xyxy[0].iterrows():
        confidence = row['confidence']

        
        if confidence < MIN_CONFIDENCE:
            continue

        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']

        # Вычисляем центр объекта
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        current_centers[label] = (center_x, center_y)

        # Рисуем прямоугольник вокруг объекта
        color = (0, 255, 0)  # Зеленый цвет для рамки
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Вычисляем скорость объекта
        speed = 0
        if label in previous_centers:
            prev_x, prev_y = previous_centers[label]
            distance = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            speed = distance / frame_delay  # Скорость в пикселях/сек

       
        text = f"{label} {confidence:.2f} Speed: {speed:.2f}px/s"
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  
    previous_centers = current_centers

  
    out.write(frame)

    # Отображаем текущий кадр с результатами
    cv2.imshow("Обнаружение объектов", frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Видео успешно сохранено как {output_video_path}")

# без учета скорости
'''
import torch
import cv2

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

# Путь к видео
video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\поток_машин1.mp4'  # Укажите путь к входному видео
output_video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\2output_video.mp4'  # Путь для сохранения видео

# Захват видео с указанного пути
cap = cv2.VideoCapture(video_path)

# Проверка открытия видео
if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл!")
    exit()

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Инициализация видеопотока для записи
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для сохранения MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Минимальная уверенность для отображения объектов
MIN_CONFIDENCE = 0.29

while True:
    ret, frame = cap.read()
    if not ret:
        print("Обработка видео завершена.")
        break

    # Выполняем детекцию объектов на текущем кадре
    results = model(frame)

    # Отображаем результаты на кадре
    for _, row in results.pandas().xyxy[0].iterrows():
        confidence = row['confidence']

        # Фильтрация объектов с низкой уверенностью
        if confidence < MIN_CONFIDENCE:
            continue

        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']

        # Рисуем прямоугольник вокруг объекта
        color = (0, 255, 0)  # Зеленый цвет для рамки
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Добавляем текст с меткой и уверенностью
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохраняем обработанный кадр в выходное видео
    out.write(frame)

    # Отображаем текущий кадр с результатами
    cv2.imshow("Обнаружение объектов", frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Видео успешно сохранено как {output_video_path}")
'''
# Для фото 
'''
import torch

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

# Выполнение детекции объектов на изображении
results = model('C:\\Users\\Admin\\Desktop\\Ai_Opencv\\velolude.jpg')

# Сохранение результата в файл
results.save()  # Сохранит результат в той же директории, где находится скрипт

# Вывод результатов в консоль
print(results.pandas().xyxy[0])  # Печатает таблицу с координатами объектов

'''