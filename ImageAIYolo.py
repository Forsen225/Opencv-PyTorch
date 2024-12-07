
import torch
'''
# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

# Выполнение детекции объектов на изображении
results = model('C:\\Users\\Admin\\Desktop\\Ai_Opencv\\i.webp')

# Сохранение результата
results.save()

# Вывод результатов в консоль
print(results.pandas().xyxy[0])
'''
import torch
import cv2

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

# Путь к видео
video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\document_5377495910123068876.mp4'  # Укажите путь к видео
output_video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\output_video.avi'  # Путь для сохранения результата

# Захват видео с указанного пути
cap = cv2.VideoCapture(video_path)

# Проверка открытия видео
if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл!")
    exit()

# Получение параметров видео для сохранения
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Инициализация видеопотока для записи
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для сохранения (например, XVID)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Обработка видео завершена.")
        break

    # Выполняем детекцию объектов на текущем кадре
    results = model(frame)

    # Отображаем результаты на кадре
    for _, row in results.pandas().xyxy[0].iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Рисуем прямоугольник вокруг объекта
        color = (0, 255, 0)  # Зеленый цвет для рамки
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Добавляем текст с меткой и уверенностью
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохраняем обработанный кадр в выходной файл
    out.write(frame)

    # Отображаем текущий кадр с результатами
    cv2.imshow("Обнаружение объектов", frame)

    # Нажмите 'q', чтобы остановить обработку
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
