
import torch
import cv2

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Admin\\Desktop\\Ai_Opencv\\yolov5n.pt')

# Инициализация веб-камеры (0 - первая камера, 1 - вторая, и т.д.)
cap = cv2.VideoCapture(0)

# Проверка открытия камеры
if not cap.isOpened():
    print("Ошибка: Не удалось получить доступ к веб-камере!")
    exit()

# Получение параметров камеры
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Вывод для сохранения обработанного видео (опционально)
output_video_path = 'C:\\Users\\Admin\\Desktop\\Ai_Opencv\\webcam_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Нажмите 'q', чтобы завершить.")

while True:
    # Читаем кадр с веб-камеры
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр!")
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

    # Сохраняем обработанный кадр (опционально)
    out.write(frame)

    # Отображаем текущий кадр с результатами
    cv2.imshow("Обнаружение объектов с веб-камеры", frame)

    # Нажмите 'q', чтобы завершить обработку
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
