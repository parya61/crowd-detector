#  Детекция людей на видео с использованием OpenCV DNN

Простой и эффективный проект на Python, выполняющий детекцию людей на видео с помощью модуля OpenCV DNN и предобученной модели MobileNet SSD. Результат — видео с отрисованными рамками и вероятностью детекции для каждого человека.

##  Структура проекта
crowd_detector/
├── data/                  # Входное видео
│   └── crowd.mp4
├── output/                # Выходное видео
│   └── crowd_annotated.mp4
├── models/                # Весы модели MobileNet SSD
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
├── src/
│   ├── detector.py        # Логика модели
│   └── utils.py           # Отрисовка и сохранение видео
├── main.py                # Точка входа
├── requirements.txt       # Зависимости Python
├── README.md              # Этот файл
##  Требования
- Python 3.9 или выше
- OpenCV
- NumPy

Установка зависимостей:
Bash

pip install -r requirements.txt


2. Запустите скрипт:
Bash

python main.py --input data/crowd.mp4 --output output/crowd_annotated.mp4
##  Пример результата
Программа создаёт новое видео, где все обнаруженные люди обведены зелёными рамками с подписью person: 0.87, где число — вероятность.

##  Особенности
- Класс распознавания: только person (ID = 15)
- Порог уверенности: 0.5
- Используется облегчённый OpenCV DNN (не требует PyTorch или TensorFlow)
