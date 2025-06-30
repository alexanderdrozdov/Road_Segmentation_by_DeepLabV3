# Road_Segmentation_by_DeepLabV3

Обучение нейронной сети для сегментации дорог по аэрофотснимкам

Датасеты:

https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset

Модель:
DeepLabV3+ mobilenet

Функция потерь: Dice Loss + BCE

Метрика качества: IoU


Файлы:

main.py - скрипт обучения модели

inference.py - "тестовая" программа, где используется уже обученная модель. На вход подается аэрофотоснимок, на выходе коллаж из 3 картинок: оригинального изображения, маски, и оригинального изображения с выделенными на нем дорогами

В папке tests некоторые результаты работы программы
