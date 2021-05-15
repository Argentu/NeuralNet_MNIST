import os
path = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import cv2 as cv

#=================================================================
# загрузка пакету даних MNIST і занесення даних в змінні
(img_train, answ_train), (img_test, answ_test) = mnist.load_data()
#=================================================================
# функція нормалізації матриці (зображення)
def norm(x: np.ndarray):
    # перетворення кольрової схеми зображення у відтінки сірого
    ret = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    # ділення кожного елемента матриці на 255 щоб отримати значення в діапазоні 0-1
    ret = ret/255
    return ret
#=================================================================
# нормалізація тренувальних даних (вони і так у відтінках сірого,
# тому просто ділимо значення матриць)
img_train = img_train / 255
img_test = img_test / 255
#=================================================================
# нормалізація тренувальних і тестових відповідей
    # перетворення чисел (типу "4") у вектор (типу "[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]")
answ_train_cat = keras.utils.to_categorical(answ_train, 10)
answ_test_cat = keras.utils.to_categorical(answ_test, 10)
#=================================================================
# ініціалізація структури нейронки (вхідний шар, 2 приховані і вихідний + bias)
    # для прихованих шарів використовується функція активації ReLU, а для вихідного - SoftMax
model = keras.Sequential([
    Flatten(input_shape = (28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
#=================================================================
# створення моделі з певними параметрами
    # оптимізація навчання - алгоритм "Adam"
    # функція втрат - категоріальна крос ентропія
    # метрика - точність
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
    )
#=================================================================
# навчання моделі і його налаштування
    # тренувальний набір
    # відповіді до набору
    # розмір міні-батча (mini-batch)
    # кількість прогону тренувань
    # частина тренувальної вибірки, яка використовується, як вибірка валідації
model.fit(img_train, answ_train_cat, batch_size=30, epochs=5, validation_split=0.2)
#=================================================================
# прогонка навченої моделі по тестовій вибірці
model.evaluate(img_test, answ_test_cat)
#=================================================================
# збереження навченої моделі в файл 'IRC_MNIST.h5'
#model.save('IRC_MNIST.h5')
#=================================================================
# передача власного зображення в модель для перевірки
img = cv.imread(os.path.join(path, 'MyTestNum.png'))
img = norm(img)
img = np.expand_dims(img, axis=0)
res = model.predict(img)
print(f"Розпізнана цифра: {np.argmax(res)}")
#=================================================================