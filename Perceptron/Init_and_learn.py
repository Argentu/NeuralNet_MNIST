import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image as im
# =================================================================
# занесення в змінну "path" шляху до директорії зі скриптом
path = os.getcwd()
# =================================================================
# це треба щоб TensorFlow не кидав щоразу попередження, що не може застосувати GPU
# (можна видалити, якщо ви встановили програму CUDA)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# =================================================================
# загрузка пакету даних MNIST і занесення даних в змінні
(img_train, answ_train), (img_test, answ_test) = mnist.load_data()
# =================================================================
# функція нормалізації матриці (зображення)
def norm(x: np.ndarray):
    # ділення кожного елемента матриці на 255 щоб отримати значення в діапазоні 0-1
    ret = x / 255

    # додавання нової осі для розширення масиву
    ret = np.expand_dims(ret, axis=0)
    return ret
# =================================================================
# функція для завантаження картинки через PIL в потрібному вигляді (+ стандартизує розмір)
def open(p: str):
    '''
    p - строкова змінна, яка приймає лише назву картинки
    і розширення (img.png або img.jpg)

    картинка має знаходитись в одному файлі зі скриптом
    '''
    img = im.open(os.path.join(path, p)).convert('L')
    (width, height) = img.size
    if width > 28 and height > 28:
        img = img.resize((28, 28), im.ANTIALIAS)
    else:
        pass
    (width, height) = img.size
    img = list(img.getdata())
    img = np.array(img)
    img = img.reshape((height, width))
    return img
# =================================================================
# нормалізація тренувальних даних
# матриці в цьому наборі вже підготовані, а функції нормалізації за це відповідає "np.expand_dims"
img_train = img_train / 255
img_test = img_test / 255
# =================================================================
# нормалізація тренувальних і тестових відповідей
# перетворення чисел (типу "4") у вектор (типу "[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]")
answ_train_cat = keras.utils.to_categorical(answ_train, 10)
answ_test_cat = keras.utils.to_categorical(answ_test, 10)
# =================================================================
# ініціалізація структури нейронки (вхідний шар, 2 приховані і вихідний + bias)
# для прихованих шарів використовується функція активації ReLU, а для вихідного - SoftMax
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
# =================================================================
# створення моделі з певними параметрами
# оптимізація навчання - алгоритм "NAdam"
# функція втрат - категоріальна крос ентропія
# метрика - точність
model.compile(
    optimizer='nadam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# =================================================================
# навчання моделі і його налаштування
# тренувальний набір
# відповіді до набору
# розмір міні-батча (mini-batch)
# кількість прогону тренувань
# частина тренувальної вибірки, яка використовується, як вибірка валідації
model.fit(img_train, answ_train_cat, batch_size=20, epochs=10, validation_split=0.2)
# =================================================================
# прогонка навченої моделі по тестовій вибірці
model.evaluate(img_test, answ_test_cat)
# =================================================================
# збереження навченої моделі в файл 'IRC_MNIST.h5'
model.save('IRC_MNIST.h5')
# =================================================================
# передача власного зображення в модель для перевірки
img = open('MyTestNum1.png')
img = norm(img)
res = model.predict(img)
print(f"Розпізнана цифра: {np.argmax(res)}")
# =================================================================
