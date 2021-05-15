import os

path = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model as lm
import numpy as np
from PIL import Image as im

def open(p: str) -> np.ndarray:
    img = im.open(os.path.join(path, p)).convert('L')
    (width, height) = img.size
    if width > 28 and height > 28:
        img = img.resize((28,28), im.ANTIALIAS)
    else:
        pass
    (width, height) = img.size
    img = list(img.getdata())
    img = np.array(img)
    img = img.reshape((height, width))
    return img

def norm(x: np.ndarray):
    ret = x / 255
    ret = np.expand_dims(ret, axis=0)
    return ret

model = lm('IRC_MNIST.h5')

img = open('MyTestNum.png')
img = norm(img)
res = model.predict(img)
print(f"Розпізнана цифра: {np.argmax(res)}")
