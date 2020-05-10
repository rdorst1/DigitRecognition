import imageio
import numpy as np
from keras.models import load_model
model = load_model("test_model1.h5")
model2 = load_model("test_model2.h5")
model3 = load_model("test_model3.h5")

# Imgur link (5)
im = imageio.imread("https://i.imgur.com/a3Rql9C.png")

# Local files
im0 = imageio.imread('../DigitRecognition/Digits/0.png')
im1 = imageio.imread('../DigitRecognition/Digits/1.png')
im2 = imageio.imread('../DigitRecognition/Digits/2.png')
im3 = imageio.imread('../DigitRecognition/Digits/3.png')
im4 = imageio.imread('../DigitRecognition/Digits/4.png')
im5 = imageio.imread('../DigitRecognition/Digits/5.png')
im6 = imageio.imread('../DigitRecognition/Digits/6.png')
im7 = imageio.imread('../DigitRecognition/Digits/7.png')
im8 = imageio.imread('../DigitRecognition/Digits/8.png')
im9 = imageio.imread('../DigitRecognition/Digits/9.png')

alt2 = imageio.imread('../DigitRecognition/Digits/alt2.png')
alt6 = imageio.imread('../DigitRecognition/Digits/alt6.png')
alt8 = imageio.imread('../DigitRecognition/Digits/alt8.png')
alt9 = imageio.imread('../DigitRecognition/Digits/alt9.png')
niek9 = imageio.imread('../DigitRecognition/Digits/niek9.png')

gray = np.dot(im2[...,:3], [ 0.299, 0.587, 0.114])
gray = gray.reshape(1, 28, 28, 1)

gray /= 255

prediction = model.predict(gray)
print("Prediction model 1: ", prediction.argmax())

prediction2 = model2.predict(gray)
print("Prediction model 2: ", prediction2.argmax())

prediction3 = model3.predict(gray)
print("Prediction model 3: ", prediction3.argmax())
