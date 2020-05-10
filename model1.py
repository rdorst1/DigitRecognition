from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Image data wordt omgezet om gebruikt te kunnen worden in het neurale netwerk
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')

# De data wordt gedeeld zodat de waardes(pixels) uit een getal bestaan tussen 0 en 1
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Model wordt gedefinieerd
def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Categorical_crossentropy wordt gebruikt omdat er meerdere classes zijn (0-9)
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model

model = larger_model()

# Model wordt getrained
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=200)

# Model wordt opgeslagen
print("The model has successfully trained")
model.save('test_model1.h5')
print("Saving the model as test_model1.h5")

# Model wordt geevalueerd, verwachting ~99%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])