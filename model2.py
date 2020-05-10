import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data.shape, train_labels.shape
test_data.shape, test_labels.shape

train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32')/255

test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32')/255

train_labels = to_categorical(train_labels, num_classes = 10)
test_labels = to_categorical(test_labels, num_classes = 10)

train_labels[1], test_labels[4]

partial_train_data = train_data[10000:]
partial_train_labels = train_labels[10000:]

val_data = train_data[:10000]
val_labels = train_labels[:10000]

partial_train_data.shape, partial_train_labels.shape, val_data.shape, val_labels.shape, test_data.shape, test_labels.shape

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

history = model.fit(partial_train_data, partial_train_labels, epochs = 20, batch_size = 128, validation_data = (val_data, val_labels), verbose = 2)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, 21)

plt.plot(epochs, loss, 'ko', label = 'Training Loss')
plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("The model has successfully trained")
model.save('test_model2.h5')
print("Saving the model as test_model3.h5")

test_loss, test_acc = model.evaluate(test_data, test_labels)
test_loss, test_acc