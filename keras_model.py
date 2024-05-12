from keras import layers
from keras import models


model = models.Sequential()
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['mae'])

history = model.fit(x, y, epochs=20, batch_size=512, validation_data=(x_val, y_val))
