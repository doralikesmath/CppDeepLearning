from keras import layers
from keras import models
import numpy as np
import time

model = models.Sequential()
model.add(layers.Dense(2, input_dim = 2))
model.add(layers.Dense(3, activation='sigmoid'))
model.add(layers.Dense(1))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['acc'])
x_train = np.array([[0,0], [1,0], [0,1], [1,1]])
y_train = np.array([[0], [1], [1], [0]])

start = time.time()
model.fit(x_train, y_train, epochs=2000, verbose=False)
print("Training took ", time.time()-start, " seconds.")
print(model.predict(x_train))
