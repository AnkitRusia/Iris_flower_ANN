import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()


test_index = []

for i in range(10) :
    test_index.append(i)
    test_index.append(i+50)
    test_index.append(i+100)


train_target = np.delete(iris.target , test_index)
train_data = np.delete(iris.data , test_index  ,axis =0)

valid_train_target = []
for i in train_target :
    cus = [0,0,1]
    if i == 0 :
        cus = [1,0,0]
    elif i == 1 :
        cus = [0,1,0]
    
    valid_train_target.append(cus)


valid_train_target = np.array(valid_train_target)



test_target = iris.target[test_index]
test_data = iris.data[test_index]




import keras
from keras.layers import Activation, Dropout
from keras.layers.core import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(16, activation='relu', input_shape = (4,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax' ))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, valid_train_target,
          epochs=100,
          verbose = 2,
          batch_size=30)

classes = model.predict_classes(test_data,
                                batch_size = 10,
                                verbose=2)


correct = 0
for i, j in zip(classes, test_target):
    if i == j :
        correct += 1
    print(i, ' = ', j)

print("Number of correct Samples : ", correct)





