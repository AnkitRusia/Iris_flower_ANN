import numpy as np
from keras.models import load_model

##You can create test data like :
test_data = np.array([
             [5.1, 3.5, 1.4, 0.2],
             [7.0, 3.2, 4.7, 1.4],
             [6.3, 3.3, 6.0, 2.5]
            ])

test_target = [0, 1, 2]

''' OR  '''

##you can import data which I have provided
##test_data = np.load('test_data_iris.npy')
##test_target = np.load('test_target_iris.npy')

model = load_model('IrisANN-1.0_accuracy.h5')
print(model.summary())
classes = model.predict_classes(test_data)

print("predicted  :  Actual")
for i, j in zip(classes, test_target) :
    print("{:<10} : {}".format(i, j))
