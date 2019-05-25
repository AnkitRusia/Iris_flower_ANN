# Iris_flower_ANN

An artificial neural network which is trained on 120 samples of iris flower data set and tested on 30 samples of testing data.
[Iris info](https://en.wikipedia.org/wiki/Iris_flower_data_set)

# Network Summary

Layer (type)                 Output Shape              Param    
=================================================================
dense_1 (Dense)              (None, 16)                80        
_________________________________________________________________
dropout_1 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                544       
_________________________________________________________________
dropout_2 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 99        
_________________________________________________________________

Total params: 723
Trainable params: 723
Non-trainable params: 0

# Files provided  :
1. Pretrained model  : IrisANN-1.0_accuracy.h5
2. Tesing data       : test_data_iris.npy
3. Testing target    : test_target_iris.npy
4. Training data     : train_data_iris.npy
5. Training target   : train_target_iris.npy

# Codes provided :
1. pretrained_model.py
2. ANN_to_classifiy_Iris.py
