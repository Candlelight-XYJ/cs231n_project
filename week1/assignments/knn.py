import random
import nnumpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from __future__ import print_function

%matplotlib inline
plt.rcParams['figure.figsize']=(10.0,8.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcPatams['image.cmap']='gray'

%load_ext autoreload
%autoreload 2


cifar10_dir='cs231n/datasets/cifar-10-batches-py'

try:
    del X_train,y_train
    del X-test,y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

#check for the size of datasets
print('training data shape:',X_train.shape)
print('training labels shape:',y_train.shape)
print('Test data shape:',X_test.shape)
print('Test labels shape:',y_test.shape)

#visualize some examples

classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes=len(classes)
samples_per_class=7

for y,cls in enumerate(classes):
    idxs=np.flatnonzeros(y_train == y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(sample_per_class,num_classes,plt_idx)
        plt.imshow(X_train[idx].astype('unit8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)

plt.show()



#subsamples the data for more efficient code execution in this exercise

num_training = 5000
mask=list(range(num_training))
X_train=X_train[mask]
y_train=y_train[mask]


num_test=500
mask=list(range(num_test))
X_test=X_test[mask]
y_test=y_test[mask]

#Reshape the image data into rows
X_train=np.reshape(X_train,(X_train.shape[0],-1))
X_test=np.reshape(X_test,(X_test.shape[0],-1))
print(X_train.shape,X_test.shape)

####

from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train,y_train)











