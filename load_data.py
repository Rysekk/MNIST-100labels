from keras.datasets import mnist
#Import de la base de donnée MNIST

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

    
import numpy as np

#creation des matrices pour stocker les 100 labels
train_X100label = np.zeros(shape=(100,28,28)).astype('uint8')
train_y100label = np.zeros(shape=(100)).astype('uint8')

#On selectionne dans X_train et Y_train les 100 labels 
counter = 0
for i in range(10):
    train_filter = np.where(train_y == [i])
    for j in range(10):
        X_train, Y_train = train_X[train_filter], train_y[train_filter]
        
        train_X100label[counter] = X_train[counter]
        train_y100label[counter] = Y_train[counter]
        counter+=1

#plot chaque dizaine des 100 labels selectionés
from matplotlib import pyplot
for i in range(10):
    pyplot.subplot(2,5,1+i)
    pyplot.imshow(train_X100label[i*10], cmap=pyplot.get_cmap('gray'))
    print("label num "+str(i)+": ",train_y100label[i*10])
    pyplot.show()



#Il ne reste plus qu'à les melanger pour avoir un truc uniforme et sans biais
# Using zip() + * operator + shuffle()
import random
   
# Shuffle two lists with same order
# Using zip() + * operator + shuffle()
temp = list(zip(train_y100label, train_X100label))
random.shuffle(temp)
train_y100label, train_X100label = zip(*temp)

train_y100label = np.array(train_y100label).astype('uint8')
train_X100label = np.array(train_X100label).astype('uint8')


#suppression des veleur qui ne servent plus
del X_train,Y_train,train_X,train_y,train_filter,i,j,counter,temp