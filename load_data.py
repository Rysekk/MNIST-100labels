from keras.datasets import mnist
import tensorflow as tf
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

y_train = np.array(train_y100label).astype('uint8')
x_train = np.array(train_X100label).astype('uint8')

#test Resnet50 avec les 100 labels


#data processing

#pour le train
x_train = x_train.reshape((100,28,28,1))
x_train = x_train.repeat(3,-1)
x_train = x_train.astype('float32')/255
x_train = tf.image.resize(x_train,[32,32])
x_train = tf.convert_to_tensor(x_train)

#pour le test
#on ajoute un canneau 
x_test = test_X.reshape((10000,28,28,1))
#pour le rgb (donc = 3)
x_test = x_test.repeat(3,-1)
#on normalise
x_test = x_test.astype('float32')/255
#on reshape l'image pour quelle corresponde a l'input du resnet50 qui doit etre du 32 x 32 
x_test = tf.image.resize(x_test,[32,32])
#on convertie le train en tensor
x_test = tf.convert_to_tensor(x_test)

#pour les y on les transformes en one hot encoder
from tensorflow.keras.utils import to_categorical
y_test = to_categorical(test_y)
y_train = to_categorical(y_train)

del X_train,Y_train,train_X,train_y,train_filter,i,j,counter,temp,test_X,test_y,train_y100label,train_X100label

#implementation resnet50
from keras.layers import  Input, Dense, Flatten
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

#importation du resnet 
inp = tf.keras.Input(shape=(32,32,3))
resnet = ResNet50(include_top=False,  pooling='none', input_tensor= inp,  weights='imagenet')

#on ajoute un Dense layer et on l'ajoute comme output pour la classification 
X = Flatten()(resnet.output)
prediction = Dense(10, activation='softmax',name='class_layer')(X)

model = Model(inputs = resnet.input, outputs = prediction)

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

historique = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,verbose=1)





































