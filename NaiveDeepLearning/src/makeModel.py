import  os
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt


def get_data():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    x_train, x_test= x_train / 255.0, x_test / 255.0
    y_train = tf.cast(y_train, dtype=tf.int32)
    y_test = tf.cast(y_test, dtype=tf.int32)
    
#    print(x_train.shape, y_train.shape)
#    print(x_test.shape, y_test.shape)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    train_dataset = train_dataset.shuffle(10000).batch(128)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    test_dataset = test_dataset.shuffle(10000).batch(128)

    
    return train_dataset,test_dataset,x_train,y_train,x_test,y_test



def draw_Sample(x_train,y_train):
    plt.figure()
    
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(x_train[i])
        plt.ylabel(y_train[i].numpy())        
    plt.show()



def train_Save(train_dataset):
    model = keras.models.Sequential([ 
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(512, activation='relu'), #784->512
        layers.Dense(256, activation='relu'), #512->256
        layers.Dense(128, activation='relu'), #256->128
        layers.Dense(64, activation='relu'), #128->64
        layers.Dense(32, activation='relu'), #64->32
        layers.Dense(10, activation='softmax')]) #32->10   330=32*10+10
    
    print(model.summary())
    
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer = optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    model.fit_generator(train_dataset, epochs=50)
    model.save(r'D:\python\HUAWEI AI\HCIA-AI V2.0 Manual\NaiveDeepLearningModel\mnistDense.h5')
    

if __name__== '__main__':

        
    train_dataset,test_dataset,x_train,y_train,x_test,y_test=get_data()
    draw_Sample(x_train,y_train)
    train_Save(train_dataset)

