'''
Created on Nov 4, 2019

@author: tfu
'''
import  os
from tensorflow.keras.models import load_model
from makeModel import get_data
from matplotlib import pyplot as plt
import cv2
import numpy as np



def image_reverse(input_image):
    input_image_cp = np.copy(input_image) 
    pixels_value_max = np.max(input_image_cp) 
    output_imgae = pixels_value_max - input_image_cp 
    return output_imgae


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
train_dataset,test_dataset,x_train,y_train,x_test,y_test=get_data()

model_loaded = load_model(r'D:\python\HUAWEI AI\HCIA-AI V2.0 Manual\NaiveDeepLearningModel\mnistDense.h5')


loss,acc=model_loaded.evaluate_generator(test_dataset)

print("loss:",loss)
print("accuarcy:",acc)



img = cv2.imread(r"D:\python\HUAWEI AI\3.PNG")
img=image_reverse(img)
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.gray()
img = cv2.resize(img,(28,28))/255
img = np.asarray(img,np.float32)

plt.imshow(img)
plt.show()

pred = model_loaded.predict_classes(img.reshape(1,28,28))
print(pred)
