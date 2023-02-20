
## Image Process + CNN Model - no slcie removal
h=w=224

def make_model():
   
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model

model = make_model()

## Load models weight
model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/imagepreprocesscnnclass.h5')



### K-means Clustering Seg + CNN - No slice removal

def make_model():
   
    model = tf.keras.models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()

model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/kmeans-cluster-seg-cnn-classif.h5')

### UNet Seg + CNN - No slice removal
## Same as the Previous Architecture 
model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-BatchNorm-CNN-model.h5')



### UNet Seg + CNN - with slice removal

def make_model():
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16, (3, 3), input_shape=(h, w, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32, (3, 3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 5
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

model = make_model()

model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-seg-sliceremove-cnn-class.h5')


### CNN model with slice removal
def make_model():
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16, (3, 3), input_shape=(h, w, 1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32, (3, 3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convulotional Layer 5
   #model.add(layers.Conv2D(256, (3, 3), padding="same"))
   #model.add(layers.BatchNormalization())
   #model.add(layers.ReLU())
   #model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

model = make_model()

model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/Image-Preprocess-sliceremove-cnn-class.h5')
