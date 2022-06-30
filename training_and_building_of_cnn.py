import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

Batch_Size=32
channels=3
EPOCHS=30
data_set=tf.keras.preprocessing.image_dataset_from_directory("C:\\Users\HP\\PycharmProjects\\pythonProject\\captured_images",seed=123,shuffle=True,image_size=(340,380)
                                                             ,batch_size=Batch_Size)
class_name=data_set.class_names


def get_data_partitions(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=5000):
    assert (train_split+test_split+val_split)==1
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    return train_ds,val_ds,test_ds


train_ds,val_ds,test_ds=get_data_partitions(data_set)
train_ds=train_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(340,380),
    layers.experimental.preprocessing.Rescaling(1.0/255)])

data_augmentation=tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
layers.experimental.preprocessing.RandomRotation(0.2)])
train_ds=train_ds.map(
    lambda x,y: (data_augmentation(x,training=True),y)).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape=(Batch_Size,340,380,channels)
n_classes=10
model=models.Sequential([resize_and_rescale,
                         layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape),
                         layers.MaxPooling2D((2,2)),
                         layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
                         layers.MaxPooling2D((2,2)),
                         layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
                         layers.MaxPooling2D((2,2)),
                         layers.Conv2D(64,(3,3),activation='relu'),
                         layers.MaxPooling2D((2,2)),
                         layers.Conv2D(64,(3,3),activation='relu'),
                         layers.MaxPooling2D((2,2)),
                         layers.Flatten(),
                         layers.Dense(64,activation='relu'),
                         layers.Dense(n_classes,activation='softmax')
                         ])


model.build(input_shape=input_shape)
#model.summary()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(train_ds,batch_size=Batch_Size,validation_data=val_ds,verbose=1,epochs=EPOCHS)

scores=model.evaluate(test_ds)

model.save("C:\\Users\\HP\\Desktop\\model\\VSK.h5")
model.save("C:\\Users\\HP\\Desktop\\krishna")

# def predict(model,img):
#     img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#     img_array=tf.expand_dims(img_array,0)
#     predictions=model.predict(img_array)
#     predicted_class=class_name[np.argmax(predictions[0])]
#     confidence=round(100*(np.max(predictions[0])),2)
#     return predicted_class,confidence

