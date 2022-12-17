#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


# In[2]:


from tensorflow import keras
import scipy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

from tensorflow.keras.preprocessing.image import load_img


# In[3]:


import os, os.path

from collections import defaultdict
from hashlib import md5
from pathlib import Path

import PIL
from PIL import Image

import matplotlib.image as mpimg


# ## Kitchenware Classification

# Competition Link: [Insert link gere]
# Dataset: [Insert link here]
# 
# Create the ff:
# - Brief description of the project
# - How to download the dataset
# 
# General Steps:
# - Gather the datasets using ImageDataGenerator() and flow_from_directory
# - Create the base model. Loading the Xception network and remove the dense layers (include_top=False). Since we dont want to retrain our convolutional layer, we use trainable = False
# - create the final model with the base_model, pooling and dense layer. all of this will be used for the inputs and outputs parameter for the final model
# - Train the Model
# - adjust the learning rate
# - Checkpointing
# - Adding more layers
# - Regularization and dropout
# - Data augmentation
# - Test the model

# ## Dataset

# Set the project directory

# In[4]:


dirs = {}

dirs['dataset_dir'] = './dataset'
dirs['raw_img_dir'] = './dataset/images'
dirs['sorted_img_dir'] = './dataset/sorted_images'
dirs['sorted_img_train'] = './dataset/sorted_images_train'
dirs['sorted_img_val'] = './dataset/sorted_images_val'
dirs['img_dir_test'] = './dataset/sorted_test'


# In[5]:


train_info = pd.read_csv(dirs['dataset_dir'] + '/train.csv', dtype = 'string')

train_info['filename'] = train_info['Id'].astype(str) + '.jpg'
train_info = train_info.sort_values(['label','Id'], ascending = [True, True])
train_info = train_info.reset_index(drop=True)

train_info


# **Check duplicate images**

# Run the python file for removing duplicates: dp_dedup.py

# In[6]:


image_dir = Path(dirs['raw_img_dir'])


# In[7]:


hash_dict = defaultdict(list)
for image in image_dir.glob('*.jpg'):
    with image.open('rb') as f:
        img_hash = md5(f.read()).hexdigest()
        hash_dict[img_hash].append(image)
len(hash_dict)


# In[8]:


duplicate_img = []
for k, v in hash_dict.items():
    if len(v) > 1:
        if v[0].name != v[1].name:
            duplicate_img.append(v[0])
            duplicate_img.append(v[1])
            print(v)
len(duplicate_img)


# In[9]:


plt.figure(figsize=(12,8))

for idx, img in enumerate(duplicate_img):
    im = PIL.Image.open(img)
    plt.subplot(6,2, idx+1)
    plt.imshow(im)
    plt.axis('off')
plt.show()


# Run the python file for removing duplicates: dp_dedup.py

# In[10]:


raw_img_dir = './dataset/images'
deduped_img_dir = './dataset/deduped_images'


# In[11]:


print(len(os.listdir(raw_img_dir)))


# In[12]:


print(len(os.listdir(deduped_img_dir)))


# Duplicates ar removed

# **Dataset Preparation**

# Already created two python files which handles the sorting of images dataset into train, val, and test sets with each image belong to the label it belongs to
# * dp_sort_images.py: for each image in the images dataset, copy to 'sorted_images' if it is included in the train.csv else 'sorted_test' if in the test.csv
# * dp_split_train_val.py: for each image in the sorted_images folder, split the images to 'sorted_images_train' and 'sorted_images_val' with a ratio of 0.8

# In[13]:


subd_train={}
for fn in Path(dirs['sorted_img_train'] ).glob('**/*'):
    if fn.is_file():
        key=str(fn.parent)
        subd_train[key] = subd_train.get(key, 0)+1
        
subd_val={}
for fn in Path(dirs['sorted_img_val'] ).glob('**/*'):
    if fn.is_file():
        key=str(fn.parent)
        subd_val[key]=subd_val.get(key, 0)+1


subd_test={}
for fn in Path(dirs['img_dir_test'] ).glob('**/*'):
    if fn.is_file():
        key=str(fn.parent)
        subd_test[key]=subd_test.get(key, 0)+1


# In[14]:


subd_train


# In[15]:


subd_val


# In[16]:


subd_test


# In[17]:


train_count = sum(subd_train.values())
val_count = sum(subd_val.values())
test_count = sum(subd_test.values())


# In[18]:


total_files = train_count + val_count + test_count
total_files


# ## Working with images

# In[19]:


dirs


# In[20]:


filepath = './dataset/sorted_images_train/cup/0003.jpg'

# Path to the folder containing the subdirectories
folder_path = dirs['sorted_img_train']

sample_img = {}

# Loop through each subdirectory in the folder
for subdir in os.listdir(folder_path):
    
    # Construct the path to the subdirectory
    subdir_path = os.path.join(folder_path, subdir)
    label = os.path.basename(subdir_path)
    
    # Check if the path is a directory (i.e. a subdirectory)
    if os.path.isdir(subdir_path):
        
        # Get the first file in the subdirectory
        file_path = os.path.join(subdir_path, os.listdir(subdir_path)[0])
        
        # Open the image using the PIL library
        sample_img[label] = file_path


# In[21]:


print(sample_img)


# In[22]:


# Create a figure and subplot with 2 columns
fig, axs = plt.subplots(nrows = 2, ncols=3)

row_counter = 1
# Loop through each item in the dictionary
for i, (label, image_path) in enumerate(sample_img.items()):
    # Load the image from the path
    image = mpimg.imread(image_path)
    
    if row_counter >= 4:
        # Plot the image on the subplot
        axs[1, i % 3].imshow(image)
        # Set the title of the subplot to the label
        axs[1, i % 3].set_title(label)
        axs[1, i % 3].axis('off')
    else:
        # Plot the image on the subplot
        axs[0, i % 3].imshow(image)
        # Set the title of the subplot to the label
        axs[0, i % 3].set_title(label)
        axs[0, i % 3].axis('off')
    
    row_counter = row_counter+1
# Show the plot
plt.axis('off')
plt.show()


# We just checked sample images for the 6 classes in the training dataset

# ## Pre-trained convolutional neural networks

# In[23]:


model = Xception(weights='imagenet', input_shape=(299, 299, 3))


# In[24]:


filepath = './dataset/sorted_images_train/cup/3632.jpg'
img = load_img(filepath, target_size=(299, 299))

x = np.array(img)

# Batch, check notes below
X = np.array([x])

X = preprocess_input(X)

pred = model.predict(X)
decode_predictions(pred)


# Based on the prediction of the pre-trained model, we can see that the model's best prediction is that it is a measuring cup

# #### Convolutional Networks

# Datasets

# In[25]:


dirs


# In[26]:


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(dirs['sorted_img_train'], 
                                         target_size=(150, 150), 
                                         batch_size=32, 
                                         class_mode='categorical')

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(dirs['sorted_img_val'], 
                                     target_size=(150, 150), 
                                     batch_size=32, 
                                     shuffle=False)


# In[27]:


train_ds.class_indices


# In[28]:


val_ds.class_indices


# Base Model

# In[29]:


base_model = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

base_model.trainable = False


# Final Model

# In[30]:


inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(6)(vectors)

model = keras.Model(inputs, outputs)


# Training the Model

# In[31]:


learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# In[32]:


history = model.fit(train_ds, epochs=20, validation_data=val_ds)


# ### Adjusting the learning rate

# In[33]:


def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(6)(vectors)
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[34]:


scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:

    # make_model() is a custom function that creates all of the code we've seen in previous sections, except for model.fit()
    model = make_model(learning_rate=lr)

    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history


# In[33]:


for lr, hist in scores.items():
    #plt.plot(hist['accuracy'], label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# Check on 0.0001 and 0.001 only

# In[37]:


keys = [0.0001,0.001]
scores_interest = {x:scores[x] for x in keys}


# In[40]:


for lr, hist in scores_interest.items():
    #plt.plot(hist['accuracy'], label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.ylim(0.86,0.9)
plt.xticks(np.arange(10))
plt.legend()


# We will use 0.001 for now

# In[42]:


learning_rate = 0.001


# ### Adjusting learning rate, inner_size, dropout

# In[110]:


checkpoint = keras.callbacks.ModelCheckpoint(
    './checkpoints/xception_v3_{lr}_{size}_{droprate}_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[112]:


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[113]:


input_size = 299

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10.0,
    height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
)


train_ds = train_gen.flow_from_directory(
    dirs['sorted_img_train'],
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    dirs['sorted_img_val'],
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


# In[114]:


learning_rate = [0.0001, 0.001, 0.01, 0.1]
size_inner_list = [1,2,32,64,128,256,512,1024]
droprates = list(np.arange(0, 1, 0.05).round(2))


# In[115]:


learning_rate


# In[ ]:


scores = {}
combi = 0
for lr in learning_rate:
    for size in size_inner_list:
        for droprate in droprates:
            combi = combi + 1
            
            print(f'learning_rate: {lr}, size: {size}, droprate: {droprate}')

            model = make_model(
                input_size = input_size,
                learning_rate=lr,
                size_inner=size,
                droprate=droprate
            )

            history = model.fit(train_ds,
                                epochs=50,
                                validation_data=val_ds)
                                #,callbacks=[checkpoint])
            scores[combi]['score'] = history.history
            scores[combi]['lr'] = lr
            scores[combi]['size_inner'] = size
            scores[combi]['droprate'] = droprate

    print()
    print()


# In[ ]:





# ### Checkpointing

# In[41]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[44]:


learning_rate = 0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint]
)


# In[45]:


model.save_weights('xception_v1_08_0.890.h5', save_format='h5')


# ### Adding additional layer

# In[48]:


def make_model(learning_rate=0.001, size_inner=100):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    
    outputs = keras.layers.Dense(6)(inner)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[49]:


learning_rate = 0.001

scores = {}

for size in [10, 100, 1000]:
    print(size)

    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history

    print()
    print()


# In[50]:


for size, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % size))

plt.xticks(np.arange(10))
#plt.yticks([0.78, 0.80, 0.82, 0.825, 0.83])
plt.legend()


# Choose 10 for the size

# ### Regularization and Dropout

# * Regularizing by freezing a part of the network
# * Adding dropout to our model
# * Experimenting with different values

# In[51]:


def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[52]:


learning_rate = 0.001
size = 10

scores = {}

for droprate in [0.0, 0.2, 0.5, 0.8]:
    print(droprate)

    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )

    history = model.fit(train_ds, epochs=30, validation_data=val_ds)
    scores[droprate] = history.history

    print()
    print()



# In[54]:


for droprate, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % droprate))

plt.ylim(0.8, 0.9)
plt.legend()


# In[55]:


hist = scores[0.2]
plt.plot(hist['val_accuracy'], label=0.2)

hist = scores[0.5]
plt.plot(hist['val_accuracy'], label=0.5)

plt.legend()
#plt.plot(hist['accuracy'], label=('val=%s' % droprate))


# Choose droprate = 0.2

# ### Data Augmentation

# In[56]:


dirs


# In[58]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10.0,
    height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
)

train_ds = train_gen.flow_from_directory(
    dirs['sorted_img_train'],
    target_size=(150, 150),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    dirs['sorted_img_val'],
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)


# In[59]:


learning_rate = 0.001
size = 10
droprate = 0.2

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds)


# In[62]:


hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()


# We can tune data augmentation later

# ### Training a Larger Model

# * Train a 299x299 model

# In[60]:


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[61]:


input_size = 299


# In[63]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10.0,
    height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
)


train_ds = train_gen.flow_from_directory(
    dirs['sorted_img_train'],
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    dirs['sorted_img_val'],
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


# In[64]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v2_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[65]:


learning_rate = 0.001
size = 10
droprate = 0.2

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])


# In[66]:


hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()


# ### Using the Model

# In[6]:


model = keras.models.load_model('xception_v2_1_27_0.955.h5')


# In[54]:


label_map = {'cup': 0, 'fork': 1, 'glass': 2, 'knife': 3, 'plate': 4, 'spoon': 5}


# In[8]:


image_size = (299,299)


# In[56]:


path = './dataset/sorted_test/0000.jpg'
img = load_img(path, target_size=(image_size))
img


# * Pre-process the image

# In[57]:


x = np.array(img)
X = np.array([x])
X = preprocess_input(X)


# * Get the prediction

# In[59]:


pred = model.predict(X)
pred_classes = pred.argmax(axis=-1)

print(pred[0])
print(pred_classes)


# Now let's apply it to the entire testing dataset
# 
# - first, we'll create a generator
# - then use the `evaluate` function to get accuracy 

# In[11]:


dirs


# In[18]:


import glob


# In[21]:


test_dir = dirs['img_dir_test']

img_files = glob.glob(os.path.join(test_dir,'*.jpg'))


# In[22]:


img_files


# In[60]:


def test_predict(path_to_img, image_size = (299,299)):
    
    img = load_img(path_to_img, target_size=(image_size))
    
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)
    
    pred = model.predict(X)
    pred_class = pred.argmax(axis=-1)
    
    return pred_class, pred[0],


# In[61]:


label_map


# In[62]:


image_list = []
pred_class_list = []
pred_proba_list = []

for file in img_files:
    pred_class ,pred_proba = test_predict(file, image_size)
    
    image_list.append(os.path.basename(file))
    pred_class_list.append(pred_class)
    pred_proba_list.append(pred_proba)
    
    print(f'{file} was successfully predicted')
    print()
    


# In[70]:


df_predict = pd.DataFrame(
    {
        'image': image_list,
        'class': [x[0] for x in pred_class_list],
        'pred_proba_list': pred_proba_list
    }
)

df_predict


# In[76]:


label_lookup = {y: x for x,y in label_map.items()}

df_submission = pd.DataFrame(
    {
        'Id': [x[:-4] for x in image_list],
        'label': [label_lookup[x[0]] for x in pred_class_list],
    }
)

df_submission = df_submission.sort_values(by=['Id'])
df_submission


# In[78]:


df_submission.to_csv('./dataset/submission_chan.csv', index = False)


# ### Model Format

# In[39]:


import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v2_1_27_0.955.h5')

tf.saved_model.save(model, 'converted_model')


# In[38]:


import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

host = 'localhost:8500'
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


# In[3]:


from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299, 299))
path = './dataset/sorted_test/0002.jpg'
X = preprocessor.from_path(path)


# In[4]:


def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

pb_request = predict_pb2.PredictRequest()
pb_request.model_spec.name = 'converted_model'
pb_request.model_spec.signature_name = 'serving_default'
pb_request.inputs['input_35'].CopyFrom(np_to_protobuf(X))


# In[34]:


pb_response = stub.Predict(pb_request, timeout=20.0)

pred = list(pb_response.outputs['dense_26'].float_val)


# In[35]:


pred


# In[36]:


label_map = {'cup': 0, 'fork': 1, 'glass': 2, 'knife': 3, 'plate': 4, 'spoon': 5}

label_lookup = {y: x for x,y in label_map.items()}

max_label = max(pred)
loc = pred.index(max_label)

pred_label = label_lookup[loc]


# In[37]:


pred_label

