###########################################################
# expanding HAPPEI train set with image augmentation
###########################################################

import numpy as np
import pandas as pd
import csv
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data_file = open('augmented_data.csv','w+')
writer = csv.writer(data_file,quoting=csv.QUOTE_NONNUMERIC)
writer.writerow(('Image', 'Group Happiness Score'))

db = pd.read_csv('Happie_Train.csv')
happiness_scores = db['Intensity'].values
image_names = db['name'].values

# creating Keras Datagenerator with different attributes for augmentation
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


# getting image 4 dimension shape
pil_image = load_img(os.path.join('./../Happie',image_names[0]))
numpy_image = img_to_array(pil_image)
numpy_image = numpy_image.reshape((1,) + numpy_image.shape)
image_shape = numpy_image.shape
print(image_shape)
batch_size = 1  # batch to be used in regeneration
image_count = 1  # To name each image
k = 5  # no of times we want to expand origional dataset

for i in range(1, len(image_names)/batch_size+1):
    flag = 0  # Flag to take care of initializing image_array
    image_array = np.zeros(image_shape)
    for img_loc in image_names[batch_size*(i-1):batch_size*i]:
        pil_image = load_img(os.path.join('./../Happie',img_loc))
        numpy_image = img_to_array(pil_image)
        numpy_image = numpy_image.reshape((1,) + numpy_image.shape)  # Resize image to add 4th diemension (batch)
        if flag == 0:
            image_array = numpy_image
            flag = 1
        else:
            image_array = np.concatenate((image_array, numpy_image), axis=0)  # Concatenating all the images to form a


    count_aug_images = 0
    for image_batch, label_list in datagen.flow(image_array, happiness_scores[batch_size*(i-1):batch_size*i],
                                                batch_size=batch_size):
        for l in range(0, len(label_list)):
            image_name = 'img_'+str(image_count)+'.jpg'
            I = array_to_img(image_batch[l]) # without reshaping
            I.save('./aug_images/'+image_name)
            image_count += 1
            print image_name
            writer.writerow((image_name, int('%d'%(label_list[l]))))
            count_aug_images += 1

        # We are augmenting 20 times of origional no of images in our Uniform dataset
        if count_aug_images >= k*batch_size:
            break

data_file.close()
