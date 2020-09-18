#basic CNN model on identifying handwritten numbers
#load libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

#load and preprocessing data  
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
number_names = [0,1,2,3,4,5,6,7,8,9]

#find the shapes and number of images in the set. each image is 28x28
'''
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))
'''

#dividing images of both sets by 255 because the images have pixel values that range from 0 to 255. They should be scaled to an easier range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#creating a CNN model with 2 dense layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

#train the model and checking test accuracy and loss
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('Test accuracy: ', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(np.argmax(predictions[0]))
print(test_labels[0])

#vertifying predictions by plotting the first 15 images . correct prediction labels are purple while incorrect labels are pink
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'purple'
  else:
    color = 'pink'

  plt.xlabel("{} {:2.0f}% ({})".format(number_names[predicted_label],
                                100*np.max(predictions_array),
                                number_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('pink')
  thisplot[true_label].set_color('purple')

'''
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
'''

#using the model
image = test_images[7]
print(image.shape)

#adding the image to a batch
img_batch = (np.expand_dims(image,0))
print(img_batch.shape)
#(1, 28, 28)

#predicting
predictions_single = probability_model.predict(img_batch)
print(predictions_single)
'''
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), number_names, rotation=45)'''
#Plotting

#grabbing the predictions
np.argmax(predictions_single[0])
