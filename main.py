import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from dataPreprocess import process_scan, train_prepare, validation_prepare
from model import get_model


normal_scan_paths = []
for x in os.listdir('MosMedData/CT-0'):
    normal_scan_paths.append(os.path.join(os.getcwd(), 'MosMedData/CT-0', x))

abnormal_scan_paths = []
for x in os.listdir('MosMedData/CT-23'):
    abnormal_scan_paths.append(os.path.join(os.getcwd(), 'MosMedData/CT-23', x))


''' Each scan is resized across height, width, and depth and rescaled '''
normal_scans = np.array([process_scan(path=path) for path in normal_scan_paths])
# print(normal_scans.shape)  # (100, 128, 128, 64)

abnormal_scans = np.array([process_scan(path=path) for path in abnormal_scan_paths])
# print(abnormal_scans.shape)  # (100, 128, 128, 64)


''' For the CT scans having presence of viral pneumonia assign 1, for the normal ones assign 0 '''
normal_labels = np.array([0 for _ in range(len(normal_scan_paths))])
abnormal_labels = np.array([1 for _ in range(len(abnormal_scan_paths))])


''' Split data in the ratio 70:30 for training and validation '''
scan_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
scan_validate = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)

label_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
label_validate = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)

# print(scan_train.shape)  # (140, 128, 128, 64)
# print(scan_validate.shape)  # (60, 128, 128, 64)
# print(label_train.shape)  # (140,)
# print(label_validate.shape)  # (60,)


''' Define data loaders '''
train_loader = tf.data.Dataset.from_tensor_slices((scan_train, label_train))
validation_loader = tf.data.Dataset.from_tensor_slices((scan_validate, label_validate))

batch_size = 2


''' Augment on the fly during training '''
train_data = (
    train_loader.shuffle(len(scan_train)).map(train_prepare).batch(batch_size).prefetch(2)
)

validation_data = (
    validation_loader.shuffle(len(scan_validate)).map(validation_prepare).batch(batch_size).prefetch(2)
)

# data = train_data.take(1)
# images, labels = list(data)[0]
# images = images.numpy()
# image = images[0]
# print(images.shape)  # (2, 128, 128, 64, 1)
# print(image.shape)  # (128, 128, 64, 1)
# print(image[:, :, 30].shape)  # (128, 128, 1)


''' Build model '''
model = get_model(width=128, height=128, depth=64)
# model.summary()


''' Compile model '''
initial_lr = 0.0001
lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=100000, decay_rate=0.96, staircase=True)

model.compile(optimizer=Adam(learning_rate=lr_schedule), loss="binary_crossentropy", metrics=['accuracy'])


''' Define callbacks '''
checkpoint_callback = ModelCheckpoint(filepath='3d_image_classification.h5', monitor='val_accuracy', save_best_only=True)
early_stop_callback = EarlyStopping(monitor='val_accuracy', patience=15)


''' Train the model, doing validation at the end of each epoch '''
epoch = 50

model.fit(
    train_data,
    epochs=epoch,
    verbose=2,
    callbacks=[checkpoint_callback, early_stop_callback],
    validation_data=validation_data
)


'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(['accuracy', 'loss']):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history['val_' + metric])
    ax[i].set_title('Model {}'.format(metric))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(metric)
    ax[i].legend(['train', 'validation'])
'''


model.load_weights('3d_image_classification.h5')

prediction = model.predict(np.expand_dims(scan_validate[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_name = ['normal', 'abnormal']
for score, name in zip(scores, class_name):
    print(
        'This model is %.2f percent confident this CT scan is %s' % ((100 * score), name)
    )
