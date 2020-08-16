# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from timeit import default_timer as timer

# Custom libraries
from tensorflow_generator import TensorflowDataGenerator, TensorflowDataGenerator_Test


#### Set the CNN parameters
num_epoch = 20
batch_size = 6
framework = 'tensorflow'
im_size = 224
num_im = 1000
model_type = 'mobilenet' #mobilenet or simple
train_type = 'manual'
predict = True

# Set up directories
root_path = './'
data_dir = os.path.join(root_path,'data/train')
log_dir = os.path.join(root_path, 'logs', framework, model_type, 'logs', train_type + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
if predict:
    ckpt_name = f'{train_type}_20200816-184346'
    ckpt_dir = os.path.join(root_path, 'logs', framework, model_type, 'ckpts', ckpt_name)
else:
    ckpt_dir = os.path.join(root_path, 'logs', framework, model_type, 'ckpts', train_type + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))


# Setup input pipeline
train_dir = os.path.join(root_path,'data/train')
test_dir = os.path.join(root_path,'data/test')

train_gen = TensorflowDataGenerator(train_dir, batch_size, im_size=im_size, num_im=num_im, shuffle=True)
val_imgs = train_gen.load_val()
test_gen = TensorflowDataGenerator_Test(test_dir, batch_size, im_size=im_size)

if model_type=='simple':
    
    # Set up model
    model = Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(im_size, im_size, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1)
    ])
    # Compile model
    learning_rate = 1e-3    

    # Get model summary
    model.summary()

elif model_type=='mobilenet':
    learning_rate = 0.0001
    # get just the feature extraction layers of mobilenet
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(im_size, im_size, 3))

    # freeze the feature extractor convolutional layers
    base_model.trainable = False
    
    # define a classification layer on top
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    # fit it all together with some dropout
    inputs = tf.keras.Input(shape=(im_size, im_size, 3))
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


else:
    raise NameError(f'model_type {model_type} not recognised')

if not predict:
    if train_type == 'auto':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate)

        # Setup callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_dir,
            save_weights_only=True,
            monitor='val_acc',
            mode='max',
            save_best_only=True)

        model.compile(loss=loss_fn, 
            optimizer=optimizer, 
            metrics=['accuracy'])
        
        # train the model
        history = model.fit(
            train_gen,
            validation_data=val_imgs,
            validation_steps=len(val_imgs[0]) // batch_size, 
            epochs=num_epoch,
            callbacks=[tensorboard_callback, model_checkpoint_callback],
            use_multiprocessing=True, 
            workers=8
        )

    elif train_type == 'manual':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate)

        train_acc = tf.keras.metrics.Mean()
        train_loss = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            acc_value = tf.math.equal(y, tf.math.round(tf.keras.activations.sigmoid(logits)))
            train_acc.update_state(acc_value)
            train_loss.update_state(loss_value)

        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            loss_value = loss_fn(y, val_logits)
            acc_value = tf.math.equal(y, tf.math.round(tf.keras.activations.sigmoid(val_logits)))
            val_acc.update_state(acc_value)
            val_loss.update_state(loss_value)
            return loss_value, acc_value

        # Setup tensorboard
        file_writer = tf.summary.create_file_writer(log_dir + "/metrics")    
        best_val_acc = 0 # for model check pointing
        # Epoch loop
        for epoch in range(1, num_epoch + 1):
            start_time = timer()
            # Training loop
            for inputs, targets in train_gen:
                train_step(inputs, targets)

            # Validation loop
            for batch_idx in range(0, len(val_imgs[1]), batch_size):
                inputs = val_imgs[0][batch_idx:batch_idx+batch_size,...]
                targets = val_imgs[1][batch_idx:batch_idx+batch_size]
                test_step(inputs, targets)

            # Log metrics to tensorboard
            end_time = timer()
            with file_writer.as_default():
                tf.summary.scalar('Loss/train', train_loss.result(), step=epoch)
                tf.summary.scalar('Loss/validation', val_loss.result(), step=epoch)
                tf.summary.scalar('Accuracy/train', train_acc.result(), step=epoch)
                tf.summary.scalar('Accuracy/validation', val_acc.result(), step=epoch)
                tf.summary.scalar('epoch_time', end_time - start_time, step=epoch)
        
            # Display metrics at the end of each epoch. 
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss.result()} \tValidation Loss: {val_loss.result()} \tTraining Accuracy: {train_acc.result()} \tValidation Accuracy: {val_acc.result()} \tTime taken: {end_time - start_time}')
        
            # checkpoint if improved
            if val_acc.result()>best_val_acc:
                model.save_weights(ckpt_dir)
                best_val_acc = val_acc.result()

            # Reset training metrics at the end of each epoch
            train_acc.reset_states()
            train_loss.reset_states()
            val_acc.reset_states()
            val_loss.reset_states()
else:
    model.load_weights(ckpt_dir)
    for inputs in test_gen:
        outputs = model(inputs, training=False)
        break