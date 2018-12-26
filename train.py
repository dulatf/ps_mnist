import tensorflow as tf
import numpy as np
import tensorflow.keras as ke
if __name__ == '__main__':
    print("Loading the MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # rescale 0..255 pixel values to 0..1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # define a simple dense model
    dense_model = ke.models.Sequential([
        ke.layers.Flatten(),
        ke.layers.Dense(512, activation=tf.nn.relu),
        ke.layers.Dropout(0.2),
        ke.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # compile the model to create tf graph
    dense_model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy',
                        metrics=[ke.metrics.sparse_categorical_accuracy])
    # run the training
    print("Training model...")
    dense_hist = dense_model.fit(x_train, y_train,
                                 epochs=20,
                                 validation_data=(x_test,y_test),
                                 use_multiprocessing=True,
                                 workers=16)
    print(dense_model.evaluate(x_test,y_test))
    # save weights
    dense_model.save_weights('/artifacts/dense_model')
