import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

def MNIST():
    tf.keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
    #plt.imshow(x_train[10])

    plt.show()


if __name__=="__main__":
    print(MNIST())