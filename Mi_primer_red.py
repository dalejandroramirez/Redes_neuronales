import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def Red_neuronal_no_optimizada(train_dir,test_dir,escala,classes):
    train_datagen=ImageDataGenerator(rescale=escala)
    test_datagen=ImageDataGenerator(rescale=escala,validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28,28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )
    validation_generator= test_datagen.flow_from_directory(
        test_dir,
        target_size=(28,28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    test_generator= test_datagen.flow_from_directory(
        test_dir,
        target_size=(28,28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
    )


    ###definimo las neurams    
    modelo_base=tf.keras.models.Sequential([ 
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"), 
    tf.keras.layers.Dense(len(classes),activation="softmax")
    ])


    bandera=input("ingrese la palabra fit si quiere ajustar ")
    if bandera=="fit":
        modelo_base.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

        history = modelo_base.fit(
            train_generator,
            epochs=20,
        validation_data=validation_generator
        )
    def visualizacion_resultados(history):
        epochs=[i for i in range(20)]
        fig,ax=plt.subplots(1,2)
        train_acc=history.history["accuracy"]
        train_loss=history.history["loss"]
        val_acc=history.history["val_accuracy"]
        val_loss=history.history["val_loss"]
        fig.set_size_inches(16,9)

        ax[0].plot(epochs,train_acc,"go-",label="entrenamiento de precision")
        ax[0].plot(epochs,val_acc,"ro-",label="validacion de precision")
        ax[0].set_title("entrenamienot y validacion de presicion")
        ax[0].legend()
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("accuracy")

        ax[1].plot(epochs,train_loss,"go-",label="entrenamiento loss")
        ax[1].plot(epochs,val_loss,"ro-",label="validacion de loss")
        ax[1].set_title("entrenamienot y validacion de loss")
        ax[1].legend()
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        plt.show()
    print(visualizacion_resultados(history))
    



 
if __name__=="__main__":
    print("hola")
    train_dir="databasesLoadData/basedatos_1/sign-language-img/Train"
    test_dir="databasesLoadData/basedatos_1/sign-language-img/Test"
    escala=256
    batch=128
    classes=[char for char in string.ascii_uppercase if char != "J" if char != "Z"]
    print(Red_neuronal_no_optimizada(train_dir,test_dir,escala,classes))
