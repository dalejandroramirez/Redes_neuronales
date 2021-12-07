import json
import codecs
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def Cargar_data_json():
    data_json=[] ##es un archivo con un diccionario con la url de la imagen y su label
    url="/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_json/data.json"
    with codecs.open(url,"rU","utf-8") as js:
        for line in js:
            data_json.append(json.loads(line))

    images=[]
    for data in data_json:
        response=requests.get(data["content"])
        img=np.asarray(Image.open(BytesIO(response.content)))
        images.append([img,data["label"]])
    plt.imshow(images[2][0].reshape(28,28))
    plt.show()
    return(0)


def Cargar_data_base64():
    import base64
    url="/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_base64/data.json"
    with open(url) as f:
        data=json.load(f)
    base64_img_bytes=data["b"].encode("utf-8")
    path_img="/home/usuario/Documentos/Platzi/Redes_neuronales/Imagenes/images.png"
    with open(path_img,"wb") as file_to_save:
        decoded_image_data=base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
    img=Image.open(path_img)
    plt.imshow(img)
    plt.show()
    return(0)

def Cargar_data_csv():
    train="/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_train/sign_mnist_train.csv"
    test=pd.read_csv("/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_test/sign_mnist_test.csv")
    train=pd.read_csv(train)
    
    labels=train["label"].values
    train.drop("label",axis=1,inplace=True)
    images=train.values 
    plt.imshow(images[1].reshape(28,28))
    plt.show()
    return(0)

def Preprocesamiento():
    train=pd.read_csv("/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_train/sign_mnist_train_clean.csv")
    test=pd.read_csv("/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_test/sign_mnist_test.csv")
    ##debemos ver si la base de datos está balanceada es decir que hay cantidades similares
    #de cada label
    """plt.figure(figsize=(10,10))
    sns.set_style("darkgrid")
    sns.countplot(train["label"])
    plt.show()"""
    y_train=train["label"]
    y_test=train["label"]
    labels=train["label"].values
    del train["label"]
    del test["label"]
    ##que hay en la data
    """print(train.info())
    print(np.unique(np.array(labels)))"""
    ##¿hay variables nulas?
    "print(train.isnull().values.any())"
    ##¿tiene duplicados?
    """print(train[train.duplicated()]) """ ##elimino lo que es ruido
    train =train.drop([317,487,595,689,802,861],axis=0)
    #print(train[train['pixel1'] =="fwefew"])
    train=train.drop([727],axis=0)     

    ##normalizar (los valores de la matrix estan entre 0, 255)
    train=train.astype(str).astype(int)
    train=train/255
    test=test/255
    
if __name__=="__main__":
    print(Preprocesamiento())