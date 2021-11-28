import json
import codecs
import requests
import numpy as np
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

if __name__=="__main__":
    print(Cargar_data_base64())