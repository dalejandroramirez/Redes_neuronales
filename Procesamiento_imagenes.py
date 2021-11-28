import json
import codecs
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def Cargar_data_json():
    data_json=[]
    url="/home/usuario/Documentos/Platzi/Redes_neuronales/databasesLoadData/sign_mnist_json/data.json"
    with codecs.open(url,"rU","utf-8") as js:
        for line in js:
            data_json.append(json.loads(line))

    images=[]

    for data in data_json:
        response=requests.get(data["content"])
        img=np.asarray(Image.open(BytesIO(response.content)))
        images.append([img,data["label"]])

    plt.imshow(images[0][0].reshape(28,28))
    plt.show()

    return(0)


if __name__=="__main__":
    print(Cargar_data_json())