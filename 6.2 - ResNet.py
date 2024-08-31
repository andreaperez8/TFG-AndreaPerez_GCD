import os
import numpy as np
import pandas as pd
from PIL import Image
# Librerias para ResNet
from tensorflow.keras.applications.resnet import ResNet101, ResNet152, preprocess_input as preprocess_input_resnet, decode_predictions as decode_predictions_resnet
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.preprocessing import image

carpeta_imagenes = r"C:\\TFG\\imagenes_sin_texto"
image_files = [archivo for archivo in os.listdir(carpeta_imagenes)]
df = pd.DataFrame({"filename": image_files})

# Estructuras de ResNet que se van a probar
modelos = [ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2]
nombres = ["ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2", "ResNet152V2"]

for index, m in enumerate(modelos):
    print(f'Modelo {nombres[index]} ------------------------------------------------')

    def preprocesar_imagen(imagen):
        img = image.load_img(imagen, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input_resnet(img_array)
        return img_preprocessed

    # Cargar el modelo ResNet completo
    resnet_model = m(weights='imagenet', include_top=True)

    # Crear la lista de argumentos para la funcion procesar_imagen
    args_list = [(ind, df.iloc[ind]["filename"]) for ind, imagen in df.iterrows()]
    predicciones_resnet = []

    # Bucle para predecir las caracteristicas de cada imagen
    for i in args_list:
        ind, imagen = i
        print(ind)

        img_preprocessed = preprocesar_imagen(os.path.join(carpeta_imagenes, imagen))
        predictions = resnet_model.predict(img_preprocessed, verbose=1)
        decoded_predictions = decode_predictions_resnet(predictions, top=5)
        for clase in decoded_predictions[0]:
            predicciones_resnet.append({'filename': imagen, 'clase': clase[1], 'probabilidad': clase[2]})

    # Convertir la lista de predicciones a DataFrame
    predicciones_resnet_df = pd.DataFrame(predicciones_resnet)

    # Guardar las predicciones en un archivo CSV
    predicciones_resnet_df.to_csv(f'predicciones_{nombres[index]}.csv', index=False)