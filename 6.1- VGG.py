import os
import numpy as np
import pandas as pd
from PIL import Image
# Librerias para VGG19
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


carpeta_imagenes = r"C:\\TFG\\imagenes_sin_texto"

image_files = [archivo for archivo in os.listdir(carpeta_imagenes)]
df = pd.DataFrame({"filename": image_files})

# Estructuras de ResNet que se van a probar
modelos = [VGG16, VGG19]
nombres = ["VGG16", "VGG19"]

for index, m in enumerate(modelos):
    print(f'Modelo {nombres[index]} ------------------------------------------------')

    def preprocesar_imagen(imagen):
        img = image.load_img(imagen, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed

    # Cargar el modelo VGG completo
    vgg_model = m(weights='imagenet', include_top=True)

    # Crear la lista de argumentos para la funcion procesar_imagen
    args_list = [(ind, df.iloc[ind]["filename"]) for ind, imagen in df.iterrows()]
    predicciones_vgg = []

    # Bucle para predecir las caracteristicas de cada imagen
    for i in args_list:
        ind, imagen = i
        print(ind)

        img_preprocessed = preprocesar_imagen(os.path.join(carpeta_imagenes, imagen))
        predictions = vgg_model.predict(img_preprocessed, verbose=1)
        decoded_predictions = decode_predictions(predictions, top=5)
        for clase in decoded_predictions[0]:
            predicciones_vgg.append({'filename': imagen, 'clase': clase[1], 'probabilidad': clase[2]})

    # Convertir la lista de predicciones a DataFrame
    predicciones_vgg_df = pd.DataFrame(predicciones_vgg)

    # Guardar las predicciones en un archivo CSV
    predicciones_vgg_df.to_csv(f'predicciones_{nombres[index]}.csv', index=False)
