import os
import numpy as np
import pandas as pd
from PIL import Image
# Librerias para EfficientNet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image

# Ruta a la carpeta de imagenes
carpeta_imagenes = input('Introduce la ruta del directorio con las imágenes: ')
image_files = [archivo for archivo in os.listdir(carpeta_imagenes)]
df = pd.DataFrame({"filename": image_files})

# Estructuras de EfficientNet que se van a probar
# modelos = [efficientnetv2-b0, efficientnet_b0]
# nombres = ["efficientnetv2-b0", "efficientnet_b0"]

def preprocesar_imagen(imagen):
    img = image.load_img(imagen, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input_efficientnet(img_array)
    return img_preprocessed

# Crear la lista de argumentos para la funcion procesar_imagen
args_list = [(ind, df.iloc[ind]["filename"]) for ind, imagen in df.iterrows()]

# for index, m in enumerate(nombres):
#     if m == "efficientnetv2-b0":
#         efficientnet_model = EfficientNetB0(weights='imagenet')
#     if m == "efficientnet_b0":
#         efficientnet_model = EfficientNetB0(weights=None)


efficientnet_model = EfficientNetB0(weights='imagenet')
predicciones = []

# Bucle para predecir las caracteristicas de cada imagen
for i in args_list:
    ind, imagen = i
    img_preprocessed = preprocesar_imagen(os.path.join(carpeta_imagenes, imagen))
    predictions = efficientnet_model.predict(img_preprocessed, verbose=0)
    decoded_predictions = decode_predictions(predictions, top=1)
    for clase in decoded_predictions[0]:
        predicciones.append({'filename': imagen, 'clase': clase[1], 'probabilidad': clase[2]})
    if ind % 100 == 0: # Para controlar por donde va
        print(ind)

# Convertir la lista de predicciones a DataFrame
predicciones_df = pd.DataFrame(predicciones)

# Guardar las predicciones en un archivo CSV
predicciones_df.to_csv(f'efficientnetv2-b0.csv', index=False)
print('CSV guardado')