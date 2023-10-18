import cv2
import os
import face_recognition as fr
from time import time

db_directory = 'DB'
resultados = {}

input_Images = fr.load_image_file('latrel.jpg')
input_Images = cv2.cvtColor(input_Images, cv2.COLOR_BGR2RGB)
encoded_input_image = fr.face_encodings(input_Images)[0]

start_time = time()

# Carrega o modelo OpenCV
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Detecta as faces na imagem de entrada
input_faces = face_cascade.detectMultiScale(input_Images, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Calcula os encodings das faces detectadas na imagem de entrada
encoded_input_faces = []
for (x, y, w, h) in input_faces:
    roi = input_Images[y:y + h, x:x + w]
    encoded_input_faces.append(fr.face_encodings(roi)[0])

# Para cada imagem no banco de dados, compara as faces detectadas na imagem de entrada com as faces no banco de dados
for filename in os.listdir(db_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        db_image = fr.load_image_file(os.path.join(db_directory, filename))
        db_image = cv2.cvtColor(db_image, cv2.COLOR_BGR2RGB)
        encoded_db_image = fr.face_encodings(db_image)[0]

        # Compara as faces detectadas na imagem de entrada com a face no banco de dados
        comparacao = fr.compare_faces(encoded_input_faces, encoded_db_image)[0]
        distancia = fr.face_distance(encoded_input_faces, encoded_db_image)[0]

        resultados[filename] = {'comparacao': comparacao, 'distancia': distancia}

end_time = time()

for filename, resultado in resultados.items():
    print(f'Comparando {filename}:')
    print(f'   Mesmo rosto? {resultado["comparacao"]}')
    print(f'   Distância: {resultado["distancia"]}')

print(f'Tempo de execução: {end_time - start_time:.2f} segundos')
