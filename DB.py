from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow,
                 waitKey, destroyAllWindows)
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path)
from numpy import argmin

cap = VideoCapture(0)
limite_distancia = 0.4


class DB:
    """Classe responsável por cuidar do acesso ao banco de dados com as imagens"""
    directory = 'DB'

    def __init__(self) -> None:
        print("Iniciando o sistema... \nCarregando o Banco de Dados... ")
        self.images, self.names = [], []
        self.get_img_and_name_general()
        print("Banco carregado com sucesso... \nIniciando o encoding das imagens...")
        self.encode_list = []
        self.find_encodings()
        print("Encoding terminado com sucesso... \nSistema iniciado com sucesso")

    def get_img_and_name_general(self) -> None:
        for cl in listdir(DB.directory):
            self.images.append(imread(f'{DB.directory}/{cl}'))
            self.names.append(path.splitext(cl)[0])

    def find_encodings(self) -> None:
        for image in self.images:
            self.encode_list.append(face_encodings(cvtColor(image, COLOR_BGR2RGB))[0])


dataBase = DB()

while True:
    success, img = cap.read()
    imgS = cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB)

    facesCurFrame = face_locations(imgS)
    encodeCurFrame = face_encodings(imgS, facesCurFrame)

    access_granted = False

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = compare_faces(dataBase.encode_list, encodeFace, limite_distancia)
        distancia = face_distance(dataBase.encode_list, encodeFace)
        match = argmin(distancia)

        if matches[match]:
            nome = dataBase.names[match].upper()
            print(nome)
            access_granted = True  # Acesso liberado se houver uma correspondência

        if distancia[match] <= limite_distancia:
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            putText(img, nome, (left, top - 10), FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if access_granted:
        # Libere o acesso aqui (por exemplo, abra uma porta)
        print("Acesso Liberado")
    else:
        # Negue o acesso aqui (por exemplo, exiba uma mensagem de negação)
        print("Acesso Negado")

    imshow('Webcam', img)

    if waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
destroyAllWindows()
