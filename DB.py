# from cv2 import (imread, cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,imshow,waitKey, destroyAllWindows)
import cv2
from face_recognition import (face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path)
from numpy import argmin


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
            self.images.append(cv2.imread(f'{DB.directory}/{cl}'))
            self.names.append(path.splitext(cl)[0])

    def find_encodings(self) -> None:
        for image in self.images:
            self.encode_list.append(face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0])


class FaceRecognitionSystem:
    def __init__(self, database, distance_limit):
        """
        Inicializa o sistema de reconhecimento facial.

        Args:
            database: Um objeto que contém dados de referência para reconhecimento facial.
            distance_limit: A distância limite para considerar uma correspondência.
        """
        self.dataBase = database
        self.limite_distancia = distance_limit
        self.cap = cv2.VideoCapture(0)  # Inicializa a câmera
        cv2.namedWindow('Webcam')

    def find_faces(self, img):
        """
        Encontra e codifica faces na imagem fornecida.

        Args:
            img: A imagem na qual as faces serão detectadas e codificadas.

        Returns:
            Uma lista de tuplas, onde cada tupla contém a codificação da face e a localização da face na imagem.
        """
        images = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)
        faces_cur_frame = face_locations(images)
        encode_cur_frame = face_encodings(images, faces_cur_frame)
        return list(zip(encode_cur_frame, faces_cur_frame))

    def process_frame(self, img):
        """
        Processa um quadro de vídeo em busca de faces correspondentes.

        Args:
            img: O quadro de vídeo a ser processado.

        Returns:
            Uma flag indicando se o acesso foi concedido e o nome da pessoa reconhecida.
        """
        access_granted = False
        nome = ""

        encodings_and_locations = self.find_faces(img)

        for encodeFace, faceLoc in encodings_and_locations:
            matches = compare_faces(self.dataBase.encode_list, encodeFace, self.limite_distancia)
            distancia = face_distance(self.dataBase.encode_list, encodeFace)
            match = argmin(distancia)

            if matches[match]:
                nome = self.dataBase.names[match].upper()
                access_granted = True  # Acesso liberado se houver uma correspondência

            if distancia[match] <= self.limite_distancia:
                top, right, bottom, left = faceLoc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return access_granted, nome

    def run(self):
        """
        Executa o sistema de reconhecimento facial em tempo real.

        Pressione 'q' para encerrar a execução.
        """
        while True:
            success, img = self.cap.read()

            access_granted, nome = self.process_frame(img)

            if access_granted:
                # Libere o acesso aqui (por exemplo, abra uma porta)
                print("Acesso Liberado")
            else:
                # Negue o acesso aqui (por exemplo, exiba uma mensagem de negação)
                print("Acesso Negado")

            cv2.imshow('Webcam', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    limite_distancia = 0.4
    myDatabase = DB()
    myFaceRecognitionSystem = FaceRecognitionSystem(myDatabase, limite_distancia)
    myFaceRecognitionSystem.run()
