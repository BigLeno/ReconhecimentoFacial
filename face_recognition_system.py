from cv2 import (cvtColor, COLOR_BGR2RGB, VideoCapture, resize, rectangle, putText, FONT_HERSHEY_SIMPLEX,
                 imshow, waitKey, destroyAllWindows, namedWindow, imwrite)
from face_recognition import (
    face_encodings, face_locations, compare_faces, face_distance)
from os import (listdir, path, makedirs)
from multiprocessing import Pool
from numpy import argmin
from string import ascii_lowercase
from random import choice
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Union

from database import DB
from models import AccessHistory


class FaceRecognitionSystem:

    def __init__(self, distance_limit=0.4) -> None:
        """Objeto que representa o sistema de reconhecimento facial"""
        self.dataBase, self.limite_distancia = DB(), distance_limit
        self.last_file_count = self.get_file_count()
        self.unknown_faces_seen_at, self.encode_list = {}, []
        self.colors = {'red': (0, 0, 255), 'green': (
            0, 255, 0), 'blue': (255, 0, 0)}
        self.quality_settings = {'qvga': (320, 240), 'vga': (
            640, 480), 'hd': (1280, 720), 'full_hd': (1920, 1080)}
        self.access_types = ["Reconhecido", "Não reconhecido"]
        self.date_and_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.csv_directory = "registro_iteracoes.csv"
        print("Iniciando o Sistema de Reconhecimento Facial... \nIniciando o encoding das imagens...")
        self.find_encodings()
        print("Encoding terminado com sucesso..")
        self.getWebcam()
        print("Sistema iniciado sem falhas")

    def getWebcam(self, quality: Optional[str] = 'vga') -> None:
        """Inicializa a webcam com a qualidade especificada"""
        self.cap = VideoCapture(0)
        if quality not in self.quality_settings:
            print(f"Qualidade de webcam não suportada: {quality}")
            return
        self.cap.set(3, self.quality_settings[quality][0])
        self.cap.set(4, self.quality_settings[quality][1])
        namedWindow('Webcam')

    def find_encodings(self) -> None:
        """Gera as codificações faciais para as imagens no banco de dados"""
        with Pool(processes=None) as pool:
            userImages = list(
                pool.map(self.dataBase.get_users_images, self.dataBase.authorizedUsers))
            self.encode_list = pool.map(self.encode_face, userImages)

    @staticmethod
    def save_img(directory, img, archive_name: Optional[str] = "sem_nome") -> None:
        """Salva uma imagem em um diretório específico com um nome de arquivo opcional"""
        makedirs(directory, exist_ok=True)
        imwrite(f"{directory}/{archive_name}.jpg", img)

    def register_acess(self, relative_path: str, acess_type: int,  name: str) -> None:
        """Registra um acesso no banco de dados com informações de tempo, tipo e nome"""
        data = pd.read_csv(self.csv_directory)
        data.loc[len(data)] = [self.date_and_time, self.access_types[acess_type],
                               f'{relative_path}/{name.lower()}.jpg']
        data.to_csv(self.csv_directory, index=False)

    def put_rectangles_and_text(self, image, positions: tuple, color: str, text: Optional[str] = '') -> None:
        """Adiciona retângulo e texto a uma imagem na posição especificada"""
        rectangle(image, (positions[3], positions[0]),
                  (positions[1], positions[2]), self.colors[color], 2)
        putText(image, text, (positions[3], positions[0]),
                FONT_HERSHEY_SIMPLEX, 0.9, self.colors[color], 2)

    def get_file_count(self) -> Union[int, int]:
        """
            Retorna o número de arquivos no diretório do banco de dados, se existir.
            Retorna 0 se o diretório não existir ou não for um diretório válido.
        """
        if path.exists(self.dataBase.db_directory) and path.isdir(self.dataBase.db_directory):
            files = listdir(self.dataBase.db_directory)
            return len(files)
        else:
            return 0

    def monitor_directory(self) -> None:
        """
            Monitora o diretório em busca de alterações no número de arquivos.
            Quando a pasta é atualizada, recarrega o banco de dados e recria as codificações.
        """
        current_file_count = self.get_file_count()
        if current_file_count != self.last_file_count:
            print("\n---------------------------\n")
            print("\nA pasta foi atualizada...")
            print("\nFechando instância atual do banco...")
            self.dataBase.close_connection()

            self.dataBase = DB()
            print("Iniciando o encoding das imagens...")
            self.find_encodings()
            print("Encoding terminado com sucesso.. \nDados atualizados com sucesso!\n")
            print("\n---------------------------\n")
            self.last_file_count = current_file_count

    @staticmethod
    def generate_unique_id(length=8) -> str:
        """Gera um ID único aleatório com o comprimento especificado"""
        return ''.join(choice(ascii_lowercase) for _ in range(length))

    @staticmethod
    def encode_face(image) -> Optional[List]:
        """Codifica o rosto em uma imagem e retorna a codificação, ou None se não for possível"""
        encoding = face_encodings(cvtColor(image, COLOR_BGR2RGB))
        if encoding:
            return encoding[0]

    @staticmethod
    def find_faces(img) -> List[Tuple[list, list]]:
        """Encontra e retorna as codificações e localizações dos rostos em uma imagem."""
        return list(zip(
            face_encodings(
                cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB),
                face_locations(cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB))),
            face_locations(
                cvtColor(resize(img, (0, 0), None, 0.25, 0.25), COLOR_BGR2RGB))
        ))

    def compare_faces_and_get_distances(self, data, new_faces) -> Tuple[list, list, int]:
        """Compara as faces e obtém as distâncias entre elas."""
        return (
            compare_faces(data, new_faces, self.limite_distancia),
            face_distance(data, new_faces),
            argmin(face_distance(data, new_faces))
        )

    def process_frame(self, img) -> Tuple[bool, str]:
        """
            Processa um frame de imagem para reconhecimento facial e 
            retorna se o acesso foi concedido e o nome, se reconhecido.
        """
        access_granted = False
        nome = ""
        id = -1

        current_time = datetime.now()
        time_delay = 5

        for encodeFace, faceLoc in self.find_faces(img):
            matches, distancia, match = self.compare_faces_and_get_distances(
                self.encode_list, encodeFace)
            top, right, bottom, left = [x * 4 for x in faceLoc]

            if matches[match] or distancia[match] <= self.limite_distancia:
                user = self.dataBase.authorizedUsers[match]
                id = user[0]
                nome = user[1].upper()

                access_granted = True
                self.put_rectangles_and_text(
                    img, (top, right, bottom, left), 'green', nome)

            else:
                unique_id = self.generate_unique_id()

                last_key = list(self.unknown_faces_seen_at.keys()
                                )[-1] if self.unknown_faces_seen_at else None
                last_seen = self.unknown_faces_seen_at.get(last_key, None)
                self.put_rectangles_and_text(
                    img, (top, right, bottom, left), 'red')

                if last_seen is None or (current_time - last_seen).total_seconds() >= time_delay:
                    print("Rosto desconhecido encontrado")
                    archive_path = f"RD/{unique_id}.jpg"

                    self.register_acess('RD', 1, unique_id)
                    self.save_img("RD", img[top:bottom, left:right], unique_id)
                    self.unknown_faces_seen_at[unique_id] = current_time

                    save_unknown = AccessHistory(
                        is_unknown=True, unknown_picture_path=archive_path)
                    self.dataBase.insert(save_unknown)

                    print("Acesso registrado!")

        return access_granted, nome, id

    def run(self) -> None:
        """Inicia o sistema de reconhecimento facial e controla a webcam em um loop contínuo."""
        timer = 5
        last_access_time = datetime.now() - timedelta(seconds=timer)

        if (len(self.encode_list) == 0):
            print(
                'Ocorreu uma exceção: Não existem usuários cadastrados no banco de dados!')
            return

        while True:
            success, img = self.cap.read()

            if not success:
                print("Um problema com a webcam foi encontrado!")
                break

            self.monitor_directory()

            current_time = datetime.now()
            time_elapsed = (current_time - last_access_time).total_seconds()
            access_granted, nome, id = self.process_frame(img)
            if access_granted and time_elapsed >= timer:
                print(f"Seja bem-vindo {nome}, acesso liberado!")
                self.register_acess('DB', 0, nome)
                save_user = AccessHistory(user_id=id, is_unknown=False)
                self.dataBase.insert(save_user)
                last_access_time = current_time
                print("Acesso registrado!")

            imshow('Webcam', img)

            if waitKey(1) & 0xFF == ord('q'):
                print("Encerrando sistema...")
                break

        self.cap.release()
        self.dataBase.close_connection()
        destroyAllWindows()
        print("Sistema encerrado com sucesso!")
