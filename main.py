import cv2
from face_recognition import face_encodings
from os import listdir, path
from multiprocessing import Pool

class DB:
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
        with Pool(processes=200) as pool:
            self.encode_list = pool.map(self.encode_face, self.images)

    def encode_face(self, image):
        encoding = face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if encoding:
            return encoding[0]
        return None

if __name__ == '__main__':
    myDatabase = DB()
    print(myDatabase.names)
