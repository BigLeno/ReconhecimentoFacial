from os import listdir, path
from cv2 import imread

class DB:
    def __init__(self, db_directory='DB') -> None:
        """Objeto responsável por cuidar do acesso ao banco de dados com as imagens"""
        self.db_directory = db_directory
        self.images, self.names= [], []
        self.get_img_and_name_general()
        
    def get_img_and_name_general(self) -> None:
        """Carrega imagens e nomes do banco de dados."""
        print("Carregando o Banco de Dados... ")
        for cl in listdir(self.db_directory):
            self.images.append(imread(f'{self.db_directory}/{cl}'))
            self.names.append(path.splitext(cl)[0])
        print("Banco de Dados carregado com sucesso!")