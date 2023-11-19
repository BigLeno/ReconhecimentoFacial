from cv2 import imread
from cv2.typing import MatLike

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import User


class DB:
    def __init__(self, db_directory='DB') -> None:
        """Objeto responsável por cuidar do acesso ao banco de dados com as imagens"""
        self.db_directory = db_directory
        print("Criando conexão com Banco de Dados... ")
        self.engine = create_engine(
            "postgresql://postgres:docker@localhost:5432/postgres", echo=False)
        self.session = Session(self.engine)

        print("Sessão inicializada com sucesso!")

        self.authorizedUsers = []
        self.populate_authorized_users()

    def close_connection(self) -> None:
        self.session.close()
        self.engine.dispose()
        print("Instância de conexão com o banco fechada com sucesso!")

    def populate_authorized_users(self) -> None:
        """Carrega imagens e nomes do banco de dados."""
        print("Indexando o Banco de Dados... ")
        users = select(User)

        for user in self.session.scalars(users):
            userPicture = imread(
                f'{self.db_directory}/{user.picture_path}')

            self.authorizedUsers.append((user.id, user.user_name, userPicture))

        print("Banco de Dados indexado com sucesso!")

    def insert(self, instance: object) -> None:
        self.session.add(instance)
        self.session.commit()

    def delete(self, instance: object) -> None:
        self.session.delete(instance)
        self.session.commit()

    @staticmethod
    def get_users_images(user) -> MatLike:
        return user[2]
