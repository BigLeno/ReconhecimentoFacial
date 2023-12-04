import logging
from cv2 import imread
from cv2.typing import MatLike

from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import User


class DB:
    def __init__(self, db_directory='DB') -> None:
        """Objeto responsável por cuidar do acesso ao banco de dados com as imagens"""
        self.db_directory = db_directory
        load_dotenv()
        logging.info("Criando conexão com Banco de Dados... ")
        db_host = os.getenv("DB_HOST")
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASS")
        db_name = os.getenv("DB_NAME")
        teste = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
        self.engine = create_engine(teste, echo=False)
        self.session = Session(self.engine)

        logging.info("Sessão do banco de dados inicializada com sucesso!")

        self.authorizedUsers = []
        self.populate_authorized_users()

    def close_connection(self) -> None:
        self.session.close()
        self.engine.dispose()
        logging.info("Instância de conexão com o banco fechada com sucesso!")

    def populate_authorized_users(self) -> None:
        """Carrega imagens e nomes do banco de dados."""
        logging.info("Indexando o Banco de Dados... ")
        users = select(User)

        for user in self.session.scalars(users):
            userPicture = imread(
                f'{self.db_directory}/{user.picture_path}')

            if userPicture is None:
                logging.warn(
                    f"Não foi possível encontrar a foto do usuário {user.user_name} na pasta DB! Prosseguindo...")
                continue

            self.authorizedUsers.append(
                (user.id, user.user_name, userPicture))

        logging.info("Banco de Dados indexado com sucesso!")

    def insert(self, instance: object) -> None:
        self.session.add(instance)
        self.session.commit()

    def delete(self, instance: object) -> None:
        self.session.delete(instance)
        self.session.commit()

    @staticmethod
    def get_users_images(user) -> MatLike:
        return user[2]
