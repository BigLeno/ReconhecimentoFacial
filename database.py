import logging
from cv2 import imread
from cv2.typing import MatLike
from os import listdir, path

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session

from models import User


class DB:
    def __init__(self, db_directory='DB') -> None:
        """Objeto responsável por cuidar do acesso ao banco de dados com as imagens"""
        self.db_directory = db_directory
        logging.info("Criando conexão com Banco de Dados... ")
        self.engine = create_engine(
            "postgresql://inpacta:recogn%40inpacta@10.6.1.30:5432/recogndb", echo=False)
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
                logging.warning(
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

    def update_users_from_directory(self) -> None:
        """Atualiza o banco de dados com base nas mudanças no diretório de imagens"""
        current_files = set(listdir(self.db_directory))
        db_users = {user.picture_path for user in self.session.scalars(select(User))}

        # Adicionar novos usuários
        new_files = current_files - db_users
        for file_name in new_files:
            user_name = path.splitext(file_name)[0].upper()
            user_picture_path = f'{self.db_directory}/{file_name}'
            user_picture = imread(user_picture_path)

            if user_picture is not None:
                new_user = User(user_name=user_name, picture_path=file_name)
                self.insert(new_user)
                self.authorizedUsers.append((new_user.id, new_user.user_name, user_picture))
                logging.info(f"Novo usuário {user_name} adicionado ao banco de dados.")

        # Remover usuários que não existem mais no diretório
        removed_files = db_users - current_files
        for file_name in removed_files:
            user_to_remove = self.session.scalars(select(User).where(User.picture_path == file_name)).first()
            if user_to_remove:
                self.delete(user_to_remove)
                self.authorizedUsers = [user for user in self.authorizedUsers if user[1] != user_to_remove.user_name]
                logging.info(f"Usuário {user_to_remove.user_name} removido do banco de dados.")

    @staticmethod
    def get_users_images(user) -> MatLike:
        return user[2]
