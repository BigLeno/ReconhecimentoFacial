# Reconhecimento Facial

Um código simples e funcional de uma futura aplicação utilizada na InPacta (Incubadora) da Universidade Federal do Rio Grande do Norte (UFRN).

# :pushpin: Tabela de conteúdos

- [Tecnologias](#computer-tecnologias)
- [Setup Database](#gear-setup-database)
- [Setup de Pacotes](#gear-setup-de-pacotes)
- [Configurar Conexão com Database](#gear-configurar-conexão)
- [Como rodar](#tv-como-rodar)

# :computer: Tecnologias

- [x] Python
- [x] Face Recognition
- [x] Postgres
- [x] SQLAlchemy
- [x] Alembic

# :gear: Setup Database

```shell
## Os passos a seguir descrevem como configurar um container docker com PostgreSQL
## Portanto, certifique-se de ter instalado o Docker em sua máquina.

$ docker run --name postgresdb -p 5432:5432 -v /caminho/para/a/pasta/da/database:/var/lib/postgresql/data -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=docker -d postgres

# O Comando "run" cria um novo container
# A opção "--name" define o nome do container como "postgresdb"
# A opção "-p" define a porta do postgres no container "5432" que será espelhada como TCP "5432", ou seja, "5432:5432"
# O caminho para a pasta database define onde serão guardados os arquivos relativos ao PostgreSQL em sua máquina.
# A opção "-e POSTGRES_USER" define o usuário principal como "postgres"
# A opção "-e POSTGRES_PASSWORD" define a senha do usuário principal como "docker"
# O comando "-d" define qual imagem docker será utilizada no container, nesse caso, "postgres"

## Como rodar o container:
$ docker start postgresdb

## Para parar o container:
$ docker stop postgresdb

```

# :gear: Setup de Pacotes

```shell
## Com Python 3 e Python PiP instalados, siga os passos abaixos para configurar o ambiente para rodar a aplicação.
## OBS: Tenha certeza de estar na raiz do projeto antes de qualquer um dos passos a seguir!

### Ambiente e Pacotes

# Crie um ambiente virtual com o virtualenv para a aplicação e ative o ambiente:
$ python3 -m virtualenv venv
$ source venv/bin/activate

## OBS: Dependendo do shell que você estiver usando, a extensão do arquivo activate pode ser necessária de alteração.
## Ex: Terminal com Fish Shell:
$ source venv/bin/activate.fish

# Use o PiP para instalar os pacotes necessários localizados no requirements.txt:
$ pip install -r requirements.txt
```

# :gear: Configurar Conexão

```ini
## No arquivo alembic.ini, altere a seguinte propriedade:
## Define a URL de conexão com seu banco de dados.
sqlalchemy.url=postgresql://usuario:senha@host:5432/nome_db
```

```python
## No arquivo database.py, procure a seguinte linha de código dentro da classe DB no método __init__:
self.engine = create_engine("postgresql://usuario:senha@host:5432/nome_db", echo=False)
```

```shell
## Rode no seu terminal para executar as migrações no banco de dados.
$ alembic upgrade head
```

# :tv: Como Rodar

```shell
## Para rodar a aplicação, apenas execute no diretório raiz da aplicação:
$ python3 main.py
```
