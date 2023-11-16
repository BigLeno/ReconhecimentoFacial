# Reconhecimento Facial
Um código simples e funcional para uma futura aplicação utilizada na InPacta (Incubadora) da  Universidade Federal do Rio Grande do Norte (UFRN).
Desenvolvido em Python com a biblioteca `OpenCV` ('cv2') para processamento de imagens e o módulo `face_recognition` para reconhecimento facial. 
O sistema pode monitorar uma webcam em tempo real, reconhecer rostos e salvar dados de acesso em um banco de dados.


# :pushpin: Tabela de conteúdos

- [Tecnologias](#computer-tecnologias)
- [Setup de Pacotes](#gear-setup-de-pacotes)
- [Como rodar](#tv-como-rodar)

# :computer: Tecnologias

- [x] Python
- [x] Face Recognition
- [x] OpenCV
- [x] Dlib
- [x] Cmake
- [x] Pandas

# :gear: Setup de Pacotes

```shell
## Com Python 3 e Python PiP instalados, siga os passos abaixos para configurar o ambiente para rodar a aplicação.
## Tenha certeza que, no Windows, tenha o pacote do Visual Studio de desenvolvimento em C++ desktop instalados.
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

# :tv: Como Rodar

```shell
## Para rodar a aplicação, apenas execute no diretório raiz da aplicação:
$ python3 main.py
```
