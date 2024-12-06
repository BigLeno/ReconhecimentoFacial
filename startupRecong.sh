#!/bin/bash

# Cria screen "Nome da sua screen"
screen -dmS recongsys

#Executa o script do loop dentro da screen
screen -S recongsys -X stuff 'python3 main.py\n'
