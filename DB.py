import cv2
import numpy as np
import face_recognition
import os

path = 'DB'
imagens = []
nomes = []
lista = os.listdir(path)
print(lista)
for cl in lista:
    curImg = cv2.imread(f'{path}/{cl}')
    imagens.append(curImg)
    nomes.append(os.path.splitext(cl)[0])
print(nomes)

def findEcodings(imagens):
    encodelist = []
    for img in imagens:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnown = findEcodings(imagens)
print("Encoding complete")

cap = cv2.VideoCapture(0)

# Defina um limite para considerar uma correspondência válida
limite_distancia = 0.4  # Você pode ajustar esse valor conforme necessário

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    access_granted = False

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace, limite_distancia)
        distancia = face_recognition.face_distance(encodelistKnown, encodeFace)
        match = np.argmin(distancia)

        if matches[match]:
            nome = nomes[match].upper()
            print(nome)
            access_granted = True  # Acesso liberado se houver uma correspondência

        if distancia[match] <= limite_distancia:
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if access_granted:
        # Libere o acesso aqui (por exemplo, abra uma porta)
        print("Acesso Liberado")
    else:
        # Negue o acesso aqui (por exemplo, exiba uma mensagem de negação)
        print("Acesso Negado")

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
