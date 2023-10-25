import cv2
import numpy as np
import face_recognition

imgFidel = face_recognition.load_image_file('images/fidel.jpg')
imgFidel = cv2.cvtColor(imgFidel, cv2.COLOR_BGR2RGB)
imgTeste = face_recognition.load_image_file('images/fidel teste.jpg')
imgTeste = cv2.cvtColor(imgTeste, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgFidel)[0]
encodeFidel = face_recognition.face_encodings(imgFidel)[0]
cv2.rectangle(imgFidel, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTeste = face_recognition.face_locations(imgTeste)[0]
encodeTeste = face_recognition.face_encodings(imgTeste)[0]
cv2.rectangle(imgTeste, (faceLocTeste[3], faceLocTeste[0]), (faceLocTeste[1], faceLocTeste[2]), (255, 0, 255), 2)

resultados = face_recognition.compare_faces([encodeFidel], encodeTeste)
distancia = face_recognition.face_distance([encodeFidel], encodeTeste)
print(resultados, distancia)

cv2.imshow("Fidel", imgFidel)
cv2.imshow("Teste", imgTeste)
cv2.waitKey(0)
cv2.destroyAllWindows()
