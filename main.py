import logging
from face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')



if __name__ == '__main__':
    myFaceRecognitionSystem = FaceRecognitionSystem()
    myFaceRecognitionSystem.run()
