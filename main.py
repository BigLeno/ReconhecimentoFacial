import logging
from face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    myFaceRecognitionSystem = FaceRecognitionSystem()
    myFaceRecognitionSystem.run()
