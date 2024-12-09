import logging
from face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y - %H:%M:%S")

if __name__ == '__main__':
    ip_camera = "http://10.6.3.129:81/stream"
    myFaceRecognitionSystem = FaceRecognitionSystem(camera_source=ip_camera)
    myFaceRecognitionSystem.run()
