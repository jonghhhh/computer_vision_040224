import cv2
import os
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

class FaceEmotion:
    def __init__(self, input_path, save_folder):
        self.input_path = input_path
        self.save_folder = save_folder
        self.face_detector = MTCNN()  # MTCNN face detector 초기화

    def face_emotion_detect(self, save=False):
        '''
        이미지에서 얼굴을 찾고 감정을 탐지
        '''
        image = cv2.imread(self.input_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(image_rgb)

        if not faces:  # 얼굴이 하나도 탐지되지 않았을 경우
            print("No faces detected.")
            return []  # 빈 결과 반환

        # 얼굴 바운딩박스별 감정 분석
        result_analysis = []
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            face_img = image_rgb[y:y+height, x:x+width]  # image_rgb에서 face 영역 추출
            try:
                analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotions = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
                result_analysis.append([i, face, emotions, dominant_emotion])
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image_rgb, str(i) + '_' + dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
            except Exception as e:
                print("Error in emotion analysis:", e)

        if save:  # 출력
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.savefig(os.path.join(self.save_folder, self.input_path.split('/')[-1].replace('.jpg', '_emotion.jpg')), bbox_inches='tight', pad_inches=0)
            plt.show()

        return result_analysis


