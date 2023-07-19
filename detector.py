import cv2
import mediapipe as mp
import numpy as np
import time

class Detector():
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5
                 ):
        
        #Parametros necessarios para inicializar o Hands
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        #Inicializando o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                        self.max_num_hands,
                                        self.complexity,
                                        self.detection_con,
                                        self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self,
                   img: np.ndarray,
                   draw_hands: bool = True
                   ):
        """Encontra as mãos na imagem e as desenha se draw = True"""
        #Correção de cores
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Coleta resultados do processo das hands e analisar
        self.results = self.hands.process(img_rgb)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self,
                    img: np.ndarray,
                    hand_number: int = 0,):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            # print(my_hand.landmark)
            for id, landmark in enumerate(my_hand.landmark):
                # print(id, landmark)
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                self.required_landmark_list.append([id, cx, cy])
                
        return self.required_landmark_list

if __name__ == "__main__":
    #dados de video
    previous_time = 0
    current_time = 0
    capture = cv2.VideoCapture(0)

    Detector = Detector()

    while True:
        _, img = capture.read()

        #Manipular frame
        img = Detector.find_hands(img)
        landmark_list = Detector.find_position(img)
        if landmark_list:
            print(landmark_list[8])
        #e retornar desenho da mão

        #determinar fps
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        #e retornar o frame com o desenho da mao
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        

        cv2.imshow("Imagem do Rodrigo", img)

        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
