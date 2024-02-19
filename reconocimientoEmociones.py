import cv2
import os
import numpy as np
import random
import tkinter as tk
import pyttsx3
import time

def emotionImage(emotion):
    # Emojis
    if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
    if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
    if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
    if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
    return image

# Lista de consejos para cada emoción
consejos = {
    'Felicidad': ['¡Que bueno verte tan feliz!', 'Agradece las pequeñas cosas.', 'Sigue disfrutando de esos momentos de alegría.', 'Rodéate de personas positivas.', 'Recuerda que la felicidad es contagiosa.'],
    'Enojo': ['Es importante manejar el enojo de manera saludable.', 'Alejate de la situación.', 'Intenta encontrar una forma positiva de liberar tu enojo.', 'Habla con alguien de confianza.', 'Respira profundamente y toma un descanso si es necesario.'],
    'Sorpresa': ['Disfruta del momento y recuerda que la vida está llena de sorpresas.', 'A veces las sorpresas pueden ser emocionantes.', 'Disfruta el momento.', 'No te preocupes si te sorprendes, es una reacción natural.', 'Aprende algo nuevo.'],
    'Tristeza': ['Es normal sentirse triste de vez en cuando.', 'Busca apoyo.', 'Haz algo que te haga feliz.', 'Recuerda que hay personas que te apoyan y te quieren.', 'Habla con alguien de confianza si necesitas desahogarte.']
}

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
#method = 'EigenFaces'
#method = 'FisherFaces'
method = 'LBPH'
if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')

dataPath = 'C:/Users/juanm/Documents/Algoritmos visual studio/EstadoDeEmocion/Data' # Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Crear una ventana de consejos
root = tk.Tk()
root.withdraw()
consejos_window = tk.Toplevel(root)
consejos_window.title("Consejos")
consejos_window.geometry("300x100")
consejos_label_1 = tk.Label(consejos_window, text="")
consejos_label_1.pack()
consejos_label_2 = tk.Label(consejos_window, text="")
consejos_label_2.pack()

# Inicializa la síntesis de voz
engine = pyttsx3.init()

while True:

    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        # EigenFaces
        if method == 'EigenFaces':
            if result[1] < 5700:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # FisherFaces
        if method == 'FisherFaces':
            if result[1] < 500:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # LBPHFace
        if method == 'LBPH':
            if result[1] < 60:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

        # Si se detecta una emoción, mostrar dos consejos al azar relacionados con esa emoción
        if result[1] < 60:
            emocion = imagePaths[result[0]]
            consejo_1, consejo_2 = random.sample(consejos[emocion], 2)
            consejos_label_1.config(text=f"Consejo 1: {consejo_1}")
            consejos_label_2.config(text=f"Consejo 2: {consejo_2}")
            consejos_window.update()

            # Síntesis de voz
            engine.say(f"Para tu emoción de {emocion} te recomiendo: {consejo_1}. Y adicionalmente, {consejo_2}.")
            engine.runAndWait()

    cv2.imshow('nFrame',nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
