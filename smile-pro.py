import cv2
from picamera2 import Picamera2, Preview

# Inicializar la cámara Pi
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.configure("preview")
picam2.start()

# Cargar el clasificador preentrenado para rostros y sonrisas
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    # Capturar el frame desde la cámara Pi
    frame = picam2.capture_array()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detección de sonrisas dentro de la región del rostro
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow('frame', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
picam2.stop()
cv2.destroyAllWindows()
