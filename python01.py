import cv2
import numpy as np

# Iniciamos la cámara
captura = cv2.VideoCapture(0)

while captura.isOpened():
    # Capturamos una imagen y la convertimos de RGB a HSV
    ret, imagen = captura.read()

    if ret:
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        h, w, m = imagen.shape
        
        # Rango de color para detectar una pelota de tenis
        bajos = np.array([30, 80, 80], dtype=np.uint8)
        altos = np.array([40, 255, 255], dtype=np.uint8)

        # Crear una máscara con solo los píxeles dentro del rango
        mask = cv2.inRange(hsv, bajos, altos)
        # Encontrar el área de los objetos detectados
        moments = cv2.moments(mask)
        area = int(moments['m00'])
        
        # Inicializamos las coordenadas y la distancia
        x, y, dist = 0, 0, 0

        if area > 100000:
            # Centro del objeto
            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])

            # Cálculos para definir la distancia en función del área
            if area >= 845695.7:
                if area >= 2301333.5:
                    dist = -(area / 307781.175) + 37.4771743
            if area < 2301333.5:
                if area >= 859737.4:
                    dist = -(area / 72079.83) + 61.9275725  
            if area < 859737.4:
                dist = -(area / 19652.8425) + 93.74622211
            dist = max(15, min(int(dist), 70))  # Limitar la distancia entre 15 y 70

            # Mostrar coordenadas y distancia
            print("Coordenadas, Area:", (x - w // 2), (-(y - h // 2)), dist)

            # Dibujar una marca en el centro del objeto
            cv2.rectangle(imagen, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), 18)
            cv2.line(imagen, (0, y), (w, y), (255, 0, 0), 2)
            cv2.line(imagen, (x, 0), (x, h), (255, 0, 0), 2)

        # Dividir la imagen en cuatro cuadrantes
        cv2.line(imagen, (0, h // 2), (w, h // 2), (150, 200, 0), 2)
        cv2.line(imagen, (w // 2, 0), (w // 2, h), (150, 200, 0), 2)

        # Mostrar la imagen y la máscara
        cv2.imshow('mask', mask)
        cv2.imshow('Camara', imagen)
        
        # Presionar 'q' para salir
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar ventanas
captura.release()
cv2.destroyAllWindows()
