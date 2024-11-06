import cv2
import numpy as np

# Iniciar la cámara
captura = cv2.VideoCapture(0)

while captura.isOpened():
    # Capturar imagen y convertir de RGB a HSV
    ret, imagen = captura.read()

    if ret:
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        h, w, m = imagen.shape

        # Rango de color para detectar un objeto oscuro (como un teléfono negro o gris oscuro)
        bajos = np.array([0, 0, 0], dtype=np.uint8)  # Ajuste de color para tonos oscuros
        altos = np.array([180, 255, 60], dtype=np.uint8)

        # Crear una máscara con solo los píxeles dentro del rango
        mask = cv2.inRange(hsv, bajos, altos)
        # Encontrar el área de los objetos detectados
        moments = cv2.moments(mask)
        area = int(moments['m00'])

        # Inicializar coordenadas y distancia
        x, y, dist = 0, 0, 0

        # Ajustar el área mínima para detectar un teléfono
        if area > 200000:
            # Centro del objeto
            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])

            # Cálculo de distancia con respecto al área
            if area >= 1000000:
                dist = -(area / 500000) + 50
            elif area < 1000000 and area >= 400000:
                dist = -(area / 200000) + 70
            else:
                dist = -(area / 100000) + 90
            dist = max(15, min(int(dist), 70))

            # Mostrar coordenadas y distancia
            print("Coordenadas, Área:", (x - w // 2), (-(y - h // 2)), dist)

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
