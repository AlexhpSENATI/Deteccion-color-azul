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

        # Rango de color para detectar un objeto con fondo azul
        bajos = np.array([90, 50, 50], dtype=np.uint8)  # Ajuste para tonos de azul
        altos = np.array([130, 255, 255], dtype=np.uint8)

        # Crear una máscara con solo los píxeles dentro del rango
        mask = cv2.inRange(hsv, bajos, altos)

        # Encontrar los contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:  # Verifica si hay contornos detectados
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por un área mínima de contorno, ajusta este valor si es necesario
                if area > 5000:  # Reduce el valor mínimo para capturar objetos más pequeños
                    # Obtener el rectángulo que encierra el contorno
                    x, y, w_contour, h_contour = cv2.boundingRect(contour)

                    # Dibujar un rectángulo alrededor del contorno detectado
                    cv2.rectangle(imagen, (x, y), (x + w_contour, y + h_contour), (0, 255, 0), 3)  # Rectángulo verde

                    # Calcular el centro del objeto detectado
                    centro_x = x + w_contour // 2
                    centro_y = y + h_contour // 2

                    # Dibujar el centro del objeto
                    cv2.circle(imagen, (centro_x, centro_y), 5, (0, 0, 255), -1)  # Punto rojo

                    # Calcular la distancia en función del área
                    dist = -(area / 500000) + 50
                    dist = max(15, min(int(dist), 70))
                    print(f"Coordenadas: ({centro_x}, {centro_y}), Distancia: {dist}")

        else:
            print("No se detectaron contornos")

        # Mostrar la imagen con el rectángulo y el centro del teléfono detectado
        cv2.imshow('Camara', imagen)
        cv2.imshow('Mask', mask)

        # Presionar 'q' para salir
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar ventanas
captura.release()
cv2.destroyAllWindows()
