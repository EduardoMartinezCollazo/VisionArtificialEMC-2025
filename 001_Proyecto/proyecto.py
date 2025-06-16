import cv2
import torch
import serial
from torchvision import transforms

# Arduino
arduino = serial.Serial('COM3', 9600)  # Ajusta el puerto

# Modelo cargado
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = transform(frame).unsqueeze(0).to(device)
    output = model(img)
    pred = output.argmax(dim=1).item()
    label = dataset.features['label'].names[pred]

    cv2.putText(frame, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", frame)

    if label.lower() == "hola":
        arduino.write(b'ON\n')  # Enviar señal a Arduino
    else:
        arduino.write(b'OFF\n')

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
