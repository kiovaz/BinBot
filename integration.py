from gpiozero import Button
import board
import digitalio
import adafruit_character_lcd.character_lcd as character_lcd
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import cv2
import time
from dotenv import load_dotenv
import os


load_dotenv()


from gpiozero import Device, Button
from gpiozero.pins.lgpio import LGPIOFactory
Device.pin_factory = LGPIOFactory()


# ======================
# CONFIGURAÇÃO DO LCD
# ======================

lcd_rs = digitalio.DigitalInOut(board.D18)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D24)
lcd_d5 = digitalio.DigitalInOut(board.D23)
lcd_d6 = digitalio.DigitalInOut(board.D27)
lcd_d7 = digitalio.DigitalInOut(board.D22)
led_Papel = digitalio.DigitalInOut(board.D25)
led_Plastico = digitalio.DigitalInOut(board.D8)
lcd_columns = 16
lcd_rows = 2

led_Plastico.direction = digitalio.Direction.OUTPUT
led_Papel.direction = digitalio.Direction.OUTPUT

lcd = character_lcd.Character_LCD_Mono(
    lcd_rs, lcd_en,
    lcd_d4, lcd_d5, lcd_d6, lcd_d7,
    lcd_columns, lcd_rows
)

# ======================
# CONFIGURAÇÃO AZURE
# ======================

prediction_key = os.getenv("prediction_key")
endpoint = os.getenv("endpoint")
project_id = os.getenv("project_id")
iteration_name = os.getenv("iteration_name")

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

# ======================
# BOTÃO
# ======================

button = Button(2)

# ======================
# LOOP DE MONITORAMENTO
# ======================

lcd.clear()
lcd.message = "Aguardando\nbotao..."

while True:
    if button.is_pressed:
        lcd.clear()
        lcd.message = "Capturando..."

        cap = cv2.VideoCapture(0)
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()

        if ret:
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_bytes = img_encoded.tobytes()

            lcd.clear()
            lcd.message = "Analisando..."

            results = predictor.classify_image(project_id, iteration_name, image_bytes)
            melhor_previsao = max(results.predictions, key=lambda p: p.probability)

            lcd.clear()
            tag = melhor_previsao.tag_name[:16]  # Limita a 16 caracteres
            prob = melhor_previsao.probability * 100

            if tag == "Plastico":
                lcd.message = f"{tag}\n{prob:.1f}% certeza"
                led_Papel.value = True
                time.sleep(5)
                led_Papel.value = False
            elif tag == "Papel" or tag == "Papelao":
                lcd.message = f"{tag}\n{prob:.1f}% certeza"
                led_Plastico.value = True
                time.sleep(5)
                led_Plastico.value = False
            else:
                print(f"{tag}\nnão disponivel")

            time.sleep(5)
            lcd.clear()
            lcd.message = "Aguardando\nbotao..."
        else:
            lcd.clear()
            lcd.message = "Erro na\ncaptura!"
            time.sleep(3)
            lcd.clear()
            lcd.message = "Aguardando\nbotao..."
