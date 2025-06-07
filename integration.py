from gpiozero import Button, OutputDevice, Device
from gpiozero.pins.lgpio import LGPIOFactory
import board
import digitalio
import adafruit_character_lcd.character_lcd as character_lcd
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import cv2
import time
from time import sleep
import threading

Device.pin_factory = LGPIOFactory()

# Configuração do hardware
button = Button(2)
rele = OutputDevice(10)
rele2 = OutputDevice(6)

# Configuração LCD
lcd_rs = digitalio.DigitalInOut(board.D18)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D24)
lcd_d5 = digitalio.DigitalInOut(board.D23)
lcd_d6 = digitalio.DigitalInOut(board.D27)
lcd_d7 = digitalio.DigitalInOut(board.D22)
lcd_columns = 16
lcd_rows = 2

lcd = character_lcd.Character_LCD_Mono(
    lcd_rs, lcd_en,
    lcd_d4, lcd_d5, lcd_d6, lcd_d7,
    lcd_columns, lcd_rows
)

prediction_key = os.getenv("prediction_key")
endpoint = os.getenv("endpoint")
project_id = os.getenv("project_id")
iteration_name = os.getenv("iteration_name")

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

# Função para capturar imagem da câmera com tratamento de exceções
def capturar_imagem():
    try:
        cap = cv2.VideoCapture(0)
        time.sleep(2)  # Espera a câmera estabilizar
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, img_encoded = cv2.imencode('.jpg', frame)
            return img_encoded.tobytes()
        else:
            raise Exception("Falha na captura da imagem.")
    except Exception as e:
        print(f"Erro ao capturar imagem: {e}")
        return None

# Função para analisar imagem via Azure com tratamento de exceções
def analisar_imagem(image_bytes):
    try:
        results = predictor.classify_image(project_id, iteration_name, image_bytes)
        melhor_previsao = max(results.predictions, key=lambda p: p.probability)
        return melhor_previsao
    except Exception as e:
        print(f"Erro ao analisar imagem: {e}")
        return None

# Função para limpar e atualizar o LCD com responsividade
def atualizar_lcd(message, clear=True, sleep_time=1):
    try:
        if clear:
            lcd.clear()
        lcd.message = message
        sleep(sleep_time)  # Dá tempo para a leitura do LCD
    except Exception as e:
        print(f"Erro ao atualizar LCD: {e}")

# Função para ativar o relé por X segundos sem travar o loop principal
def ativar_rele_temporizado(rele_obj, tempo):
    def worker():
        rele_obj.off()  # Ativa o relé (liga o LED)
        sleep(tempo)
        rele_obj.on()   # Desativa o relé (desliga o LED)
    threading.Thread(target=worker).start()

# Função principal para lidar com o botão e realizar a captura e análise
def acao_botao():
    atualizar_lcd("Capturando...", clear=True)
    image_bytes = capturar_imagem()

    if not image_bytes:
        atualizar_lcd("Erro na\ncaptura!", clear=True, sleep_time=3)
        return

    atualizar_lcd("Analisando...", clear=True)
    melhor_previsao = analisar_imagem(image_bytes)

    if not melhor_previsao:
        atualizar_lcd("Erro na\nanálise!", clear=True, sleep_time=3)
        return

    tag = melhor_previsao.tag_name[:16]  # Limita a 16 caracteres
    prob = melhor_previsao.probability * 100

    if tag == "Plastico":
        atualizar_lcd(f"{tag}\n{prob:.1f}% certeza")
        ativar_rele_temporizado(rele, 8)
    elif tag == "Papel" or tag == "Papelao":
        atualizar_lcd(f"{tag}\n{prob:.1f}% certeza")
        ativar_rele_temporizado(rele2, 8)
    else:
        atualizar_lcd(f"{tag} indisp", clear=True, sleep_time=3)
        rele.on()  # Desliga o LED de Plástico
        rele2.on() # Desliga o LED de Papel/Papelão

    sleep(1)  # Pausa para a responsividade do LCD

# Estado inicial
atualizar_lcd("Aguardando\nbotao...", clear=True)

while True:
    # Liga os relés no estado inicial
    rele.on()
    rele2.on()

    if button.is_pressed:
        acao_botao()

    sleep(0.1)  # Evita uso excessivo da CPU
