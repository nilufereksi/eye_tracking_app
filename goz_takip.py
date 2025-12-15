import cv2
import numpy as np
import math
from collections import deque, Counter
import pyautogui
import time

# --- GÜVENLİK VE AYARLAR ---
# DİKKAT: Program kapanıyorsa bu ayarı False yapıyoruz.
# Programı durdurmak için konsola gelip CTRL+C yapman gerekebilir.
pyautogui.FAILSAFE = False 

last_action = None 

# Göz Kırpma Ayarları (Senin ayarladığın değerler)
BLINK_THRESHOLD = 8  # Senin için ideal sayı
blink_counter = 0     
click_cooldown = False 

# --- ARAYÜZ ÇİZİM FONKSİYONU ---
def arayuz_ciz(frame, aktif_eylem, blink_durumu):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    renk_aktif = (0, 255, 0)      
    renk_pasif = (30, 30, 30)     
    renk_cerceve = (200, 200, 200)
    renk_click = (0, 0, 255) 
    
    pad, btn_w, btn_h = 20, 120, 80
    butonlar = {
        "YUKARI": {"coords": (w//2 - btn_w//2, pad, w//2 + btn_w//2, pad + btn_h)},
        "ASAGI":  {"coords": (w//2 - btn_w//2, h - pad - btn_h, w//2 + btn_w//2, h - pad)},
        "SOL":    {"coords": (pad, h//2 - btn_h//2, pad + btn_w, h//2 + btn_h//2)},
        "SAG":    {"coords": (w - pad - btn_w, h//2 - btn_h//2, w - pad, h//2 + btn_h//2)}
    }

    cx, cy = w // 2, h // 2
    
    if blink_durumu > 0:
        doluluk = (blink_durumu / BLINK_THRESHOLD) * 360
        renk_gosterge = renk_click if blink_durumu >= BLINK_THRESHOLD else (0, 255, 255)
        cv2.ellipse(overlay, (cx, cy), (30, 30), 0, 0, doluluk, renk_gosterge, 4)
        if blink_durumu >= BLINK_THRESHOLD:
            cv2.putText(overlay, "TIKLANDI!", (cx-60, cy-40), cv2.FONT_HERSHEY_BOLD, 1, renk_click, 2)
    else:
        cv2.line(overlay, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(overlay, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

    for key, val in butonlar.items():
        x1, y1, x2, y2 = val["coords"]
        is_active = (key in aktif_eylem)
        
        if is_active:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), renk_aktif, -1)
            text_color = (0, 0, 0)
            scale = 0.8
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), renk_pasif, -1)
            text_color = (255, 255, 255)
            scale = 0.6
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), renk_cerceve, 1)
        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(overlay, key, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, scale, text_color, 1)

    cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
    durum_yazisi = "CLICK" if blink_durumu >= BLINK_THRESHOLD else aktif_eylem
    cv2.putText(overlay, f"DURUM: {durum_yazisi}", (20, h-12), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def nothing(x): pass

def tetikle_eylem(aci):
    if 0 <= aci < 45: return "SAG-UST"
    elif 45 <= aci < 135: return "YUKARI"
    elif 135 <= aci < 225: return "SOL-UST"
    elif 225 <= aci < 315: return "ASAGI"
    else: return "SAG-ALT"

def klavye_simulasyonu(eylem):
    global last_action
    if eylem == last_action or eylem == "Bekleniyor..." or eylem == "CLICK": return

    try:
        if eylem == "YUKARI": pyautogui.press('up') 
        elif eylem == "ASAGI": pyautogui.press('down') 
        elif "SAG" in eylem: pyautogui.press('right') 
        elif "SOL" in eylem: pyautogui.press('left') 
        print(f"Tuş Basıldı: {eylem}")
    except Exception as e:
        print(f"Klavye Hatası: {e}")
        
    last_action = eylem

# --- INIT ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cv2.namedWindow('Ayarlar')
cv2.resizeWindow('Ayarlar', 300, 100)
cv2.createTrackbar('Isik Esigi', 'Ayarlar', 40, 255, nothing)

karar_tamponu = deque(maxlen=15)

print("Program Baslatildi. Kapatmak icin 'q' tusuna basin.")

while True:
    try: # --- HATA YAKALAMA BLOĞU ---
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_val = cv2.getTrackbarPos('Isik Esigi', 'Ayarlar')
        
        stabil_aksiyon = "Bekleniyor..." 
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            blink_counter = 0
            click_cooldown = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # GÖZ TESPİTİ (Hassasiyet 20 yapıldı)
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 20)
            
            # --- GÖZ KIRPMA (BLINK) ---
            if len(eyes) == 0:
                blink_counter += 1
                # Konsolda sayacı görmek istersen:
                # print(f"Sayac: {blink_counter}") 
            else:
                blink_counter = 0
                click_cooldown = False 

            if blink_counter >= BLINK_THRESHOLD and not click_cooldown:
                # ENTER veya DOUBLE CLICK (Buradan değiştirebilirsin)
                pyautogui.press('enter') 
                # pyautogui.doubleClick() 
                
                print(">>> CLICK BASILDI! <<<")
                click_cooldown = True 
                stabil_aksiyon = "CLICK"

            # --- GÖZ BEBEĞİ TAKİBİ ---
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                
                _, threshold = cv2.threshold(eye_roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                
                if len(contours) > 0:
                    pupil = contours[0]
                    M = cv2.moments(pupil)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(roi_color, (ex + cx, ey + cy), 3, (0, 0, 255), -1)
                        
                        eye_center_x, eye_center_y = ew // 2, eh // 2
                        angle_deg = math.degrees(math.atan2(-(cy - eye_center_y), cx - eye_center_x))
                        if angle_deg < 0: angle_deg += 360
                            
                        ham_aksiyon = tetikle_eylem(angle_deg)
                        karar_tamponu.append(ham_aksiyon)
                        
                        if len(karar_tamponu) > 0:
                            stabil_aksiyon = Counter(karar_tamponu).most_common(1)[0][0]
                        
                        cv2.imshow("Threshold (Ayar)", threshold)

        if stabil_aksiyon != "CLICK": 
            klavye_simulasyonu(stabil_aksiyon)
        
        arayuz_ciz(frame, stabil_aksiyon, blink_counter)
        cv2.imshow('Goz Takip Projesi', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty('Goz Takip Projesi', cv2.WND_PROP_VISIBLE) < 1: break
        
    except Exception as e:
        # Hata olursa program kapanmasın, hatayı yazsın
        print(f"!!! BEKLENMEYEN HATA: {e}")
        time.sleep(1) # Hata spamını engellemek için biraz bekle

cap.release()
cv2.destroyAllWindows()