import cv2
import numpy as np
from ultralytics import YOLO
import time
import textwrap
import random
import threading
import os
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURAZIONE MODELLI ---
print("Caricamento modello Segmentazione (Oggetti)...")
model_seg = YOLO('yolov8n-seg.pt') 

print("Caricamento modello Pose (Scheletri Umani)...")
model_pose = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

# --- MEMVID INTEGRATION ---
# Forza l'uso della cache locale per evitare errori di rete
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

try:
    from memvid import MemvidEncoder
    MEMVID_AVAILABLE = True
    print("Memvid integrato: Visione Mnemonica attiva (modalità offline).")
except ImportError:
    MEMVID_AVAILABLE = False
    print("Memvid non trovato. Funzionalità di memoria disabilitata.")

# Variabili globali per la memoria e il threading
session_memory = []
memory_lock = threading.Lock()
running = True
last_encoded_count = 0
LIVE_VIDEO_FILE = "biomemory_live.mp4"
LIVE_INDEX_FILE = "biomemory_live_index.json"
VIDEO_REBUILD_INTERVAL = 10  # secondi

# --- BACKGROUND ENCODER THREAD ---
def background_encoder_thread():
    """Thread che ricostruisce periodicamente il video Memvid."""
    global last_encoded_count
    
    while running:
        time.sleep(VIDEO_REBUILD_INTERVAL)
        
        if not running:
            break
            
        with memory_lock:
            current_count = len(session_memory)
            if current_count > 0 and current_count != last_encoded_count:
                chunks_to_encode = session_memory.copy()
        
        if current_count > 0 and current_count != last_encoded_count:
            try:
                print(f"\n[MEMVID] Encoding {current_count} chunks...")
                encoder = MemvidEncoder()
                encoder.add_chunks(chunks_to_encode)
                encoder.build_video(LIVE_VIDEO_FILE, LIVE_INDEX_FILE)
                last_encoded_count = current_count
                print(f"[MEMVID] Video aggiornato: {LIVE_VIDEO_FILE}")
            except Exception as e:
                print(f"[MEMVID ERROR] {e}")

# --- IMPOSTAZIONI LOGICHE ---
prompt_interval = 3.0
last_check_time = 0
last_detected_state = None
current_prompt = "Inizializzazione sistema di analisi comportamentale..."

TEXT_W, TEXT_H = 900, 700 
QR_DISPLAY_SIZE = (512, 512)

# --- TRADUZIONI E MOOD ---
translations = {
    'person': 'soggetto umano',
    'cup': 'tazza',
    'cell phone': 'cellulare',
    'bottle': 'bottiglia',
    'chair': 'sedia',
    'laptop': 'computer portatile',
    'mouse': 'mouse',
    'keyboard': 'tastiera',
    'book': 'libro',
    'tv': 'schermo',
    'backpack': 'zainetto'
}

intros = [
    "L'analisi rileva:",
    "Il sensore rileva:",
    "La scena è composta da:",
    "La macchina rileva:"
]

verbs = [
    "è posizionato",
    "occupa lo spazio",
    "si trova",
    "risiede"
]

connectors = [
    "mentre si osserva",
    "insieme a",
    "affiancato da",
    "seguito da"
]

styles = [
    "Analisi comportamentale attiva.",
    "Resa tecnica documentaristica.",
    "Rilevamento postura completato.",
    "Estetica da sorveglianza ad alta definizione.",
    "Scena analizzata con algoritmi biometrici."
]

# --- FUNZIONI ---
def classify_pose(keypoints):
    """Analizza i keypoints per determinare la posa."""
    if keypoints is None or len(keypoints) == 0:
        return "in posa neutra"

    nose = keypoints[0]
    shoulder_l = keypoints[5]
    shoulder_r = keypoints[6]
    wrist_l = keypoints[9]
    wrist_r = keypoints[10]
    
    pose_desc = []
    hands_up = False
    
    if (wrist_l[1] > 0 and wrist_l[1] < nose[1]) or (wrist_r[1] > 0 and wrist_r[1] < nose[1]):
        pose_desc.append("con braccio alzato")
        hands_up = True
    elif (wrist_l[1] > 0 and wrist_l[1] < shoulder_l[1]) and (wrist_r[1] > 0 and wrist_r[1] < shoulder_r[1]):
        pose_desc.append("con entrambe le mani sollevate")
        hands_up = True

    if not hands_up:
        if abs(wrist_l[1] - nose[1]) < 50 or abs(wrist_r[1] - nose[1]) < 50:
            pose_desc.append("con mano al volto")

    if not pose_desc:
        return "in postura statica"
    
    return " ".join(pose_desc)

def get_spatial_position(box, frame_width):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "nel quadrante sinistro"
    elif center_x > (frame_width / 3) * 2:
        return "nel quadrante destro"
    else:
        return "al centro"

def generate_bio_narrative(detections):
    """Genera la narrativa bio dalla lista di detection."""
    if not detections:
        return "Segnale video attivo. Nessun soggetto rilevante identificato. Scena vuota."

    groups = {'nel quadrante sinistro': [], 'al centro': [], 'nel quadrante destro': []}
    
    for item in detections:
        name = item['name']
        pos = item['pos']
        pose = item.get('pose', '')
        
        ita_name = translations.get(name, name)
        
        if name == 'person' and pose:
            full_desc = f"{ita_name} {pose}"
        else:
            full_desc = ita_name
            
        groups[pos].append(full_desc)

    prompt_parts = [random.choice(intros)]
    active_zones = [k for k, v in groups.items() if v]
    
    sentences = []
    for zone in active_zones:
        objs = groups[zone]
        if len(objs) > 1:
            obj_text = ", ".join(objs[:-1]) + " e " + objs[-1]
        else:
            obj_text = objs[0]
        sentences.append(f"{zone} {random.choice(verbs)} {obj_text}")
    
    if len(sentences) == 1:
        prompt_parts.append(sentences[0])
    else:
        full_body = sentences[0]
        for i in range(1, len(sentences)):
            connector = random.choice(connectors)
            full_body += f", {connector} {sentences[i]}"
        prompt_parts.append(full_body)

    prompt_parts.append(". " + random.choice(styles))
    return " ".join(prompt_parts)

def draw_text_with_pil(img, text, position, font_size=30, color=(220, 220, 220)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    wrapped_text = textwrap.wrap(text, width=45)
    y = position[1]
    for line in wrapped_text:
        draw.text((position[0], y), line, font=font, fill=color)
        y += font_size + 12
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- AVVIO THREAD BACKGROUND ---
if MEMVID_AVAILABLE:
    encoder_thread = threading.Thread(target=background_encoder_thread, daemon=True)
    encoder_thread.start()
    print("Thread encoder Memvid avviato.")

# --- VIDEO PLAYBACK STATE ---
video_playback_cap = None
video_frame_index = 0
last_video_check = 0

# --- MAIN LOOP ---
print("Sistema Biometrico Avviato. Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    # 1. ESEGUIAMO ENTRAMBI I MODELLI
    res_seg = model_seg(frame, verbose=False, conf=0.4)
    res_pose = model_pose(frame, verbose=False, conf=0.5)

    # 2. VISUALIZZAZIONE COMBINATA (Schermo 1: Visione Sensoriale)
    annotated_frame = res_seg[0].plot() 
    annotated_frame = res_pose[0].plot(img=annotated_frame, boxes=False)

    # 3. LOGICA TESTUALE (Timer 3s)
    if time.time() - last_check_time > prompt_interval:
        
        current_items = []
        
        boxes = res_seg[0].boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                name = model_seg.names[cls_id]
                pos_str = get_spatial_position(coords, w)
                
                item_data = {'name': name, 'pos': pos_str, 'pose': ''}
                
                if name == 'person':
                    if res_pose[0].keypoints is not None and len(res_pose[0].keypoints.data) > 0:
                        try:
                            kpts = res_pose[0].keypoints.data[i].cpu().numpy()
                            pose_text = classify_pose(kpts)
                            item_data['pose'] = pose_text
                        except IndexError:
                            pass
                
                current_items.append((item_data['pos'], item_data['name'], item_data['pose']))

        current_items = sorted(list(set(current_items)))
        narrative_input = [{'pos': x[0], 'name': x[1], 'pose': x[2]} for x in current_items]

        if current_items != last_detected_state:
            current_prompt = generate_bio_narrative(narrative_input)
            print(f"[BIO-UPDATE]: {current_prompt}")
            
            if MEMVID_AVAILABLE:
                timestamp = time.strftime("%H:%M:%S")
                with memory_lock:
                    session_memory.append(f"[{timestamp}] {current_prompt}")

            last_detected_state = current_items
        
        last_check_time = time.time()

    # 4. VISUALIZZAZIONE TESTO (Schermo 2: Visione Testuale)
    text_screen_base = np.zeros((TEXT_H, TEXT_W, 3), dtype=np.uint8)
    text_screen = draw_text_with_pil(text_screen_base, current_prompt, (50, 60), font_size=32)

    # 5. VISIONE MNEMONICA (Schermo 3: Playback del video QR)
    qr_display = np.zeros((QR_DISPLAY_SIZE[1], QR_DISPLAY_SIZE[0], 3), dtype=np.uint8)
    
    if MEMVID_AVAILABLE and os.path.exists(LIVE_VIDEO_FILE):
        # Riapri il video se è stato aggiornato (controllo ogni 2 sec)
        if time.time() - last_video_check > 2:
            if video_playback_cap is not None:
                video_playback_cap.release()
            video_playback_cap = cv2.VideoCapture(LIVE_VIDEO_FILE)
            video_frame_index = 0
            last_video_check = time.time()
        
        if video_playback_cap is not None and video_playback_cap.isOpened():
            ret_qr, qr_frame = video_playback_cap.read()
            if ret_qr:
                qr_display = cv2.resize(qr_frame, QR_DISPLAY_SIZE)
                video_frame_index += 1
            else:
                # Loop: torna all'inizio
                video_playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_frame_index = 0
    else:
        # Placeholder se il video non esiste ancora
        cv2.putText(qr_display, "Attesa encoding...", (50, 256), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    cv2.imshow('Schermo 1: Visione Sensoriale', annotated_frame)
    cv2.imshow('Schermo 2: Visione Testuale', text_screen)
    cv2.imshow('Schermo 3: Visione Mnemonica', qr_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
running = False
cap.release()
if video_playback_cap is not None:
    video_playback_cap.release()
cv2.destroyAllWindows()

# Salvataggio finale con timestamp
if MEMVID_AVAILABLE and session_memory:
    print(f"\n--- MEMVID: Salvataggio finale di {len(session_memory)} ricordi ---")
    try:
        encoder = MemvidEncoder()
        
        filename = f"biomemory_{int(time.time())}.mp4"
        indexname = filename.replace(".mp4", "_index.json")
        
        print("Codifica in corso...")
        encoder.add_chunks(session_memory)
        encoder.build_video(filename, indexname)
        print(f"Memoria salvata con successo in: {filename}")
        print(f"Indice salvato in: {indexname}")
        
    except Exception as e:
        print(f"ERRORE salvataggio Memvid: {e}")