import threading
import numpy as np

# Globální event pro signalizaci zastavení procesování/náhledů
stop_processing_event = threading.Event()
stop_preview_event = threading.Event()

# Globální proměnná pro ukládání aktuálního průběhu (0-100)
current_progress = 0

# Globální callback funkce pro aktualizaci progress baru v GUI
progress_callback = None

# Globální proměnná pro ukládání vybrané Region of Interest (ROI)
# Formát: (x, y, w, h) nebo None, pokud není vybráno žádné ROI
selected_roi = None

# Příznak pro indikaci, zda má náhled aplikovat ROI
apply_roi_in_preview = False

# Globální reference na Tkinter root okno (pro thread-safe GUI aktualizace)
gui_root = None

# Výchozí barvy pro selektivní zaostření (BGR formát)
DEFAULT_TARGET_COLOR_BGR = [0, 0, 255] # Výchozí je červená v BGR

# Prahové hodnoty pro analýzu rozmazanosti (pro doporučování parametrů)
BLURRINESS_LOW = 50
BLURRINESS_MEDIUM = 200

# Prahové hodnoty pro analýzu šumu (pro doporučování parametrů)
NOISE_HIGH = 15

# Prahové hodnoty pro analýzu jasu (pro doporučování parametrů)
BRIGHTNESS_LOW = 50
BRIGHTNESS_HIGH = 200

# Prahové hodnoty pro analýzu kontrastu (pro doporučování parametrů)
CONTRAST_LOW = 40
CONTRAST_HIGH = 80

# Video kodek a audio kodek pro výstup
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"

# Názvy oken náhledu
PREVIEW_ORIGINAL_WINDOW_NAME = "Původní video (Q=konec, S=uložit, R=reset ROI)"
PREVIEW_PROCESSED_WINDOW_NAME = "Náhled efektu (Q=konec, S=uložit, R=reset ROI)"

# Výchozí maximální šířka pro náhled videa (pro úsporu výkonu)
PREVIEW_MAX_WIDTH = 800

# Frekvence aktualizace systémových metrik (v ms)
SYSTEM_METRICS_UPDATE_INTERVAL_MS = 1000

# Typy odšumovacích algoritmů
DENOISE_ALGO_GAUSSIAN = "gaussian"
DENOISE_ALGO_NL_MEANS = "non_local_means"

# Konstanty pro color correction
COLOR_CORRECTION_BRIGHTNESS_MIN = -100
COLOR_CORRECTION_BRIGHTNESS_MAX = 100
COLOR_CORRECTION_BRIGHTNESS_DEFAULT = 0

COLOR_CORRECTION_CONTRAST_MIN = 0
COLOR_CORRECTION_CONTRAST_MAX = 3.0 # Factor
COLOR_CORRECTION_CONTRAST_DEFAULT = 1.0

COLOR_CORRECTION_SATURATION_MIN = 0
COLOR_CORRECTION_SATURATION_MAX = 2.0 # Factor
COLOR_CORRECTION_SATURATION_DEFAULT = 1.0

COLOR_CORRECTION_HUE_MIN = -180
COLOR_CORRECTION_HUE_MAX = 180
COLOR_CORRECTION_HUE_DEFAULT = 0

# --- NOVÉ PRO CUDA ---
USE_CUDA_GPU = True # Globální příznak, zda používat CUDA (nastaví se v GUI)
CUDA_AVAILABLE = True # Indikuje, zda je CUDA skutečně detekována a dostupná