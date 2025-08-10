import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import time
import os
import json
import psutil # Pro sledování systémových metrik
import cv2 # Pro detekci CUDA

import config # Import config pro sdílené proměnné

def process_in_background(func, *args, **kwargs):
    """Spustí funkci v samostatném vlákně, aby se zabránilo zamrznutí GUI."""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True  # Zajistí, že vlákno skončí při ukončení hlavního programu
    thread.start()
    return thread

def cancel_processing():
    """Zruší aktuálně probíhající zpracování videa a náhled."""
    config.stop_processing_event.set()
    config.stop_preview_event.set() # Zastavit i náhled
    messagebox.showinfo("Info", "Zpracování videa bylo zrušeno.")

def update_progress(value):
    """Aktualizuje progress bar v GUI (bezpečně z jiného vlákna)."""
    if config.progress_callback and config.gui_root:
        # Použijeme after() pro thread-safe aktualizaci GUI
        config.gui_root.after(0, lambda: config.progress_callback(value))

def check_processing_complete(thread, process_button_ref, cancel_button_ref):
    """Kontroluje, zda bylo zpracování dokončeno a obnoví stav GUI."""
    thread.join()  # Počká na dokončení vlákna zpracování
    
    if config.gui_root:
        config.gui_root.after(0, lambda: process_button_ref.config(state=tk.NORMAL))
        config.gui_root.after(0, lambda: cancel_button_ref.config(state=tk.DISABLED))
        config.gui_root.after(0, lambda: update_progress(0))
    
    # Vyčistit stop event pro další běh
    config.stop_processing_event.clear()

def check_video_file(path):
    """Zkontroluje, zda je cesta platný video soubor."""
    if not os.path.isfile(path):
        return False, "Soubor neexistuje."
    if not path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')): # Rozšířeno o .webm
        return False, "Nepodporovaný video formát."
    return True, ""

def show_help_dialog():
    """Zobrazí okno s nápovědou."""
    help_window = tk.Toplevel(config.gui_root)
    help_window.title("Nápověda")
    help_window.geometry("800x600")

    help_text = """
    Popis funkcí a návod k použití:

    1.  **Analyze Video**:
        *   Provede komplexní analýzu vlastností videa (rozmazanost, šum, jas, kontrast, pohyb).
        *   Navrhne optimální parametry pro různé režimy zpracování na základě analýzy.
        *   Otevře nové okno s detailními výsledky analýzy a umožní aplikovat navržené parametry do GUI.

    2.  **Processing Modes**:
        *   **Unsharp Mask**: Standardní technika ostření.
            *   Parametry: Amount Range (intenzita), Radius (poloměr rozostření).
        *   **Focus Mask**: Aplikuje ostření na celý snímek videa s proměnlivým poloměrem v čase.
            *   Parametry: Amount Range, Radius Range.
        *   **Smart Focus**: Inteligentní ostření na základě detekce hran a map důležitosti.
            *   Parametry: Amount Range, Radius Range, Edge Threshold Range, Saliency Threshold Range.
        *   **Denoise**: Odstranění šumu.
            *   Typy: Gaussian Blur (jednoduché), Non-Local Means (pokročilejší).
            *   Parametry (pro Non-Local Means): h (intenzita filtru), templateWindowSize, searchWindowSize.
        *   **Gradient Mask**: Aplikuje plynulý přechod mezi rozmazanými a ostřejšími oblastmi (např. vertikální gradient).
            *   Parametry: Blur Strength, Sharp Strength.
        *   **Stabilize Video**: Redukuje roztřesenost videozáznamu.
            *   Žádné specifické parametry v GUI.
        *   **Adaptive Focus**: Automaticky upravuje ostření na základě analýzy obsahu každého snímku. Nejlépe se používá po analýze videa.
            *   Žádné specifické parametry v GUI, interně využívá výsledky analýzy.
        *   **Color Selective Focus**: Zaostří pouze oblasti s určitou barvou.
            *   Parametry: Target Color (B,G,R), Tolerance (podobnost barev), Amount, Radius.
        *   **Color Correction**: Základní úpravy barev.
            *   Parametry: Brightness (Jas), Contrast (Kontrast), Saturation (Saturace), Hue (Odstín).

    3.  **Kroky použití**:
        *   **Vyberte vstupní soubor**: Klikněte na "Procházet" vedle "Vstupní soubor" a vyberte své video.
        *   **Vyberte výstupní soubor**: Klikněte na "Procházet" vedle "Výstupní soubor" a určete, kam se zpracované video uloží. (Volitelné: Pokud není nastaveno, automaticky se uloží s příponou "_edited.mp4").
        *   **Analyzovat (Doporučeno)**: Klikněte na tlačítko "Analyzovat video" pro získání podrobných informací o videu a navržených parametrech. Poté se můžete rozhodnout je použít.
        *   **Vyberte režim zpracování**: Vyberte režim ze sekce "Funkce". Pole parametrů se aktualizují na základě vybraného režimu.
        *   **Upravte parametry**: Ručně upravte parametry pro zvolený režim, nebo použijte "Aplikovat navržené parametry" z kroku analýzy.
        *   **Živý náhled (NOVINKA!)**:
            *   Klikněte na "Živý náhled" pro zobrazení aplikovaného efektu v reálném čase.
            *   **Výběr ROI**: V okně náhledu můžete myší nakreslit obdélník pro definování oblasti zájmu (Region of Interest - ROI). Zpracována bude pouze tato oblast. ROI bude zvýrazněna.
            *   **Odstranit ROI**: Klikněte na tlačítko "Vymazat ROI" v GUI pro vymazání vybrané oblasti a zpracování celého snímku.
            *   **Přepínač ROI v náhledu**: Pomocí zaškrtávacího políčka "Aplikovat ROI v živém náhledu" můžete ovládat, zda má být vybrané ROI aktivní během živého náhledu.
            *   **Interaktivní slidery**: V okně náhledu se nyní zobrazují slidery pro klíčové parametry, které můžete měnit v reálném čase.
            *   **Klávesové zkratky v náhledu (okno OpenCV)**:
                *   '**q**': Ukončení náhledu.
                *   '**s**': Uložení aktuálního zpracovaného snímku jako obrázku.
                *   '**r**': Resetování výběru ROI v okně náhledu (nakreslete nový).
        *   **Zpracovat video**: Klikněte na "Zpracovat video" pro spuštění celého zpracování a uložení výstupu.
        *   **Zrušit**: Klikněte na "Zrušit zpracování" pro zastavení jakéhokoli probíhajícího zpracování.
        *   **Uložit/Načíst nastavení**: Pomocí těchto tlačítek můžete ukládat a načítat aktuální nastavení parametrů GUI do/ze souboru JSON.
        *   **Povolit GPU akceleraci (CUDA)**: Zaškrtněte toto políčko pro pokus o použití GPU (CUDA) pro zpracování, pokud je k dispozici a OpenCV je s CUDA zkompilováno.

    4.  **Důležité poznámky**:
        *   Zpracování videa je výpočetně náročné. Velká videa nebo efekty s vysokou intenzitou budou trvat déle.
        *   Živý náhled zpracovává snímky se sníženou rychlostí pro lepší odezvu.
        *   Ujistěte se, že máte nainstalované potřebné kodeky (např. `libx264`).
    """
    text_widget = tk.Text(help_window, wrap="word", padx=10, pady=10)
    text_widget.insert("1.0", help_text)
    text_widget.config(state="disabled")  # Zamezí úpravám textu
    
    scrollbar = ttk.Scrollbar(help_window, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.pack(expand=True, fill="both")

def save_settings_to_file(settings):
    """Uloží aktuální nastavení do JSON souboru."""
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON soubory", "*.json")],
            title="Uložit nastavení"
        )
        if not file_path:
            return False
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
            
        messagebox.showinfo("Úspěch", f"Nastavení bylo úspěšně uloženo do {file_path}")
        return True
    except Exception as e:
        messagebox.showerror("Chyba", f"Chyba při ukládání nastavení: {e}")
        return False

def load_settings_from_file():
    """Načte nastavení z JSON souboru."""
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON soubory", "*.json")],
            title="Načíst nastavení"
        )
        if not file_path:
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        messagebox.showinfo("Úspěch", f"Nastavení bylo úspěšně načteno z {file_path}")
        return settings
    except Exception as e:
        messagebox.showerror("Chyba", f"Chyba při načítání nastavení: {e}")
        return None

def get_system_metrics():
    """Získá aktuální systémové metriky CPU a paměti."""
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking call
    mem_info = psutil.virtual_memory()
    mem_percent = mem_info.percent
    
    # Simple representation of usage for GUI indicators
    cpu_indicator_str = ""
    if cpu_percent < 25: cpu_indicator_str = "■□□□□"
    elif cpu_percent < 50: cpu_indicator_str = "■■□□□"
    elif cpu_percent < 75: cpu_indicator_str = "■■■□□"
    else: cpu_indicator_str = "■■■■□"
    
    mem_indicator_str = ""
    if mem_percent < 25: mem_indicator_str = "■□□□□"
    elif mem_percent < 50: mem_indicator_str = "■■□□□"
    elif mem_percent < 75: mem_indicator_str = "■■■□□"
    else: mem_indicator_str = "■■■■□"

    # GPU placeholder or real detection
    gpu_indicator_str = "N/A" 
    if config.CUDA_AVAILABLE:
        try:
            # Note: This is a very basic check. Real GPU usage requires more specific libraries
            # like pynvml for NVIDIA, or monitoring processes.
            # Here, we just check if a CUDA device is active.
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                # Assuming device 0 for simplicity
                # No direct way to get usage % from cv2.cuda alone, this is more for detection.
                gpu_indicator_str = "■■■■■" # Simply indicate presence
            else:
                gpu_indicator_str = "□□□□□ (No active device)"

        except Exception:
            gpu_indicator_str = "N/A (Error checking GPU)"
    
    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "cpu_indicator_str": cpu_indicator_str,
        "mem_indicator_str": mem_indicator_str,
        "gpu_indicator_str": gpu_indicator_str 
    }

def check_cuda_availability():
    """Zkontroluje, zda je CUDA k dispozici a nastaví globální příznak."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            config.CUDA_AVAILABLE = True
            print("CUDA je k dispozici a nalezena zařízení.")
        else:
            config.CUDA_AVAILABLE = False
            print("CUDA není k dispozici nebo nejsou nalezena žádná zařízení.")
    except AttributeError:
        config.CUDA_AVAILABLE = False
        print("Modul cv2.cuda není k dispozici (OpenCV nebylo zkompilováno s CUDA podporou).")
    except Exception as e:
        config.CUDA_AVAILABLE = False
        print(f"Chyba při kontrole CUDA dostupnosti: {e}")
