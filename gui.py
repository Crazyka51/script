import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import colorchooser
import json
import os
import threading
import time

# Import vlastních modulů
import config
import utils
import video_analyzer
import video_processor
import preview_manager

# Globální proměnné pro GUI widgety (přístupné různými funkcemi)
input_file = None
output_file = None
mode_var = None
denoise_algo_var = None 
amount_min = None
amount_max = None
radius_entry = None
radius_min = None
radius_max = None
edge_min = None
edge_max = None
saliency_min = None
saliency_max = None
blur_strength = None
sharp_strength = None
color_entry = None
tolerance_entry = None
brightness_entry = None 
contrast_entry = None 
saturation_entry = None 
hue_entry = None 
progress_var = None
progress_bar = None
cancel_button = None
process_button = None
roi_checkbox_var = None
mode_info_text = None
params_frame = None
color_picker_button = None
gpu_checkbox_var = None # NOVÉ: proměnná pro GPU checkbox
cpu_label, mem_label, gpu_label = None, None, None # Pro systémové metriky
nlm_h_entry, nlm_template_size_entry, nlm_search_size_entry = None, None, None # Pro NLM Denoise
denoise_algo_label, denoise_algo_dropdown = None, None # Pro denoise algorithm selection
use_cuda_var = None

# Cache pro navržené parametry z analýzy
_cached_suggested_params = {}

def select_input_file():
    path = filedialog.askopenfilename(filetypes=[("Video soubory", "*.mp4;*.avi;*.mov;*.mkv;*.webm")])
    if path:
        input_file.set(path)
        if not output_file.get(): 
            base, ext = os.path.splitext(path)
            output_file.set(f"{base}_edited.mp4")

def select_output_file():
    path = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        filetypes=[("MP4 soubory", "*.mp4"), ("AVI soubory", "*.avi"), ("MOV soubory", "*.mov"), ("MKV soubory", "*.mkv"), ("WEBM soubory", "*.webm")],
        title="Uložit video jako"
    )
    if path:
        output_file.set(path)

def clear_roi():
    config.selected_roi = None
    messagebox.showinfo("ROI", "Vybraná oblast zájmu (ROI) byla zrušena.")

def update_progress_bar_gui(value):
    """Callback pro aktualizaci progress baru v GUI."""
    progress_var.set(value)
    config.gui_root.update_idletasks()

def update_system_metrics_gui():
    """Aktualizuje systémové metriky v GUI."""
    metrics = utils.get_system_metrics()
    
    cpu_label.config(text=f"CPU: {metrics['cpu_percent']:.1f}% ({metrics['cpu_indicator_str']})")
    mem_label.config(text=f"Memory: {metrics['mem_percent']:.1f}% ({metrics['mem_indicator_str']})")
    gpu_label.config(text=f"GPU: {metrics['gpu_indicator_str']}") 
    
    # Naplánovat další aktualizaci
    config.gui_root.after(config.SYSTEM_METRICS_UPDATE_INTERVAL_MS, update_system_metrics_gui)

def process_video_gui_handler():
    """Obsluhuje kliknutí na tlačítko 'Zpracovat video'."""
    input_path = input_file.get()
    output_path = output_file.get()
    mode = mode_var.get()
    use_cuda_enabled = gpu_checkbox_var.get() # Získat stav GPU checkboxu

    is_valid, msg = utils.check_video_file(input_path)
    if not is_valid:
        messagebox.showerror("Chyba", f"Vstupní soubor není platný: {msg}")
        return

    # Shromáždit aktuální parametry GUI
    current_gui_params = {}
    try:
        # Obecné parametry pro ostření
        current_gui_params["amount_range"] = (float(amount_min.get()), float(amount_max.get()))
        current_gui_params["radius"] = int(radius_entry.get()) if radius_entry.get() else 5

        # Specifické parametry pro režimy
        if mode in ["focus_mask", "smart_focus"]:
            current_gui_params["radius_range"] = (int(radius_min.get()), int(radius_max.get()))
        if mode == "smart_focus":
            current_gui_params["edge_threshold_range"] = (int(edge_min.get()), int(edge_max.get()))
            current_gui_params["saliency_threshold_range"] = (float(saliency_min.get()), float(saliency_max.get()))
        if mode == "denoise":
            current_gui_params["denoise_type"] = denoise_algo_var.get()
            current_gui_params["strength"] = int(radius_entry.get()) if radius_entry.get() else 10 # Gaussian strength
            current_gui_params["nlm_h"] = int(nlm_h_entry.get()) if nlm_h_entry.get() else 10
            current_gui_params["nlm_template_size"] = int(nlm_template_size_entry.get()) if nlm_template_size_entry.get() else 7
            current_gui_params["nlm_search_size"] = int(nlm_search_size_entry.get()) if nlm_search_size_entry.get() else 21
        if mode == "gradient_mask":
            current_gui_params["blur_strength"] = int(blur_strength.get())
            current_gui_params["sharp_strength"] = float(sharp_strength.get())
        if mode == "color_selective_focus":
            current_gui_params["target_color"] = [int(x) for x in color_entry.get().split(",")]
            current_gui_params["tolerance"] = int(tolerance_entry.get())
            current_gui_params["amount"] = float(amount_min.get()) 
            current_gui_params["radius"] = int(radius_entry.get()) 
        if mode == "color_correction":
            current_gui_params["brightness"] = float(brightness_entry.get())
            current_gui_params["contrast"] = float(contrast_entry.get())
            current_gui_params["saturation"] = float(saturation_entry.get())
            current_gui_params["hue"] = float(hue_entry.get())
        
        # Předat cachované navržené parametry, pokud jsou k dispozici (zejména pro Adaptive Focus)
        if _cached_suggested_params:
            current_gui_params['suggested_params'] = _cached_suggested_params

    except ValueError as e:
        messagebox.showerror("Chyba", f"Neplatná hodnota parametru: {e}")
        return

    # Deaktivovat tlačítko zpracování, aktivovat zrušení
    process_button.config(state=tk.DISABLED)
    cancel_button.config(state=tk.NORMAL)

    # Nastavit globální progress callback
    config.progress_callback = update_progress_bar_gui

    # Spustit zpracování ve vlákně na pozadí
    process_thread = threading.Thread(
        target=lambda: _run_processing_and_report(
            input_path, output_path, mode, current_gui_params, config.selected_roi, use_cuda_enabled
        )
    )
    process_thread.daemon = True
    process_thread.start()

    # Spustit vlákno pro sledování dokončení a opětovné povolení tlačítek
    monitor_thread = threading.Thread(
        target=utils.check_processing_complete, 
        args=(process_thread, process_button, cancel_button)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

def _run_processing_and_report(input_path, output_path, mode, params, roi, use_cuda_enabled):
    """Pomocná funkce pro spuštění zpracování videa a hlášení úspěchu/selhání."""
    success = video_processor.process_video_main(input_path, output_path, mode, params, roi, use_cuda_enabled)
    
    if config.gui_root: 
        if success:
            if not config.stop_processing_event.is_set(): 
                config.gui_root.after(0, lambda: messagebox.showinfo("Úspěch", f"Video bylo úspěšně zpracováno a uloženo do {output_path}"))
        else:
            if not config.stop_processing_event.is_set(): 
                 config.gui_root.after(0, lambda: messagebox.showerror("Chyba", "Zpracování videa se nezdařilo. Zkontrolujte konzoli pro detaily."))

def live_preview_gui_handler():
    """Obsluhuje kliknutí na tlačítko 'Živý náhled'."""
    input_path = input_file.get()
    mode = mode_var.get()
    use_cuda_for_preview = gpu_checkbox_var.get() # Získat stav GPU checkboxu pro náhled

    is_valid, msg = utils.check_video_file(input_path)
    if not is_valid:
        messagebox.showerror("Chyba", f"Vstupní soubor není platný: {msg}")
        return
    
    # Získat aktuální parametry GUI pro náhled
    current_preview_params = {}
    try:
        current_preview_params["amount_range"] = (float(amount_min.get()), float(amount_max.get()))
        current_preview_params["radius"] = int(radius_entry.get()) if radius_entry.get() else 5

        if mode in ["focus_mask", "smart_focus"]:
            current_preview_params["radius_range"] = (int(radius_min.get()), int(radius_max.get()))
        if mode == "smart_focus":
            current_preview_params["edge_threshold_range"] = (int(edge_min.get()), int(edge_max.get()))
            current_preview_params["saliency_threshold_range"] = (float(saliency_min.get()), float(saliency_max.get()))
        if mode == "denoise":
            current_preview_params["denoise_type"] = denoise_algo_var.get()
            current_preview_params["strength"] = int(radius_entry.get()) # Gaussian strength
            current_preview_params["nlm_h"] = int(nlm_h_entry.get()) if nlm_h_entry.get() else 10
            current_preview_params["nlm_template_size"] = int(nlm_template_size_entry.get()) if nlm_template_size_entry.get() else 7
            current_preview_params["nlm_search_size"] = int(nlm_search_size_entry.get()) if nlm_search_size_entry.get() else 21
        if mode == "gradient_mask":
            current_preview_params["blur_strength"] = int(blur_strength.get())
            current_preview_params["sharp_strength"] = float(sharp_strength.get())
        if mode == "color_selective_focus":
            current_preview_params["target_color"] = [int(x) for x in color_entry.get().split(",")]
            current_preview_params["tolerance"] = int(tolerance_entry.get())
            current_preview_params["amount"] = float(amount_min.get())
            current_preview_params["radius"] = int(radius_entry.get())
        if mode == "color_correction":
            current_preview_params["brightness"] = float(brightness_entry.get())
            current_preview_params["contrast"] = float(contrast_entry.get())
            current_preview_params["saturation"] = float(saturation_entry.get())
            current_preview_params["hue"] = float(hue_entry.get())

        if _cached_suggested_params:
            current_preview_params['suggested_params'] = _cached_suggested_params

    except ValueError as e:
        messagebox.showerror("Chyba", f"Neplatná hodnota parametru pro náhled: {e}")
        return

    config.apply_roi_in_preview = roi_checkbox_var.get()

    threading.Thread(
        target=preview_manager.live_preview_video,
        args=(input_path, mode, current_preview_params, use_cuda_for_preview)
    ).start()

def analyze_and_suggest_gui_handler():
    """Obsluhuje kliknutí na tlačítko 'Analyzovat video'."""
    input_path = input_file.get()

    is_valid, msg = utils.check_video_file(input_path)
    if not is_valid:
        messagebox.showerror("Chyba", f"Vstupní soubor není platný: {msg}")
        return

    process_button.config(state=tk.DISABLED)
    cancel_button.config(state=tk.DISABLED) 

    config.progress_callback = update_progress_bar_gui

    def run_analysis_task():
        global _cached_suggested_params
        try:
            analysis_results = video_analyzer.enhanced_video_analysis(input_path)
            if config.stop_processing_event.is_set(): 
                config.stop_processing_event.clear() 
                config.gui_root.after(0, lambda: messagebox.showinfo("Info", "Analýza videa byla zrušena."))
                return

            suggested_params = video_analyzer.adaptive_parameter_suggestion(analysis_results)
            _cached_suggested_params = suggested_params 

            config.gui_root.after(0, lambda: show_analysis_results_dialog(analysis_results, suggested_params))
        except Exception as e:
            config.gui_root.after(0, lambda: messagebox.showerror("Chyba", f"Došlo k chybě během analýzy: {e}"))
            print(f"Analysis error: {e}") 
        finally:
            config.gui_root.after(0, lambda: process_button.config(state=tk.NORMAL))
            config.gui_root.after(0, lambda: cancel_button.config(state=tk.DISABLED)) 
            config.gui_root.after(0, lambda: update_progress_bar_gui(0)) 

    threading.Thread(target=run_analysis_task).start()

def show_analysis_results_dialog(analysis_results, suggested_params):
    """Zobrazí výsledky analýzy a doporučené parametry v novém okně."""
    results_window = tk.Toplevel(config.gui_root)
    results_window.title("Výsledky analýzy videa")
    results_window.geometry("600x600") 
    results_window.configure(bg='#000000')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Toplevel", background='#000000')

    notebook = ttk.Notebook(results_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    analysis_tab = ttk.Frame(notebook, style='TFrame')
    params_tab = ttk.Frame(notebook, style='TFrame')
    notebook.add(analysis_tab, text="Analýza")
    notebook.add(params_tab, text="Doporučené parametry")
    
    # Obsah záložky Analýza
    ttk.Label(analysis_tab, text="Výsledky analýzy videa", font=("Helvetica", 12, "bold"), style='TLabel').grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
    
    metrics_display = [
        ("Rozmazanost:", analysis_results['blurriness'], ".2f"),
        ("Úroveň šumu:", analysis_results['noise'], ".2f"),
        ("Průměrný jas:", analysis_results['brightness'], ".2f"),
        ("Kontrast:", analysis_results['contrast'], ".2f"),
        ("Pohyb:", analysis_results['motion'], ".2f")
    ]

    for i, (label_text, value, fmt) in enumerate(metrics_display):
        ttk.Label(analysis_tab, text=label_text, style='TLabel').grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(analysis_tab, text=f"{value:{fmt}}", style='TLabel').grid(row=i+1, column=1, padx=10, pady=5, sticky="w")
    
    interpretation_frame = ttk.LabelFrame(analysis_tab, text="Interpretace", style='TLabelframe')
    interpretation_frame.grid(row=len(metrics_display)+1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
    
    interpretation_text = ""
    if analysis_results['blurriness'] < config.BLURRINESS_LOW:
        interpretation_text += "Video je relativně ostré. "
    elif analysis_results['blurriness'] < config.BLURRINESS_MEDIUM:
        interpretation_text += "Video má střední úroveň rozmazanosti. "
    else:
        interpretation_text += "Video je značně rozmazané. "
        
    if analysis_results['noise'] > config.NOISE_HIGH:
        interpretation_text += "Detekována vysoká úroveň šumu. "
    else:
        interpretation_text += "Úroveň šumu je přijatelná. "
        
    if analysis_results['brightness'] < config.BRIGHTNESS_LOW:
        interpretation_text += "Video je poměrně tmavé. "
    elif analysis_results['brightness'] > config.BRIGHTNESS_HIGH:
        interpretation_text += "Video je velmi světlé. "
    else:
        interpretation_text += "Jas videa je v optimálním rozsahu. "
        
    if analysis_results['contrast'] < config.CONTRAST_LOW:
        interpretation_text += "Video má nízký kontrast. "
    elif analysis_results['contrast'] > config.CONTRAST_HIGH:
        interpretation_text += "Video má vysoký kontrast. "
    else:
        interpretation_text += "Kontrast videa je v optimálním rozsahu. "
    
    if analysis_results['motion'] > 0.5: 
        interpretation_text += "Detekován významný pohyb, stabilizace může být užitečná."
    else:
        interpretation_text += "Pohyb ve videu je minimální."

    ttk.Label(interpretation_frame, text=interpretation_text, wraplength=550, style='TLabel').pack(padx=10, pady=10)
    
    # Obsah záložky Parametry
    ttk.Label(params_tab, text="Doporučené parametry", font=("Helvetica", 12, "bold"), style='TLabel').grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
    
    param_display = [
        ("Amount rozsah:", suggested_params['amount_range']),
        ("Radius rozsah:", suggested_params['radius_range']),
        ("Edge threshold rozsah:", suggested_params['edge_threshold_range']),
        ("Saliency threshold rozsah:", suggested_params['saliency_threshold_range']),
        ("Denoise strength (Gauss):", suggested_params['denoise_strength']),
        ("NLM h:", suggested_params['nlm_h']),
        ("NLM TemplateSize:", suggested_params['nlm_template_size']),
        ("NLM SearchSize:", suggested_params['nlm_search_size']),
        ("Blur strength (GradMask):", suggested_params['blur_strength']),
        ("Sharp strength (GradMask):", suggested_params['sharp_strength']),
        ("Suggested Radius (Single):", suggested_params['suggested_radius_entry_val']),
        ("Brightness (Color Corr):", suggested_params['brightness']),
        ("Contrast (Color Corr):", suggested_params['contrast']),
        ("Saturation (Color Corr):", suggested_params['saturation']),
        ("Hue (Color Corr):", suggested_params['hue'])
    ]

    for i, (label_text, value) in enumerate(param_display):
        ttk.Label(params_tab, text=label_text, style='TLabel').grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(params_tab, text=str(value), style='TLabel').grid(row=i+1, column=1, padx=10, pady=5, sticky="w")
    
    def apply_suggested_params():
        amount_min.delete(0, tk.END)
        amount_min.insert(0, str(suggested_params['amount_range'][0]))
        amount_max.delete(0, tk.END)
        amount_max.insert(0, str(suggested_params['amount_range'][1]))
        
        radius_min.delete(0, tk.END)
        radius_min.insert(0, str(suggested_params['radius_range'][0]))
        radius_max.delete(0, tk.END)
        radius_max.insert(0, str(suggested_params['radius_range'][1]))
        
        edge_min.delete(0, tk.END)
        edge_min.insert(0, str(suggested_params['edge_threshold_range'][0]))
        edge_max.delete(0, tk.END)
        edge_max.insert(0, str(suggested_params['edge_threshold_range'][1]))
        
        saliency_min.delete(0, tk.END)
        saliency_min.insert(0, str(suggested_params['saliency_threshold_range'][0]))
        saliency_max.delete(0, tk.END)
        saliency_max.insert(0, str(suggested_params['saliency_threshold_range'][1]))
        
        radius_entry.delete(0, tk.END)
        radius_entry.insert(0, str(suggested_params['suggested_radius_entry_val']))

        nlm_h_entry.delete(0, tk.END)
        nlm_h_entry.insert(0, str(suggested_params['nlm_h']))
        nlm_template_size_entry.delete(0, tk.END)
        nlm_template_size_entry.insert(0, str(suggested_params['nlm_template_size']))
        nlm_search_size_entry.delete(0, tk.END)
        nlm_search_size_entry.insert(0, str(suggested_params['nlm_search_size']))
        
        blur_strength.delete(0, tk.END)
        blur_strength.insert(0, str(suggested_params['blur_strength']))
        
        sharp_strength.delete(0, tk.END)
        sharp_strength.insert(0, str(suggested_params['sharp_strength']))

        brightness_entry.delete(0, tk.END)
        brightness_entry.insert(0, str(suggested_params['brightness']))
        contrast_entry.delete(0, tk.END)
        contrast_entry.insert(0, str(suggested_params['contrast']))
        saturation_entry.delete(0, tk.END)
        saturation_entry.insert(0, str(suggested_params['saturation']))
        hue_entry.delete(0, tk.END)
        hue_entry.insert(0, str(suggested_params['hue']))
        
        messagebox.showinfo("Úspěch", "Doporučené parametry byly aplikovány do GUI.")
        results_window.destroy()
    
    ttk.Button(params_tab, text="Použít tyto parametry do GUI", command=apply_suggested_params, style='TButton').grid(row=len(param_display)+1, column=0, columnspan=2, pady=20)


def save_settings_gui_handler():
    """Uloží aktuální nastavení GUI."""
    settings = {
        "mode": mode_var.get(),
        "denoise_algo": denoise_algo_var.get(),
        "amount_min": amount_min.get(),
        "amount_max": amount_max.get(),
        "radius": radius_entry.get(),
        "radius_min": radius_min.get(),
        "radius_max": radius_max.get(),
        "edge_min": edge_min.get(),
        "edge_max": edge_max.get(),
        "saliency_min": saliency_min.get(),
        "saliency_max": saliency_max.get(),
        "nlm_h": nlm_h_entry.get(),
        "nlm_template_size": nlm_template_size_entry.get(),
        "nlm_search_size": nlm_search_size_entry.get(),
        "blur_strength": blur_strength.get(),
        "sharp_strength": sharp_strength.get(),
        "color": color_entry.get(),
        "tolerance": tolerance_entry.get(),
        "brightness": brightness_entry.get(),
        "contrast": contrast_entry.get(),
        "saturation": saturation_entry.get(),
        "hue": hue_entry.get(),
        "use_cuda_gpu": gpu_checkbox_var.get() # Uložit stav GPU
    }
    utils.save_settings_to_file(settings)

def load_settings_gui_handler():
    """Načte nastavení do GUI z JSON souboru."""
    settings = utils.load_settings_from_file()
    if settings:
        mode_var.set(settings.get("mode", "unsharp"))
        denoise_algo_var.set(settings.get("denoise_algo", config.DENOISE_ALGO_GAUSSIAN))
        amount_min.delete(0, tk.END)
        amount_min.insert(0, settings.get("amount_min", "1.0"))
        amount_max.delete(0, tk.END)
        amount_max.insert(0, settings.get("amount_max", "1.5"))
        radius_entry.delete(0, tk.END)
        radius_entry.insert(0, settings.get("radius", "5"))
        radius_min.delete(0, tk.END)
        radius_min.insert(0, settings.get("radius_min", "20"))
        radius_max.delete(0, tk.END)
        radius_max.insert(0, settings.get("radius_max", "40"))
        edge_min.delete(0, tk.END)
        edge_min.insert(0, settings.get("edge_min", "80"))
        edge_max.delete(0, tk.END)
        edge_max.insert(0, settings.get("edge_max", "120"))
        saliency_min.delete(0, tk.END)
        saliency_min.insert(0, settings.get("saliency_min", "0.5"))
        saliency_max.delete(0, tk.END)
        saliency_max.insert(0, settings.get("saliency_max", "0.7"))
        nlm_h_entry.delete(0, tk.END)
        nlm_h_entry.insert(0, settings.get("nlm_h", "10"))
        nlm_template_size_entry.delete(0, tk.END)
        nlm_template_size_entry.insert(0, settings.get("nlm_template_size", "7"))
        nlm_search_size_entry.delete(0, tk.END)
        nlm_search_size_entry.insert(0, settings.get("nlm_search_size", "21"))
        blur_strength.delete(0, tk.END)
        blur_strength.insert(0, settings.get("blur_strength", "10"))
        sharp_strength.delete(0, tk.END)
        sharp_strength.insert(0, settings.get("sharp_strength", "1.5"))
        color_entry.delete(0, tk.END)
        color_entry.insert(0, settings.get("color", "0,0,255"))
        tolerance_entry.delete(0, tk.END)
        tolerance_entry.insert(0, settings.get("tolerance", "30"))
        brightness_entry.delete(0, tk.END)
        brightness_entry.insert(0, settings.get("brightness", "0"))
        contrast_entry.delete(0, tk.END)
        contrast_entry.insert(0, settings.get("contrast", "1.0"))
        saturation_entry.delete(0, tk.END)
        saturation_entry.insert(0, settings.get("saturation", "1.0"))
        hue_entry.delete(0, tk.END)
        hue_entry.insert(0, settings.get("hue", "0"))
        
        # Nastavit stav GPU checkboxu a aktualizovat jeho stav
        gpu_checkbox_var.set(settings.get("use_cuda_gpu", False))
        update_gpu_checkbox_state() # Zajistit správnou aktualizaci po načtení

        update_param_info()

def choose_color():
    """Otevře dialog pro výběr barvy a nastaví hodnotu BGR do pole."""
    color_code = colorchooser.askcolor(title="Vyberte barvu")
    if color_code[0]: 
        r, g, b = [int(c) for c in color_code[0]]
        color_entry.delete(0, tk.END)
        color_entry.insert(0, f"{b},{g},{r}") 

def update_param_info(*args):
    """Aktualizuje zobrazení parametrů a textu pro zvolený režim."""
    mode = mode_var.get()
    
    mode_info_text.config(state=tk.NORMAL)
    mode_info_text.delete(1.0, tk.END)

    all_entries = [
        amount_min, amount_max, radius_entry, radius_min, radius_max, 
        edge_min, edge_max, saliency_min, saliency_max, 
        blur_strength, sharp_strength, color_entry, tolerance_entry,
        nlm_h_entry, nlm_template_size_entry, nlm_search_size_entry,
        brightness_entry, contrast_entry, saturation_entry, hue_entry
    ]
    for entry in all_entries:
        entry.delete(0, tk.END)
        entry.grid_remove() 
    
    color_picker_button.grid_remove() 
    denoise_algo_label.grid_remove()
    denoise_algo_dropdown.grid_remove()


    # Zobrazení společných parametrů (Amount Range)
    ttk.Label(params_frame, text="Amount (Min):", style='TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=5)
    amount_min.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    ttk.Label(params_frame, text="Amount (Max):", style='TLabel').grid(row=0, column=2, sticky="w", padx=5, pady=5)
    amount_max.grid(row=0, column=3, sticky="w", padx=5, pady=5)
    amount_min.insert(0, "1.0")
    amount_max.insert(0, "1.5")


    if mode == "unsharp":
        mode_info_text.insert(tk.END, "Unsharp Mask: Standardní ostření s pevným poloměrem. Zvyšuje ostrost hran a detailů.")
        ttk.Label(params_frame, text="Radius:", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        radius_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        radius_entry.insert(0, "5")

    elif mode == "denoise":
        mode_info_text.insert(tk.END, "Denoise: Odstranění šumu z videa. Vyberte typ algoritmu.")
        
        denoise_algo_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        denoise_algo_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        def update_denoise_params_display(*args):
            # Skrýt všechny denoise parametry nejprve
            radius_entry.grid_remove()
            nlm_h_entry.grid_remove()
            nlm_template_size_entry.grid_remove()
            nlm_search_size_entry.grid_remove()
            
            # Zobrazit parametry pro zvolený algoritmus
            if denoise_algo_var.get() == config.DENOISE_ALGO_GAUSSIAN:
                ttk.Label(params_frame, text="Strength (Radius):", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
                radius_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
                radius_entry.insert(0, "10")
            elif denoise_algo_var.get() == config.DENOISE_ALGO_NL_MEANS:
                ttk.Label(params_frame, text="NLM h (Strength):", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
                nlm_h_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
                nlm_h_entry.insert(0, "10")

                ttk.Label(params_frame, text="NLM Template Size:", style='TLabel').grid(row=3, column=0, sticky="w", padx=5, pady=5)
                nlm_template_size_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
                nlm_template_size_entry.insert(0, "7")

                ttk.Label(params_frame, text="NLM Search Size:", style='TLabel').grid(row=4, column=0, sticky="w", padx=5, pady=5)
                nlm_search_size_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
                nlm_search_size_entry.insert(0, "21")

        denoise_algo_var.trace_add('write', update_denoise_params_display)
        update_denoise_params_display() 

    elif mode == "focus_mask":
        mode_info_text.insert(tk.END, "Focus Mask: Aplikuje ostřící masku na celý snímek s poloměrem, který se může měnit v průběhu videa.")
        ttk.Label(params_frame, text="Radius (Min):", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        radius_min.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(params_frame, text="Radius (Max):", style='TLabel').grid(row=1, column=2, sticky="w", padx=5, pady=5)
        radius_max.grid(row=1, column=3, sticky="w", padx=5, pady=5)
        radius_min.insert(0, "5")
        radius_max.insert(0, "10")

    elif mode == "smart_focus":
        mode_info_text.insert(tk.END, "Smart Focus: Inteligentní ostření založené na detekci hran a důležitých (salientních) oblastech. Parametry se mohou měnit v průběhu videa.")
        ttk.Label(params_frame, text="Radius (Min):", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        radius_min.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(params_frame, text="Radius (Max):", style='TLabel').grid(row=1, column=2, sticky="w", padx=5, pady=5)
        radius_max.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(params_frame, text="Edge Threshold (Min):", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
        edge_min.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(params_frame, text="Edge Threshold (Max):", style='TLabel').grid(row=2, column=2, sticky="w", padx=5, pady=5)
        edge_max.grid(row=2, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(params_frame, text="Saliency Threshold (Min):", style='TLabel').grid(row=3, column=0, sticky="w", padx=5, pady=5)
        saliency_min.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(params_frame, text="Saliency Threshold (Max):", style='TLabel').grid(row=3, column=2, sticky="w", padx=5, pady=5)
        saliency_max.grid(row=3, column=3, sticky="w", padx=5, pady=5)
        
        radius_min.insert(0, "5")
        radius_max.insert(0, "10")
        edge_min.insert(0, "50")
        edge_max.insert(0, "150")
        saliency_min.insert(0, "0.3")
        saliency_max.insert(0, "0.8")

    elif mode == "gradient_mask":
        mode_info_text.insert(tk.END, "Gradient Mask: Vytváří vizuální přechod mezi rozmazanými a ostřenými oblastmi, např. od shora dolů.")
        ttk.Label(params_frame, text="Blur Strength:", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        blur_strength.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(params_frame, text="Sharp Strength:", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
        sharp_strength.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        blur_strength.insert(0, "10")
        sharp_strength.insert(0, "1.5")

    elif mode == "stabilize":
        mode_info_text.insert(tk.END, "Stabilize Video: Snižuje roztřesenost kamery ve videu. Žádné uživatelské parametry nejsou vyžadovány.")
        amount_min.grid_remove() # Skrýt, protože se nepoužívá
        amount_max.grid_remove()

    elif mode == "adaptive_focus":
        mode_info_text.insert(tk.END, "Adaptive Focus: Automaticky analyzuje každý snímek a aplikuje vhodné zaostření. Nejlépe se používá po spuštění analýzy videa.")
        amount_min.grid_remove() 
        amount_max.grid_remove()

    elif mode == "color_selective_focus":
        mode_info_text.insert(tk.END, "Color Selective Focus: Zaostřuje pouze oblasti odpovídající určitému barevnému rozsahu. Je třeba vybrat cílovou barvu a toleranci.")
        ttk.Label(params_frame, text="Target Color (B,G,R):", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        color_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        color_picker_button.grid(row=1, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(params_frame, text="Tolerance (0-255):", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tolerance_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(params_frame, text="Amount:", style='TLabel').grid(row=3, column=0, sticky="w", padx=5, pady=5)
        amount_min.grid(row=3, column=1, sticky="w", padx=5, pady=5) 
        ttk.Label(params_frame, text="Radius:", style='TLabel').grid(row=4, column=0, sticky="w", padx=5, pady=5)
        radius_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        color_entry.insert(0, f"{config.DEFAULT_TARGET_COLOR_BGR[0]},{config.DEFAULT_TARGET_COLOR_BGR[1]},{config.DEFAULT_TARGET_COLOR_BGR[2]}")
        tolerance_entry.insert(0, "30")
        amount_min.insert(0, "1.5") 
        radius_entry.insert(0, "5")

    elif mode == "color_correction":
        mode_info_text.insert(tk.END, "Color Correction: Základní úpravy jasu, kontrastu, saturace a odstínu videa.")
        ttk.Label(params_frame, text="Brightness (-100 to 100):", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
        brightness_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        brightness_entry.insert(0, str(config.COLOR_CORRECTION_BRIGHTNESS_DEFAULT))

        ttk.Label(params_frame, text="Contrast (0.0 to 3.0):", style='TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=5)
        contrast_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        contrast_entry.insert(0, str(config.COLOR_CORRECTION_CONTRAST_DEFAULT))

        ttk.Label(params_frame, text="Saturation (0.0 to 2.0):", style='TLabel').grid(row=3, column=0, sticky="w", padx=5, pady=5)
        saturation_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        saturation_entry.insert(0, str(config.COLOR_CORRECTION_SATURATION_DEFAULT))

        ttk.Label(params_frame, text="Hue (-180 to 180):", style='TLabel').grid(row=4, column=0, sticky="w", padx=5, pady=5)
        hue_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        hue_entry.insert(0, str(config.COLOR_CORRECTION_HUE_DEFAULT))

    mode_info_text.config(state=tk.DISABLED) 

def update_gpu_checkbox_state():
    """Aktualizuje stav GPU checkboxu na základě dostupnosti CUDA."""
    if config.CUDA_AVAILABLE:
        gpu_checkbox.config(state=tk.NORMAL)
    else:
        gpu_checkbox.config(state=tk.DISABLED)
        gpu_checkbox_var.set(False) # Zajistit, že je odškrtnuto, pokud není k dispozici

def gui_interface():
    """Vytvoří grafické uživatelské rozhraní (GUI) pro zpracování videa."""
    global input_file, output_file, mode_var, denoise_algo_var, amount_min, amount_max, radius_entry
    global radius_min, radius_max, edge_min, edge_max, saliency_min, saliency_max
    global blur_strength, sharp_strength, color_entry, tolerance_entry
    global brightness_entry, contrast_entry, saturation_entry, hue_entry 
    global progress_var, progress_bar, cancel_button, process_button, mode_info_text
    global params_frame, color_picker_button, roi_checkbox_var
    global cpu_label, mem_label, gpu_label 
    global nlm_h_entry, nlm_template_size_entry, nlm_search_size_entry 
    global denoise_algo_label, denoise_algo_dropdown 
    global gpu_checkbox_var, gpu_checkbox # NOVÉ: GPU checkbox

    root = tk.Tk()
    config.gui_root = root 
    root.title("Nástroj pro zpracování videa")
    root.geometry("1000x1200") 
    
    style = ttk.Style()
    style.theme_use('clam')
    
    root.configure(bg='#000000')
    style.configure('.', background='#000000', foreground='#ffffff')
    style.configure('TFrame', background='#000000')
    style.configure('TLabel', background='#000000', foreground='#ffffff', font=('Segoe UI', 10))
    style.configure('TLabelframe', background='#000000', bordercolor='#808080', 
                   lightcolor='#808080', darkcolor='#808080', relief=tk.GROOVE)
    style.configure('TLabelframe.Label', background='#000000', foreground='#ffffff')

    style.configure('blue.Horizontal.TProgressbar', troughcolor='#000000', 
                   background='#1E90FF', thickness=15)
    style.configure('teal.Horizontal.TProgressbar', troughcolor='#000000',
                   background='#40E0D0', thickness=15)
    style.configure('purple.Horizontal.TProgressbar', troughcolor='#000000',
                   background='#9400D3', thickness=15)

    style.configure('TButton', background='#2d2d2d', foreground='#ADD8E6',
                   bordercolor='#808080', relief=tk.RAISED, padding=8,
                   font=('Segoe UI', 10, 'bold'))
    style.map('TButton',
             background=[('active', '#1E90FF')],
             foreground=[('active', '#ffffff')])
    
    style.configure('TCheckbutton', background='#000000', foreground='#ffffff')
    style.map('TCheckbutton',
              background=[('active', '#000000')],
              foreground=[('active', '#ffffff')])

    style.configure('TEntry', fieldbackground='#1a1a1a', foreground='#ffffff',
                   insertcolor='#ffffff', bordercolor='#808080')
    style.configure('TCombobox', fieldbackground='#1a1a1a', foreground='#ffffff',
                   selectbackground='#1E90FF', selectforeground='#ffffff')
    style.map('TCombobox', fieldbackground=[('readonly', '#1a1a1a')]) 

    # --- Hlavní části rozložení ---
    top_panel = ttk.Frame(root, style='TFrame')
    top_panel.pack(fill=tk.X, padx=10, pady=(10,5))
    
    main_panel = ttk.Frame(root, style='TFrame')
    main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    nav_panel = ttk.Frame(main_panel, style='TFrame')
    nav_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
    
    workspace = ttk.LabelFrame(main_panel, text="Video Processing", style='TLabelframe')
    workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    action_panel = ttk.Frame(main_panel, style='TFrame')
    action_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
    
    status_panel = ttk.Frame(root, style='TFrame')
    status_panel.pack(fill=tk.X, padx=10, pady=(5,10))

    # --- Horní panel s metrikami ---
    metrics_frame = ttk.LabelFrame(top_panel, text="System Metrics", style='TLabelframe')
    metrics_frame.pack(fill=tk.X, padx=5, pady=5)
    
    cpu_label = ttk.Label(metrics_frame, text="CPU: N/A", foreground="#32CD32")
    cpu_label.grid(row=0, column=0, padx=5, sticky="w")
    
    mem_label = ttk.Label(metrics_frame, text="Memory: N/A", foreground="#FFFF00")
    mem_label.grid(row=0, column=1, padx=5, sticky="w")
    
    gpu_label = ttk.Label(metrics_frame, text="GPU: N/A", foreground="#FF0000")
    gpu_label.grid(row=0, column=2, padx=5, sticky="w")

    update_system_metrics_gui()

    # --- Navigační panel (Výběr režimu) ---
    ttk.Label(nav_panel, text="Funkce", style='TLabel').pack(pady=5)
    
    nav_buttons_data = [
        ("Unsharp Mask", "unsharp"),
        ("Focus Mask", "focus_mask"), 
        ("Smart Focus", "smart_focus"),
        ("Denoise", "denoise"),
        ("Gradient Mask", "gradient_mask"),
        ("Stabilize Video", "stabilize"),
        ("Adaptive Focus", "adaptive_focus"),
        ("Color Selective Focus", "color_selective_focus"),
        ("Color Correction", "color_correction") 
    ]
    
    mode_var = tk.StringVar(value="unsharp")
    for text, value in nav_buttons_data:
        rb = ttk.Radiobutton(nav_panel, text=text, 
                            variable=mode_var, value=value,
                            style='TCheckbutton') 
        rb.pack(fill=tk.X, pady=2)
    mode_var.trace_add('write', update_param_info) 

    # --- Panel akčních tlačítek ---
    ttk.Label(action_panel, text="Akce", style='TLabel').pack(pady=5)
    
    action_buttons_data = [
        ("Process Video", process_video_gui_handler),
        ("Live Preview", live_preview_gui_handler), 
        ("Analyze Video", analyze_and_suggest_gui_handler), 
        ("Cancel Processing", utils.cancel_processing),
        ("Save Settings", save_settings_gui_handler),
        ("Load Settings", load_settings_gui_handler),
        ("Help", utils.show_help_dialog)
    ]
    
    for text, command in action_buttons_data:
        btn = ttk.Button(action_panel, text=text, 
                        style='TButton',
                        command=command)
        btn.pack(fill=tk.X, pady=2)
        if text == "Cancel Processing":
            cancel_button = btn 
        elif text == "Process Video":
            process_button = btn 

    # NOVÉ: GPU Acceleration Checkbox
    gpu_checkbox_var = tk.BooleanVar(value=False)
    gpu_checkbox = ttk.Checkbutton(action_panel, text="Povolit GPU (CUDA)", variable=gpu_checkbox_var, style='TCheckbutton')
    gpu_checkbox.pack(fill=tk.X, pady=5)
    gpu_checkbox.bind("<Button-1>", lambda event: config.gui_root.after_idle(update_gpu_checkbox_state)) # Ensures state updates AFTER click

    # --- Pracovní plocha ---
    # Výběr souborů
    file_frame = ttk.LabelFrame(workspace, text="Soubory", style='TLabelframe')
    file_frame.pack(fill=tk.X, padx=5, pady=5)

    input_file = tk.StringVar()
    output_file = tk.StringVar()
    
    ttk.Label(file_frame, text="Vstupní soubor:", style='TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(file_frame, textvariable=input_file, style='TEntry', width=50).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(file_frame, text="Procházet", command=select_input_file, style='TButton').grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(file_frame, text="Výstupní soubor:", style='TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(file_frame, textvariable=output_file, style='TEntry', width=50).grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(file_frame, text="Procházet", command=select_output_file, style='TButton').grid(row=1, column=2, padx=5, pady=5)

    # Sekce ROI
    roi_frame = ttk.LabelFrame(workspace, text="Oblast zájmu (ROI)", style='TLabelframe')
    roi_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(roi_frame, text="Vybrané ROI:", style='TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=5)
    roi_info_label = ttk.Label(roi_frame, text="Žádné", style='TLabel') 
    roi_info_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    
    def update_roi_label_periodic():
        if config.selected_roi:
            roi_info_label.config(text=f"({config.selected_roi[0]}, {config.selected_roi[1]}, {config.selected_roi[2]}x{config.selected_roi[3]})")
        else:
            roi_info_label.config(text="Žádné")
        config.gui_root.after(100, update_roi_label_periodic) 

    update_roi_label_periodic() 

    ttk.Button(roi_frame, text="Vymazat ROI", command=clear_roi, style='TButton').grid(row=0, column=2, padx=5, pady=5)
    
    roi_checkbox_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(roi_frame, text="Aplikovat ROI v živém náhledu", variable=roi_checkbox_var, style='TCheckbutton').grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)


    # Informační textový rámeček pro popis režimu
    info_frame = ttk.LabelFrame(workspace, text="Informace o režimu", style='TLabelframe')
    info_frame.pack(fill=tk.X, padx=5, pady=5)
    
    mode_info_text = tk.Text(info_frame, height=4, wrap=tk.WORD, font=('Helvetica', 8), bg='#1a1a1a', fg='#ffffff')
    mode_info_text.pack(fill=tk.X, padx=5, pady=5)
    mode_info_text.insert(tk.END, "Vyberte režim zpracování pro zobrazení jeho popisu a požadovaných parametrů.")
    mode_info_text.config(state=tk.DISABLED)

    # Rámeček parametrů
    params_frame = ttk.LabelFrame(workspace, text="Parametry", style='TLabelframe')
    params_frame.pack(fill=tk.X, padx=5, pady=5)

    # Vstupní widgety parametrů (inicializovány globálně pro aktualizaci funkcí update_param_info)
    amount_min = ttk.Entry(params_frame, width=8, style='TEntry')
    amount_max = ttk.Entry(params_frame, width=8, style='TEntry')
    radius_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    radius_min = ttk.Entry(params_frame, width=8, style='TEntry')
    radius_max = ttk.Entry(params_frame, width=8, style='TEntry')
    edge_min = ttk.Entry(params_frame, width=8, style='TEntry')
    edge_max = ttk.Entry(params_frame, width=8, style='TEntry')
    saliency_min = ttk.Entry(params_frame, width=8, style='TEntry')
    saliency_max = ttk.Entry(params_frame, width=8, style='TEntry')
    blur_strength = ttk.Entry(params_frame, width=8, style='TEntry')
    sharp_strength = ttk.Entry(params_frame, width=8, style='TEntry')
    color_entry = ttk.Entry(params_frame, width=15, style='TEntry')
    tolerance_entry = ttk.Entry(params_frame, width=8, style='TEntry')

    denoise_algo_var = tk.StringVar(value=config.DENOISE_ALGO_GAUSSIAN)
    denoise_algo_label = ttk.Label(params_frame, text="Denoise Algorithm:", style='TLabel')
    denoise_algo_dropdown = ttk.OptionMenu(params_frame, denoise_algo_var, config.DENOISE_ALGO_GAUSSIAN, config.DENOISE_ALGO_GAUSSIAN, config.DENOISE_ALGO_NL_MEANS)
    
    nlm_h_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    nlm_template_size_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    nlm_search_size_entry = ttk.Entry(params_frame, width=8, style='TEntry')

    brightness_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    contrast_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    saturation_entry = ttk.Entry(params_frame, width=8, style='TEntry')
    hue_entry = ttk.Entry(params_frame, width=8, style='TEntry')

    color_picker_button = ttk.Button(params_frame, text="Vybrat barvu", command=choose_color, style='TButton')

    # Počáteční kontrola dostupnosti CUDA a aktualizace GUI
    utils.check_cuda_availability()
    update_gpu_checkbox_state()

    update_param_info()

    # --- Progress bar ---
    progress_frame = ttk.Frame(workspace, style='TFrame')
    progress_frame.pack(fill=tk.X, padx=5, pady=5)
    
    progress_var = tk.IntVar(value=0)
    ttk.Label(progress_frame, text="Průběh:", style='TLabel').pack(side=tk.LEFT, padx=5)
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100, length=400, style='blue.Horizontal.TProgressbar')
    progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # --- Dolní stavový panel s barevnými progress bary (pouze vizuální) ---
    progress_blue = ttk.Progressbar(status_panel, style='blue.Horizontal.TProgressbar', variable=progress_var)
    progress_teal = ttk.Progressbar(status_panel, style='teal.Horizontal.TProgressbar', variable=progress_var)
    progress_purple = ttk.Progressbar(status_panel, style='purple.Horizontal.TProgressbar', variable=progress_var)
    progress_blue.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    progress_teal.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    progress_purple.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    # --- Klávesové zkratky ---
    root.bind("<F1>", lambda event: utils.show_help_dialog())
    root.bind("<Escape>", lambda event: utils.cancel_processing())

    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())
    
    root.mainloop()
