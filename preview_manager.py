import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import time
import threading
import os
import tkinter as tk # Jen pro messagebox, ne celý GUI

import config # Import config pro sdílené proměnné
import utils # Import utils pro aktualizace průběhu, check_video_file atd.
import video_processor # Import funkcí procesoru
import video_analyzer # Pro adaptive_focus k získání navržených parametrů, pokud je třeba

# Globální proměnné pro výběr ROI v náhledu
drawing = False
ix, iy = -1, -1
current_roi_selection_drawing = None # (x, y, w, h) pro aktuální kreslení
preview_clip_duration = 0 # Pro dynamické parametry v preview

# Global reference to trackbar positions to avoid recreating them on each frame
_trackbar_values = {}


def _mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_roi_selection_drawing
    
    # Param obsahuje (frame_w, frame_h) - původní rozlišení snímku
    (original_frame_w, original_frame_h) = param

    # Měřítko souřadnic myši zpět na původní velikost snímku
    # Předpokládáme, že náhled je škálován proporčně k PREVIEW_MAX_WIDTH
    current_preview_width = config.PREVIEW_MAX_WIDTH
    current_preview_height = int(original_frame_h * (config.PREVIEW_MAX_WIDTH / original_frame_w)) if original_frame_w > 0 else original_frame_h

    scale_x = original_frame_w / current_preview_width
    scale_y = original_frame_h / current_preview_height

    scaled_x = int(x * scale_x)
    scaled_y = int(y * scale_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = scaled_x, scaled_y
        current_roi_selection_drawing = None # Reset při začátku nového kreslení

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Uložit škálované ROI pro kreslení
            current_roi_selection_drawing = (min(ix, scaled_x), min(iy, scaled_y),
                                             abs(ix - scaled_x), abs(iy - scaled_y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        final_x1, final_y1 = ix, iy
        final_x2, final_y2 = scaled_x, scaled_y
        
        # Zajistit kladnou šířku/výšku
        x_coord = min(final_x1, final_x2)
        y_coord = min(final_y1, final_y2)
        width = abs(final_x1 - final_x2)
        height = abs(final_y1 - final_y2)

        config.selected_roi = (x_coord, y_coord, width, height)
        current_roi_selection_drawing = None # Vyčistit kreslící ROI, jakmile je potvrzeno
        print(f"ROI vybráno: {config.selected_roi}") # Ladění

# Trackbar callback (empty, as we poll values directly in the loop)
def _trackbar_callback(val, name):
    _trackbar_values[name] = val

def live_preview_video(input_path, mode, initial_gui_params, use_cuda_for_preview):
    """
    Zobrazí živý náhled zpracovaného videa s možností výběru ROI a interaktivními slidery.
    `initial_gui_params` jsou parametry z GUI pro inicializaci sliderů.
    `use_cuda_for_preview` indikuje, zda se má pokusit použít CUDA pro náhled.
    """
    global drawing, ix, iy, current_roi_selection_drawing, preview_clip_duration, _trackbar_values

    is_valid, msg = utils.check_video_file(input_path)
    if not is_valid:
        tk.messagebox.showerror("Chyba", f"Vstupní soubor není platný video formát: {msg}")
        return

    clip = None
    try:
        config.stop_preview_event.clear() # Vyčistit stop event pro nový náhled
        drawing = False
        ix, iy = -1, -1
        current_roi_selection_drawing = None
        _trackbar_values.clear() # Clear trackbar values for new preview

        clip = VideoFileClip(input_path)
        preview_clip_duration = clip.duration # Uložit délku klipu pro dynamické parametry
        
        # Získat první snímek pro určení rozměrů
        first_frame = next(clip.iter_frames(fps=clip.fps, dtype="uint8"))
        original_height, original_width = first_frame.shape[:2]

        # Určit rozměry pro změnu velikosti pro náhled
        if original_width > config.PREVIEW_MAX_WIDTH:
            scale_factor = config.PREVIEW_MAX_WIDTH / original_width
            preview_width = config.PREVIEW_MAX_WIDTH
            preview_height = int(original_height * scale_factor)
        else:
            preview_width, preview_height = original_width, original_height
        
        # Nastavit okna OpenCV a mouse callback
        cv2.namedWindow(config.PREVIEW_ORIGINAL_WINDOW_NAME)
        cv2.namedWindow(config.PREVIEW_PROCESSED_WINDOW_NAME)
        cv2.setMouseCallback(config.PREVIEW_PROCESSED_WINDOW_NAME, _mouse_callback, (original_width, original_height))

        # Upravit FPS pro výkon náhledu (např. 10 FPS nebo aktuální, pokud je nižší)
        preview_fps = min(clip.fps, 10) 

        # --- Nastavení Trackbarů na základě režimu ---
        # Store default/initial values in _trackbar_values and then create trackbars
        
        if mode == "unsharp":
            _trackbar_values["Amount"] = int(initial_gui_params["amount_range"][0] * 100)
            _trackbar_values["Radius"] = initial_gui_params["radius"]
            cv2.createTrackbar("Amount", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Amount"], 300, lambda val: _trackbar_callback(val, "Amount")) # Max 3.0
            cv2.createTrackbar("Radius", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Radius"], 20, lambda val: _trackbar_callback(val, "Radius"))
        elif mode == "denoise":
            denoise_type = initial_gui_params["denoise_type"]
            _trackbar_values["denoise_type"] = denoise_type # Store for internal logic
            if denoise_type == config.DENOISE_ALGO_GAUSSIAN:
                _trackbar_values["Strength"] = initial_gui_params["strength"]
                cv2.createTrackbar("Strength", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Strength"], 30, lambda val: _trackbar_callback(val, "Strength"))
            elif denoise_type == config.DENOISE_ALGO_NL_MEANS:
                _trackbar_values["h"] = initial_gui_params["nlm_h"]
                _trackbar_values["TemplateSize"] = initial_gui_params["nlm_template_size"]
                _trackbar_values["SearchSize"] = initial_gui_params["nlm_search_size"]
                cv2.createTrackbar("h", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["h"], 50, lambda val: _trackbar_callback(val, "h")) # h param
                cv2.createTrackbar("TemplateSize", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["TemplateSize"], 21, lambda val: _trackbar_callback(val, "TemplateSize")) # Odd numbers only, max 21
                cv2.setTrackbarMin("TemplateSize", config.PREVIEW_PROCESSED_WINDOW_NAME, 3) # Min template size
                cv2.createTrackbar("SearchSize", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["SearchSize"], 50, lambda val: _trackbar_callback(val, "SearchSize")) # Odd numbers only, max 50
                cv2.setTrackbarMin("SearchSize", config.PREVIEW_PROCESSED_WINDOW_NAME, 7) # Min search size
        elif mode == "focus_mask":
            _trackbar_values["Amount (Min)"] = int(initial_gui_params["amount_range"][0] * 100)
            _trackbar_values["Radius (Min)"] = initial_gui_params["radius_range"][0]
            cv2.createTrackbar("Amount (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Amount (Min)"], 300, lambda val: _trackbar_callback(val, "Amount (Min)"))
            cv2.createTrackbar("Radius (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Radius (Min)"], 50, lambda val: _trackbar_callback(val, "Radius (Min)"))
        elif mode == "smart_focus":
            _trackbar_values["Amount (Min)"] = int(initial_gui_params["amount_range"][0] * 100)
            _trackbar_values["Radius (Min)"] = initial_gui_params["radius_range"][0]
            _trackbar_values["Edge Th (Min)"] = initial_gui_params["edge_threshold_range"][0]
            _trackbar_values["Saliency Th (Min)"] = int(initial_gui_params["saliency_threshold_range"][0] * 100)
            cv2.createTrackbar("Amount (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Amount (Min)"], 300, lambda val: _trackbar_callback(val, "Amount (Min)"))
            cv2.createTrackbar("Radius (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Radius (Min)"], 50, lambda val: _trackbar_callback(val, "Radius (Min)"))
            cv2.createTrackbar("Edge Th (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Edge Th (Min)"], 255, lambda val: _trackbar_callback(val, "Edge Th (Min)"))
            cv2.createTrackbar("Saliency Th (Min)", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Saliency Th (Min)"], 100, lambda val: _trackbar_callback(val, "Saliency Th (Min)")) # 0-100 for 0-1.0
        elif mode == "gradient_mask":
            _trackbar_values["Blur Strength"] = initial_gui_params["blur_strength"]
            _trackbar_values["Sharp Strength"] = int(initial_gui_params["sharp_strength"] * 100)
            cv2.createTrackbar("Blur Strength", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Blur Strength"], 50, lambda val: _trackbar_callback(val, "Blur Strength"))
            cv2.createTrackbar("Sharp Strength", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Sharp Strength"], 300, lambda val: _trackbar_callback(val, "Sharp Strength"))
        elif mode == "color_selective_focus":
            _trackbar_values["Amount"] = int(initial_gui_params["amount"] * 100)
            _trackbar_values["Radius"] = initial_gui_params["radius"]
            _trackbar_values["Tolerance"] = initial_gui_params["tolerance"]
            cv2.createTrackbar("Amount", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Amount"], 300, lambda val: _trackbar_callback(val, "Amount"))
            cv2.createTrackbar("Radius", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Radius"], 20, lambda val: _trackbar_callback(val, "Radius"))
            cv2.createTrackbar("Tolerance", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Tolerance"], 255, lambda val: _trackbar_callback(val, "Tolerance"))
            # Dummy trackbars for color reference, cannot change color interactively via trackbar
            cv2.createTrackbar(f"Color B:{initial_gui_params['target_color'][0]}", config.PREVIEW_PROCESSED_WINDOW_NAME, 0, 1, lambda val: None)
            cv2.createTrackbar(f"Color G:{initial_gui_params['target_color'][1]}", config.PREVIEW_PROCESSED_WINDOW_NAME, 0, 1, lambda val: None)
            cv2.createTrackbar(f"Color R:{initial_gui_params['target_color'][2]}", config.PREVIEW_PROCESSED_WINDOW_NAME, 0, 1, lambda val: None)
        elif mode == "color_correction":
            _trackbar_values["Brightness"] = initial_gui_params["brightness"] + 100
            _trackbar_values["Contrast"] = int(initial_gui_params["contrast"] * 100)
            _trackbar_values["Saturation"] = int(initial_gui_params["saturation"] * 100)
            _trackbar_values["Hue"] = initial_gui_params["hue"] + 180
            cv2.createTrackbar("Brightness", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Brightness"], 200, lambda val: _trackbar_callback(val, "Brightness")) # Map -100 to 100 to 0 to 200
            cv2.createTrackbar("Contrast", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Contrast"], 300, lambda val: _trackbar_callback(val, "Contrast")) # Map 0 to 3.0 to 0 to 300
            cv2.createTrackbar("Saturation", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Saturation"], 200, lambda val: _trackbar_callback(val, "Saturation")) # Map 0 to 2.0 to 0 to 200
            cv2.createTrackbar("Hue", config.PREVIEW_PROCESSED_WINDOW_NAME, _trackbar_values["Hue"], 360, lambda val: _trackbar_callback(val, "Hue")) # Map -180 to 180 to 0 to 360

        # adaptive_focus and stabilize modes don't have interactive parameters via trackbars
        # Stabilize is too complex for real-time per-frame changes.
        # Adaptive focus adapts automatically, no direct user sliders needed.

        for t_frame in clip.iter_frames(fps=preview_fps, dtype="uint8"):
            if config.stop_preview_event.is_set():
                break # Ukončit smyčku, pokud je přijat signál zastavení

            current_t = clip.reader.pos / clip.fps # Získat aktuální čas v klipu pro dynamické parametry

            processed_frame = t_frame.copy()
            active_roi_for_effect = config.selected_roi if config.apply_roi_in_preview else None

            # --- Získat hodnoty z Trackbarů a aplikovat efekt ---
            try:
                if mode == "unsharp":
                    amount = _trackbar_values.get("Amount", 100) / 100.0
                    radius = _trackbar_values.get("Radius", 5)
                    if radius == 0: radius = 1 # Vyhnout se radius 0
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.unsharp_mask_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview, amount, radius
                    )
                elif mode == "denoise":
                    denoise_type = _trackbar_values.get("denoise_type", config.DENOISE_ALGO_GAUSSIAN)
                    if denoise_type == config.DENOISE_ALGO_GAUSSIAN:
                        strength = _trackbar_values.get("Strength", 10)
                        if strength == 0: strength = 1
                        processed_frame = video_processor._apply_effect_to_frame_with_roi(
                            t_frame, video_processor.denoise_gaussian_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview, strength
                        )
                    elif denoise_type == config.DENOISE_ALGO_NL_MEANS:
                        h = _trackbar_values.get("h", 10)
                        template_size = _trackbar_values.get("TemplateSize", 7)
                        search_size = _trackbar_values.get("SearchSize", 21)
                        if template_size % 2 == 0: template_size += 1
                        if search_size % 2 == 0: search_size += 1
                        if template_size < 3: template_size = 3 
                        if search_size < 7: search_size = 7 
                        processed_frame = video_processor._apply_effect_to_frame_with_roi(
                            t_frame, video_processor.denoise_fast_nl_means_effect, active_roi_for_effect, preview_clip_duration, current_t, False, h, template_size, search_size
                        )
                elif mode == "focus_mask":
                    amount_min_val = _trackbar_values.get("Amount (Min)", 100) / 100.0
                    radius_min_val = _trackbar_values.get("Radius (Min)", 5)
                    if radius_min_val == 0: radius_min_val = 1
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.focus_mask_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview, amount_min_val, radius_min_val
                    )
                elif mode == "smart_focus":
                    amount_min_val = _trackbar_values.get("Amount (Min)", 100) / 100.0
                    radius_min_val = _trackbar_values.get("Radius (Min)", 5)
                    edge_th_min_val = _trackbar_values.get("Edge Th (Min)", 80)
                    saliency_th_min_val = _trackbar_values.get("Saliency Th (Min)", 50) / 100.0
                    if radius_min_val == 0: radius_min_val = 1
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.smart_focus_effect, active_roi_for_effect, preview_clip_duration, current_t, False,
                        amount_min_val, radius_min_val, edge_th_min_val, saliency_th_min_val
                    )
                elif mode == "gradient_mask":
                    blur_s = _trackbar_values.get("Blur Strength", 10)
                    sharp_s = _trackbar_values.get("Sharp Strength", 150) / 100.0
                    if blur_s == 0: blur_s = 1
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.gradient_mask_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview, blur_s, sharp_s
                    )
                elif mode == "stabilize":
                    processed_frame = t_frame # Real-time stabilization preview is too complex
                elif mode == "adaptive_focus":
                    suggested_params_for_adaptive = initial_gui_params.get('suggested_params', {})
                    if not suggested_params_for_adaptive:
                        dummy_analysis_results = { 'blurriness': 100, 'noise': 10, 'brightness': 100, 'contrast': 50, 'motion': 0 } 
                        suggested_params_for_adaptive = video_analyzer.adaptive_parameter_suggestion(dummy_analysis_results)

                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.adaptive_focus_effect, active_roi_for_effect, preview_clip_duration, current_t, False,
                        suggested_params_for_adaptive['amount_range'], suggested_params_for_adaptive['radius_range']
                    )
                elif mode == "color_selective_focus":
                    amount = _trackbar_values.get("Amount", 150) / 100.0
                    radius = _trackbar_values.get("Radius", 5)
                    tolerance = _trackbar_values.get("Tolerance", 30)
                    if radius == 0: radius = 1
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.color_selective_focus_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview,
                        initial_gui_params["target_color"], tolerance, amount, radius
                    )
                elif mode == "color_correction":
                    brightness = _trackbar_values.get("Brightness", 100) - 100 # Map back to -100 to 100
                    contrast = _trackbar_values.get("Contrast", 100) / 100.0
                    saturation = _trackbar_values.get("Saturation", 100) / 100.0
                    hue = _trackbar_values.get("Hue", 180) - 180 # Map back to -180 to 180
                    
                    processed_frame = video_processor._apply_effect_to_frame_with_roi(
                        t_frame, video_processor.color_correction_effect, active_roi_for_effect, preview_clip_duration, current_t, use_cuda_for_preview,
                        brightness, contrast, saturation, hue
                    )
                else:
                    processed_frame = t_frame 

            except Exception as e:
                print(f"Chyba při aplikaci efektu v náhledu: {e}")
                processed_frame = t_frame # Fallback na původní snímek
            
            # Kreslení vybraného ROI na zpracovaný snímek
            if config.selected_roi:
                x, y, w, h = config.selected_roi
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Zelený obdélník
            
            # Kreslení aktuálně kreslícího ROI na zpracovaný snímek
            if drawing and current_roi_selection_drawing:
                x, y, w, h = current_roi_selection_drawing
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Červený obdélník pro kreslení

            # Změna velikosti pro zobrazení
            resized_original = cv2.resize(t_frame, (preview_width, preview_height))
            resized_processed = cv2.resize(processed_frame, (preview_width, preview_height))
            
            cv2.imshow(config.PREVIEW_ORIGINAL_WINDOW_NAME, resized_original)
            cv2.imshow(config.PREVIEW_PROCESSED_WINDOW_NAME, resized_processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Konec
                break
            elif key == ord('s'): # Uložit aktuální snímek
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"preview_frame_{timestamp}.jpg", processed_frame)
                tk.messagebox.showinfo("Info", f"Snímek uložen jako preview_frame_{timestamp}.jpg")
            elif key == ord('r'): # Reset ROI kreslení
                config.selected_roi = None
                current_roi_selection_drawing = None
                drawing = False

        cv2.destroyAllWindows()
    except Exception as e:
        tk.messagebox.showerror("Chyba", f"Došlo k chybě během živého náhledu: {e}")
        print(f"Preview error: {e}") # Pro ladění
    finally:
        if clip:
            clip.close()
        cv2.destroyAllWindows() # Zajistit, že všechna okna jsou zavřena
        config.stop_preview_event.clear() # Vyčistit stop event po dokončení náhledu