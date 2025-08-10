import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip

import config # Import config pro sdílené proměnné a konstanty
import utils # Import utils pro aktualizace průběhu

def estimate_noise_level(gray_img):
    """Odhaduje úroveň šumu v obraze pomocí robustnějšího přístupu."""
    # Použijeme bilatelární filtr k vyhlazení, ale zachování hran
    smooth = cv2.bilateralFilter(gray_img, 9, 75, 75)
    
    # Vypočítáme rozdíl mezi originálem a vyhlazeným obrazem
    noise_map = cv2.absdiff(gray_img, smooth)
    
    # Prahování pro izolaci oblastí, kde je šum (malé hodnoty rozdílu)
    _, noise_mask = cv2.threshold(noise_map, 5, 255, cv2.THRESH_BINARY_INV)
    noise_mask = noise_mask.astype(bool) # Konvertujeme na boolean masku
    
    if np.sum(noise_mask) > 0:
        noise_estimate = np.std(gray_img[noise_mask])
    else:
        # Pokud nejsou nalezeny dostatečně homogenní oblasti, použijeme obecnější odhad
        noise_estimate = np.std(gray_img)
        
    return noise_estimate


def enhanced_video_analysis(clip_path):
    """Provádí komplexní analýzu videa včetně rozmazanosti, pohybu, obsahu a šumu."""
    metrics = {
        'blurriness': [],
        'motion': [],
        'noise': [],
        'brightness': [],
        'contrast': []
    }
    
    try:
        clip = VideoFileClip(clip_path)
    except Exception as e:
        print(f"Chyba při načítání klipu pro analýzu: {e}")
        return {k: 0 for k in metrics.keys()}

    total_frames = int(clip.fps * clip.duration)
    sample_fps = min(clip.fps, 10) # Vzorkování max 10 FPS pro rychlejší analýzu
    
    # Získat indexy snímků pro rovnoměrné vzorkování po celém videu
    frame_indices = np.linspace(0, total_frames - 1, int(clip.duration * sample_fps), dtype=int)
    if len(frame_indices) == 0: # Zpracování velmi krátkých klipů
        frame_indices = [0] if total_frames > 0 else []

    prev_gray = None
    
    for i, frame_idx in enumerate(frame_indices):
        if config.stop_processing_event.is_set():
            clip.close()
            return {k: 0 for k in metrics.keys()} # Vrátit nuly, pokud bylo zrušeno
            
        t = frame_idx / clip.fps
        try:
            frame = clip.get_frame(t)
        except Exception as e:
            print(f"Chyba při získávání snímku {frame_idx} pro analýzu: {e}")
            continue # Přeskočit problematický snímek

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analýza rozmazanosti (Variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blurriness'].append(laplacian_var)
        
        # Analýza šumu
        noise_estimate = estimate_noise_level(gray)
        metrics['noise'].append(noise_estimate)
        
        # Analýza jasu a kontrastu
        metrics['brightness'].append(np.mean(gray))
        metrics['contrast'].append(np.std(gray))
        
        # Analýza pohybu (Farneback optický tok)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            metrics['motion'].append(np.mean(motion_magnitude))
        prev_gray = gray
        
        # Aktualizace progress baru (0-30% pro analýzu)
        utils.update_progress(int((i / len(frame_indices)) * 30))

    clip.close()

    # Vypočítat průměrné hodnoty, ošetřit prázdné seznamy pro velmi krátké klipy
    result = {k: np.mean(v) if v else 0 for k, v in metrics.items()}
    return result


def adaptive_parameter_suggestion(analysis_results):
    """Doporučuje optimální parametry na základě komplexní analýzy videa."""
    blurriness = analysis_results['blurriness']
    noise = analysis_results['noise']
    brightness = analysis_results['brightness']
    contrast = analysis_results['contrast']
    
    # Základní parametry podle rozmazanosti
    if blurriness < config.BLURRINESS_LOW:
        amount_range = (0.8, 1.2)
        radius_range = (3, 5)
    elif blurriness < config.BLURRINESS_MEDIUM:
        amount_range = (1.2, 1.8)
        radius_range = (5, 10)
    else:
        amount_range = (1.8, 2.5)
        radius_range = (10, 15)
    
    # Úprava podle šumu
    if noise > config.NOISE_HIGH:  # Vysoký šum
        amount_range = (max(0.5, amount_range[0] * 0.8), amount_range[1] * 0.9)
        radius_range = (radius_range[0] + 2, radius_range[1] + 5)
    
    # Úprava podle jasu
    if brightness < config.BRIGHTNESS_LOW:  # Tmavé video
        amount_range = (amount_range[0] * 1.1, min(3.0, amount_range[1] * 1.2))
    elif brightness > config.BRIGHTNESS_HIGH:  # Velmi světlé video
        amount_range = (amount_range[0] * 0.9, amount_range[1] * 0.9)
    
    # Úprava podle kontrastu
    if contrast < config.CONTRAST_LOW:  # Nízký kontrast
        amount_range = (amount_range[0] * 1.1, min(3.0, amount_range[1] * 1.2))
    
    # Zaokrouhlení hodnot
    amount_range = (round(amount_range[0], 2), round(amount_range[1], 2))
    radius_range = (max(1, int(radius_range[0])), max(2, int(radius_range[1])))
    
    # Doporučené parametry pro edge a saliency thresholds
    edge_threshold_range = (80, 120)
    saliency_threshold_range = (0.5, 0.7)
    
    # Úprava edge threshold podle kontrastu
    if contrast < config.CONTRAST_LOW:
        edge_threshold_range = (60, 100)  # Nižší prahy pro nízký kontrast
    elif contrast > config.CONTRAST_HIGH:
        edge_threshold_range = (100, 150)  # Vyšší prahy pro vysoký kontrast
    
    # Doplnění výchozí hodnoty pro denoise_strength (pro Gaussian)
    denoise_strength = max(5, min(30, int(noise * 2))) 

    # Parametry pro Non-Local Means (orientační hodnoty založené na šumu)
    nlm_h = max(3, min(30, int(noise * 1.5))) # Filter strength
    nlm_template_size = 7 # Typically odd and small
    nlm_search_size = 21 # Typically odd and larger

    # Blur/Sharp strength pro Gradient Mask
    blur_strength = max(5, min(20, int(noise * 1.5)))
    sharp_strength = max(1.0, min(2.5, 1.0 + blurriness / 150)) # Upraveno na základě rozmazanosti

    # Zajištění, že radius_entry (pro single radius režimy) dostane rozumnou výchozí hodnotu z radius_range
    suggested_radius_entry_val = radius_range[0] if radius_range[0] > 0 else 5

    return {
        'amount_range': amount_range,
        'radius_range': radius_range,
        'edge_threshold_range': edge_threshold_range,
        'saliency_threshold_range': saliency_threshold_range,
        'denoise_strength': denoise_strength,
        'nlm_h': nlm_h,
        'nlm_template_size': nlm_template_size,
        'nlm_search_size': nlm_search_size,
        'blur_strength': blur_strength,
        'sharp_strength': round(sharp_strength, 2),
        'suggested_radius_entry_val': suggested_radius_entry_val,
        # Defaultní pro Color Correction (analýza je nenavrhuje, jen dává výchozí)
        'brightness': config.COLOR_CORRECTION_BRIGHTNESS_DEFAULT,
        'contrast': config.COLOR_CORRECTION_CONTRAST_DEFAULT,
        'saturation': config.COLOR_CORRECTION_SATURATION_DEFAULT,
        'hue': config.COLOR_CORRECTION_HUE_DEFAULT
    }
