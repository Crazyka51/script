import cv2
import numpy as np
from moviepy.editor import VideoFileClip, vfx
import math

import config # Import config pro sdílené proměnné
import utils # Import utils pro aktualizace průběhu

def _apply_effect_to_frame_with_roi(frame, effect_func_for_single_frame, roi, total_frames, current_t, use_cuda_for_effect, *args, **kwargs):
    """
    Pomocná funkce pro aplikaci efektu na jeden snímek, volitelně v rámci ROI a s podporou CUDA.
    Také se stará o aktualizaci průběhu.
    """
    if config.stop_processing_event.is_set():
        return frame # Vrátit původní snímek, pokud bylo zrušeno

    processed_frame = frame.copy()

    # Zpracování snímku buď na CPU nebo GPU
    if use_cuda_for_effect and config.CUDA_AVAILABLE:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            if roi:
                x, y, w, h = roi
                x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                if w > 0 and h > 0:
                    gpu_roi_frame = gpu_frame.colRange(x, x+w).rowRange(y, y+h)
                    processed_gpu_roi = effect_func_for_single_frame(gpu_roi_frame, *args, **kwargs)
                    processed_gpu_roi.copyTo(gpu_roi_frame) # Copy back to original GPU mat ROI
                # No 'else' here, if ROI is invalid, we proceed with the full frame operation
                # to not return an uninitialized gpu_frame in theory.
                # However, our effect_func_for_single_frame is designed to work on the ROI directly.
                # So if ROI is invalid, we just don't apply the effect on GPU_ROI, resulting in original.
                
                # If effect applied only to ROI, the rest of the frame remains unchanged on GPU.
                # So, we need to download the whole GPU frame if ROI was processed.
                processed_frame = gpu_frame.download()
            else: # No ROI, process entire frame on GPU
                processed_gpu_frame = effect_func_for_single_frame(gpu_frame, *args, **kwargs)
                processed_frame = processed_gpu_frame.download()

        except cv2.error as e:
            print(f"CUDA Error in _apply_effect_to_frame_with_roi: {e}. Falling back to CPU.")
            # Fallback to CPU processing in case of CUDA error
            if roi:
                x, y, w, h = roi
                x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                if w > 0 and h > 0:
                    roi_frame = frame[y:y+h, x:x+w]
                    processed_roi = effect_func_for_single_frame(roi_frame, *args, **kwargs)
                    processed_frame[y:y+h, x:x+w] = processed_roi
            else:
                processed_frame = effect_func_for_single_frame(frame, *args, **kwargs)
        except Exception as e:
            print(f"General Error in _apply_effect_to_frame_with_roi (CUDA path): {e}. Falling back to CPU.")
            if roi:
                x, y, w, h = roi
                x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                if w > 0 and h > 0:
                    roi_frame = frame[y:y+h, x:x+w]
                    processed_roi = effect_func_for_single_frame(roi_frame, *args, **kwargs)
                    processed_frame[y:y+h, x:x+w] = processed_roi
            else:
                processed_frame = effect_func_for_single_frame(frame, *args, **kwargs)
    else: # Use CPU processing
        if roi:
            x, y, w, h = roi
            x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            if w > 0 and h > 0:
                roi_frame = frame[y:y+h, x:x+w]
                processed_roi = effect_func_for_single_frame(roi_frame, *args, **kwargs)
                processed_frame[y:y+h, x:x+w] = processed_roi
        else:
            processed_frame = effect_func_for_single_frame(frame, *args, **kwargs)

    # Aktualizace progress baru (předpokládáme 30% pro analýzu, 70% pro zpracování)
    if total_frames > 0 and current_t is not None:
        # Přibližný index snímku pro výpočet průběhu
        # current_t * clip.fps is not directly available here in _apply_effect_to_frame_with_roi
        # We assume total_frames here refers to the total number of frames in the clip (not loop iteration)
        # This update logic belongs to the caller (make_frame_processor) for accuracy
        pass # Moved progress update to make_frame_processor

            
    return processed_frame.astype(np.uint8)

# --- Základní funkce pro zpracování videa (efekty na jeden snímek) ---
# Tyto funkce nyní přijímají buď np.ndarray nebo cv2.cuda_GpuMat
# A samy si volí CPU/GPU operace

def unsharp_mask_effect(frame_or_gpu_mat, amount, radius):
    """Aplikuje unsharp mask na jeden snímek (CPU/GPU kompatibilní)."""
    if isinstance(frame_or_gpu_mat, np.ndarray):
        blurred = cv2.GaussianBlur(frame_or_gpu_mat, (2 * radius + 1, 2 * radius + 1), radius)
        sharpened = float(amount + 1) * frame_or_gpu_mat - float(amount) * blurred
        return np.clip(sharpened, 0, 255)
    else: # Assumed cv2.cuda_GpuMat
        # CUDA Gaussian blur only works on single channel or 4-channel. We need to split if BGR
        # Or, use cv2.cuda.createGaussianFilter
        
        # Simple CUDA path for blur
        stream = cv2.cuda_Stream()
        gpu_blurred = cv2.cuda.createGaussianFilter((frame_or_gpu_mat.rows, frame_or_gpu_mat.cols), (2 * radius + 1, 2 * radius + 1), radius, radius, cv2.BORDER_DEFAULT).apply(frame_or_gpu_mat, stream)
        
        # Sharpening calculation needs to be done on CPU or carefully with CUDA ops
        # This part is more complex to do fully on GPU without element-wise ops or custom kernels.
        # For simplicity, download for sharpening and re-upload if needed, or stick to CPU for sharpening step.
        # For now, let's keep the sharpening arithmetic on CPU after GPU blur if needed.
        # This makes hybrid CPU/GPU which is common.
        
        # For now, keeping it CPU-only after blur or for full operation for simplicity.
        # A full CUDA implementation for this would involve custom kernels or splitting channels.
        # As per plan, we apply the blur on GPU and then do the arithmetic on CPU for sharpening
        # if the input was GpuMat, or it needs to be carefully vectorized with CUDA element-wise ops.
        
        # More realistic CUDA implementation for sharpening:
        # Multiply by amount+1, then subtract amount*blurred
        # This usually involves cv2.cuda.add, cv2.cuda.subtract, cv2.cuda.multiply
        
        # Simplified approach: If GPU is used for blur, download, sharpen on CPU, then return.
        # This is a common pattern for operations not fully supported by simple CUDA API.
        
        # Let's try basic arithmetic with GpuMat, may require explicit conversions or specific CUDA ops
        # It's more robust to do this on CPU if not all steps are natively in cv2.cuda.
        
        # Fallback/Hybrid: perform blur on GPU, then download and finish on CPU
        cpu_frame = frame_or_gpu_mat.download()
        cpu_blurred = gpu_blurred.download()

        sharpened = float(amount + 1) * cpu_frame - float(amount) * cpu_blurred
        return np.clip(sharpened, 0, 255)

def denoise_gaussian_effect(frame_or_gpu_mat, strength):
    """Aplikuje Gaussovo odšumění na jeden snímek (CPU/GPU kompatibilní)."""
    if isinstance(frame_or_gpu_mat, np.ndarray):
        return cv2.GaussianBlur(frame_or_gpu_mat, (2 * strength + 1, 2 * strength + 1), strength)
    else: # Assumed cv2.cuda_GpuMat
        stream = cv2.cuda_Stream()
        # Ensure kernel size is odd
        kernel_size = (2 * strength + 1, 2 * strength + 1)
        # CreateGaussianFilter for BGR works directly
        gpu_denoised = cv2.cuda.createGaussianFilter(frame_or_gpu_mat.type(), kernel_size, strength, strength, cv2.BORDER_DEFAULT).apply(frame_or_gpu_mat, stream)
        return gpu_denoised


def denoise_fast_nl_means_effect(frame, h_param, templateWindowSize, searchWindowSize):
    """Aplikuje Non-Local Means odšumění na jeden snímek (CPU pouze, jak bylo diskutováno)."""
    # NLM barevná verze v cv2.cuda API není přímo dostupná, nebo je složitá pro implementaci.
    # Ponecháme ji na CPU.
    if len(frame.shape) == 3: # Color image
        return cv2.fastNlMeansDenoisingColored(frame, None, h_param, h_param, templateWindowSize, searchWindowSize)
    else: # Grayscale image
        return cv2.fastNlMeansDenoising(frame, None, h_param, templateWindowSize, searchWindowSize)


def focus_mask_effect(frame_or_gpu_mat, amount, radius):
    """Aplikuje focus mask na jeden snímek (CPU/GPU kompatibilní)."""
    if isinstance(frame_or_gpu_mat, np.ndarray):
        blurred = cv2.GaussianBlur(frame_or_gpu_mat, (2 * radius + 1, 2 * radius + 1), radius)
        sharpened = float(amount + 1) * frame_or_gpu_mat - float(amount) * blurred
        return np.clip(sharpened, 0, 255)
    else: # Assumed cv2.cuda_GpuMat, similar to unsharp_mask_effect
        stream = cv2.cuda_Stream()
        gpu_blurred = cv2.cuda.createGaussianFilter(frame_or_gpu_mat.type(), (2 * radius + 1, 2 * radius + 1), radius, radius, cv2.BORDER_DEFAULT).apply(frame_or_gpu_mat, stream)
        
        # Hybrid approach: download and finish on CPU
        cpu_frame = frame_or_gpu_mat.download()
        cpu_blurred = gpu_blurred.download()
        sharpened = float(amount + 1) * cpu_frame - float(amount) * cpu_blurred
        return np.clip(sharpened, 0, 255)


def smart_focus_effect(frame, amount, radius, edge_threshold, saliency_threshold):
    """Aplikuje smart focus na jeden snímek (CPU pouze, pro komplexní pipeline)."""
    # Jak bylo diskutováno, tato komplexní pipeline zůstává na CPU pro tuto verzi.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 3)

    # Fourier transform pro saliency map
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    log_magnitude = np.log1p(magnitude)
    saliency_map = np.fft.ifftshift(np.fft.ifft2(np.exp(log_magnitude + 1j * phase)).real)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    saliency_mask = cv2.threshold(saliency_map, int(saliency_threshold * 255), 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_or(edges, saliency_mask)

    blurred = cv2.GaussianBlur(frame, (2 * radius + 1, 2 * radius + 1), radius)
    sharpened = float(amount + 1) * frame - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    result = np.where(combined_mask[:, :, None] != 0, sharpened, frame)
    return result

def gradient_mask_effect(frame_or_gpu_mat, blur_strength, sharp_strength):
    """Aplikuje gradient mask na jeden snímek (CPU/GPU kompatibilní, hybridní)."""
    if isinstance(frame_or_gpu_mat, np.ndarray):
        frame = frame_or_gpu_mat
    else: # Assumed cv2.cuda_GpuMat
        frame = frame_or_gpu_mat.download() # Always download for complex blend logic

    height, width, _ = frame.shape
    gradient = np.linspace(0, 1, height).reshape(-1, 1).astype(np.float32)
    gradient = np.repeat(gradient, width, axis=1)

    # Blur can be on GPU, then downloaded
    if isinstance(frame_or_gpu_mat, np.ndarray): # If original input was CPU
        blurred = cv2.GaussianBlur(frame, (2 * blur_strength + 1, 2 * blur_strength + 1), blur_strength)
    else: # If original input was GpuMat and CUDA available
        stream = cv2.cuda_Stream()
        gpu_temp = cv2.cuda_GpuMat()
        gpu_temp.upload(frame) # Re-upload to GPU for blur if it was downloaded
        gpu_blurred = cv2.cuda.createGaussianFilter(gpu_temp.type(), (2 * blur_strength + 1, 2 * blur_strength + 1), blur_strength, blur_strength, cv2.BORDER_DEFAULT).apply(gpu_temp, stream)
        blurred = gpu_blurred.download()

    sharpened = float(sharp_strength + 1) * frame - float(sharp_strength) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    combined = (gradient[:, :, None] * sharpened + (1 - gradient[:, :, None]) * blurred)
    return np.clip(combined, 0, 255)


def adaptive_focus_effect(frame, base_amount_range, base_radius_range):
    """Aplikuje adaptivní zaostření na jeden snímek (CPU pouze, protože analýza je CPU)."""
    # Analýza snímku pro adaptivní zaostření je na CPU.
    # Proto i aplikace efektu by měla být na CPU pro konzistenci a jednoduchost.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200) # Pevné Canny prahy pro per-frame analýzu
    edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

    # Dynamické nastavení parametrů na základě obsahu snímku
    if texture < 50: # Nízká textura - jemné zaostření
        amount = base_amount_range[0]
        radius = base_radius_range[0]
    elif edge_density > 0.1: # Vysoká hustota hran - střední zaostření
        amount = (base_amount_range[0] + base_amount_range[1]) / 2
        radius = (base_radius_range[0] + base_radius_range[1]) / 2
    else: # Standardní zaostření
        amount = base_amount_range[1]
        radius = base_radius_range[1]

    if brightness < 50: # Tmavý snímek - zvýšit ostrost
        amount = min(amount * 1.2, 2.5)

    blurred = cv2.GaussianBlur(frame, (2 * int(radius) + 1, 2 * int(radius) + 1), int(radius))
    sharpened = float(amount + 1) * frame - float(amount) * blurred
    return np.clip(sharpened, 0, 255)


def color_selective_focus_effect(frame_or_gpu_mat, target_color_bgr, tolerance, amount, radius):
    """Aplikuje selektivní zaostření na základě barvy na jeden snímek (CPU/GPU hybridní)."""
    # Barvy a maskování je pro jednoduchost CPU
    if isinstance(frame_or_gpu_mat, np.ndarray):
        frame = frame_or_gpu_mat
    else:
        frame = frame_or_gpu_mat.download() # Download for inRange and blending

    lower_bound = np.maximum(np.array(target_color_bgr) - tolerance, 0)
    upper_bound = np.minimum(np.array(target_color_bgr) + tolerance, 255)
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    
    # Blur/Sharpen can be GPU accelerated
    if config.USE_CUDA_GPU and config.CUDA_AVAILABLE and isinstance(frame_or_gpu_mat, cv2.cuda_GpuMat):
        stream = cv2.cuda_Stream()
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame) # Re-upload to GPU for blur if it was downloaded for mask
        gpu_blurred = cv2.cuda.createGaussianFilter(gpu_frame.type(), (2 * radius + 1, 2 * radius + 1), radius, radius, cv2.BORDER_DEFAULT).apply(gpu_frame, stream)
        # Arithmetic for sharpen is still CPU-bound after download for simplicity
        cpu_blurred = gpu_blurred.download()
        sharpened = float(amount + 1) * frame - float(amount) * cpu_blurred
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else: # CPU path
        blurred = cv2.GaussianBlur(frame, (2 * radius + 1, 2 * radius + 1), radius)
        sharpened = float(amount + 1) * frame - float(amount) * blurred
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
    result = mask_3d * sharpened + (1 - mask_3d) * frame
    return np.clip(result, 0, 255)


def color_correction_effect(frame_or_gpu_mat, brightness, contrast, saturation, hue):
    """
    Aplikuje základní korekce barev (jas, kontrast, saturace, odstín) na jeden snímek (CPU/GPU hybridní).
    """
    if isinstance(frame_or_gpu_mat, np.ndarray):
        frame = frame_or_gpu_mat
    else:
        frame = frame_or_gpu_mat.download() # Pro složitější úpravy barev stahujeme na CPU

    corrected_frame = frame.astype(np.float32)

    # Jas
    corrected_frame = corrected_frame + brightness 

    # Kontrast 
    corrected_frame = ((corrected_frame / 255.0 - 0.5) * contrast + 0.5) * 255.0

    # Saturace a Odstín (v HSV)
    # GPU konverze na HSV je možná, ale úpravy HSV kanálů a zpětná konverze je komplexnější pro GPU.
    # Prozatím zůstaneme na CPU pro HSV operace.
    hsv_frame = cv2.cvtColor(corrected_frame.astype(np.uint8), cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv_frame)

    s = np.clip(s * saturation, 0, 255).astype(np.uint8)

    h = h.astype(np.int32) + int(hue / 360.0 * 255.0) 
    h = np.mod(h, 256).astype(np.uint8) 

    hsv_frame = cv2.merge([h, s, v])
    final_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR_FULL)

    final_frame = np.clip(final_frame, 0, 255)
    return final_frame.astype(np.uint8)


# --- Funkce pro zpracování klipu (integrující efekty a ROI) ---

def unsharp_mask(clip, amount_range=(1.0, 1.5), radius=5, roi=None, use_cuda=False):
    total_frames = int(clip.fps * clip.duration)
    
    def make_frame_processor(get_frame, t):
        amount = amount_range[0] + (amount_range[1] - amount_range[0]) * (t / clip.duration)
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), unsharp_mask_effect, roi, total_frames, t, use_cuda, amount, radius)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def denoise(clip, denoise_type=config.DENOISE_ALGO_GAUSSIAN, strength=10, h_param=10, templateWindowSize=7, searchWindowSize=21, roi=None, use_cuda=False):
    total_frames = int(clip.fps * clip.duration)
    
    def make_frame_processor(get_frame, t):
        if denoise_type == config.DENOISE_ALGO_GAUSSIAN:
            processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), denoise_gaussian_effect, roi, total_frames, t, use_cuda, strength)
        elif denoise_type == config.DENOISE_ALGO_NL_MEANS:
            # NLM is CPU-only in this implementation
            processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), denoise_fast_nl_means_effect, roi, total_frames, t, False, h_param, templateWindowSize, searchWindowSize)
        else:
            processed_frame = get_frame(t) # Should not happen, but fallback
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def focus_mask(clip, amount_range=(1.0, 1.5), radius_range=(50, 100), roi=None, use_cuda=False):
    min_amount, max_amount = amount_range
    min_radius, max_radius = radius_range
    total_frames = int(clip.fps * clip.duration)

    def make_frame_processor(get_frame, t):
        amount = min_amount + (max_amount - min_amount) * (t / clip.duration)
        radius = int(min_radius + (max_radius - min_radius) * (t / clip.duration))
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), focus_mask_effect, roi, total_frames, t, use_cuda, amount, radius)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def smart_focus(clip, amount_range=(1.0, 1.5), radius_range=(20, 40), 
                edge_threshold_range=(80, 120), saliency_threshold_range=(0.5, 0.7), roi=None, use_cuda=False):
    min_amount, max_amount = amount_range
    min_radius, max_radius = radius_range
    min_edge_threshold, max_edge_threshold = edge_threshold_range
    min_saliency_threshold, max_saliency_threshold = saliency_threshold_range
    total_frames = int(clip.fps * clip.duration)

    def make_frame_processor(get_frame, t):
        amount = min_amount + (max_amount - min_amount) * (t / clip.duration)
        radius = int(min_radius + (max_radius - min_radius) * (t / clip.duration))
        edge_threshold = int(min_edge_threshold + (max_edge_threshold - min_edge_threshold) * (t / clip.duration))
        saliency_threshold = min_saliency_threshold + (max_saliency_threshold - min_saliency_threshold) * (t / clip.duration)
        
        # Smart focus is CPU-only
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), smart_focus_effect, roi, total_frames, t, False,
                                               amount, radius, edge_threshold, saliency_threshold)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def gradient_mask(clip, blur_strength=10, sharp_strength=1.5, roi=None, use_cuda=False):
    total_frames = int(clip.fps * clip.duration)

    def make_frame_processor(get_frame, t):
        # Gradient mask is CPU-hybrid (blur can be GPU, but blending is CPU)
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), gradient_mask_effect, roi, total_frames, t, use_cuda, blur_strength, sharp_strength)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def stabilize_video(clip):
    """Stabilizuje roztřesené video. CPU pouze."""
    # Stabilizace je CPU-only v této implementaci.
    total_frames = int(clip.fps * clip.duration)
    
    transforms = []
    prev_gray = None
    
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype="uint8")):
        if config.stop_processing_event.is_set():
            return clip 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            if prev_pts is not None and len(prev_pts) >= 4:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                idx = np.where(status == 1)[0]
                if len(idx) >= 4:
                    prev_pts = prev_pts[idx]
                    curr_pts = curr_pts[idx]
                    try:
                        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if m is not None:
                            transforms.append(m)
                        else: 
                            transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)) 
                    except cv2.error:
                        transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
                else:
                    transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            else:
                transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        else:
            transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            
        prev_gray = gray
        utils.update_progress(int((i / total_frames) * 30)) 

    if len(transforms) == 0:
        return clip 

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.zeros_like(trajectory)
    
    window_size = min(15, len(transforms)) 
    for i in range(len(transforms)):
        start = max(0, i - window_size // 2)
        end = min(len(transforms), i + window_size // 2 + 1)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)
    
    smoothed_transforms = []
    for i in range(len(transforms)):
        diff = smoothed_trajectory[i] - trajectory[i]
        corrected_transform = transforms[i] + diff.reshape(2, 3)
        smoothed_transforms.append(corrected_transform)
    
    def stabilize_frame_processor(get_frame, t):
        if config.stop_processing_event.is_set():
            return get_frame(t) 
            
        frame_idx = int(t * clip.fps)
        if frame_idx >= len(smoothed_transforms): 
            frame_idx = len(smoothed_transforms) - 1
        
        frame = get_frame(t)
        transform = smoothed_transforms[frame_idx]
        
        h, w = frame.shape[:2]
        try:
            # warpAffine can be GPU accelerated if GPU mat is passed, but the rest of stabilization is CPU.
            # For simplicity, keep it CPU, or wrap this specific operation in CUDA block if frame is GpuMat.
            stabilized = cv2.warpAffine(frame, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            border = int(min(w, h) * 0.05) 
            if h - 2*border > 0 and w - 2*border > 0:
                stabilized = stabilized[border:h-border, border:w-border]
                stabilized = cv2.resize(stabilized, (w, h), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            print(f"Chyba při stabilizaci snímku {frame_idx}: {e}")
            stabilized = frame 
        
        utils.update_progress(30 + int((frame_idx / total_frames) * 70)) 
        return stabilized.astype(np.uint8)
        
    return clip.fl(stabilize_frame_processor)

def adaptive_focus(clip, suggested_params, roi=None, use_cuda=False):
    """Automaticky analyzuje obsah videa a aplikuje vhodné parametry zaostření. CPU pouze."""
    # Adaptive focus relies on per-frame analysis which is CPU-bound in this implementation.
    total_frames = int(clip.fps * clip.duration)
    
    base_amount_range = suggested_params.get('amount_range', (1.0, 1.5))
    base_radius_range = suggested_params.get('radius_range', (5, 10))

    def make_frame_processor(get_frame, t):
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), adaptive_focus_effect, roi, total_frames, t, False,
                                               base_amount_range, base_radius_range)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)


def color_selective_focus(clip, target_color_bgr, tolerance=30, amount=1.5, radius=5, roi=None, use_cuda=False):
    total_frames = int(clip.fps * clip.duration)

    def make_frame_processor(get_frame, t):
        # Color selective focus is CPU-hybrid
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), color_selective_focus_effect, roi, total_frames, t, use_cuda,
                                               target_color_bgr, tolerance, amount, radius)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)

def color_correction(clip, brightness=0, contrast=1.0, saturation=1.0, hue=0, roi=None, use_cuda=False):
    total_frames = int(clip.fps * clip.duration)

    def make_frame_processor(get_frame, t):
        # Color correction is CPU-hybrid
        processed_frame = _apply_effect_to_frame_with_roi(get_frame(t), color_correction_effect, roi, total_frames, t, use_cuda,
                                               brightness, contrast, saturation, hue)
        utils.update_progress(30 + int((t / clip.duration) * 70))
        return processed_frame
    
    return clip.fl(make_frame_processor)


# --- Hlavní orchestrátor zpracování ---

def process_video_main(input_path, output_path, mode, current_gui_params, selected_roi, use_cuda_enabled):
    """
    Orchestrování zpracování videa na základě vybraného režimu a aktuálních GUI parametrů.
    Tato funkce NEPROVÁDÍ analýzu ani návrhy parametrů. Používá to, co je poskytnuto.
    """
    config.stop_processing_event.clear() # Zajistit, že událost je na začátku čistá
    utils.update_progress(0) # Resetovat průběh na začátku

    clip = None
    processed_clip = None
    try:
        is_valid, msg = utils.check_video_file(input_path)
        if not is_valid:
            raise ValueError(msg)

        clip = VideoFileClip(input_path)
        if clip.audio is None:
            print("Upozornění: Video nemá zvukovou stopu.") 

        # Determine if CUDA should actually be used for this processing run
        actual_use_cuda = use_cuda_enabled and config.CUDA_AVAILABLE

        if mode == "unsharp":
            processed_clip = unsharp_mask(
                clip, 
                amount_range=current_gui_params.get("amount_range"), 
                radius=current_gui_params.get("radius"),
                roi=selected_roi,
                use_cuda=actual_use_cuda
            )
        elif mode == "focus_mask":
            processed_clip = focus_mask(
                clip, 
                amount_range=current_gui_params.get("amount_range"), 
                radius_range=current_gui_params.get("radius_range"),
                roi=selected_roi,
                use_cuda=actual_use_cuda
            )
        elif mode == "smart_focus":
            processed_clip = smart_focus(
                clip, # Smart Focus is CPU-only
                amount_range=current_gui_params.get("amount_range"),
                radius_range=current_gui_params.get("radius_range"),
                edge_threshold_range=current_gui_params.get("edge_threshold_range"),
                saliency_threshold_range=current_gui_params.get("saliency_threshold_range"),
                roi=selected_roi,
                use_cuda=False # Explicitly False for smart_focus
            )
        elif mode == "denoise":
            processed_clip = denoise(
                clip, 
                denoise_type=current_gui_params.get("denoise_type"),
                strength=current_gui_params.get("strength"),
                h_param=current_gui_params.get("nlm_h"),
                templateWindowSize=current_gui_params.get("nlm_template_size"),
                searchWindowSize=current_gui_params.get("nlm_search_size"),
                roi=selected_roi,
                # NLM is CPU-only, Gaussian can be CUDA
                use_cuda=(actual_use_cuda and current_gui_params.get("denoise_type") == config.DENOISE_ALGO_GAUSSIAN)
            )
        elif mode == "gradient_mask":
            processed_clip = gradient_mask(
                clip, 
                blur_strength=current_gui_params.get("blur_strength"), 
                sharp_strength=current_gui_params.get("sharp_strength"),
                roi=selected_roi,
                use_cuda=actual_use_cuda # Blur can be CUDA
            )
        elif mode == "stabilize":
            processed_clip = stabilize_video(clip) # Stabilize is CPU-only
        elif mode == "adaptive_focus":
            suggested_params_for_adaptive = current_gui_params.get('suggested_params', {})
            if not suggested_params_for_adaptive:
                 import video_analyzer # Importovat lokálně
                 dummy_analysis_results = { 'blurriness': 100, 'noise': 10, 'brightness': 100, 'contrast': 50, 'motion': 0 }
                 suggested_params_for_adaptive = video_analyzer.adaptive_parameter_suggestion(dummy_analysis_results)
            
            processed_clip = adaptive_focus(
                clip, # Adaptive Focus is CPU-only
                suggested_params=suggested_params_for_adaptive,
                roi=selected_roi,
                use_cuda=False # Explicitly False for adaptive_focus
            )
        elif mode == "color_selective_focus":
            processed_clip = color_selective_focus(
                clip,
                target_color_bgr=current_gui_params.get("target_color"),
                tolerance=current_gui_params.get("tolerance"),
                amount=current_gui_params.get("amount"), 
                radius=current_gui_params.get("radius"), 
                roi=selected_roi,
                use_cuda=actual_use_cuda # Sharpen can be CUDA
            )
        elif mode == "color_correction":
            processed_clip = color_correction(
                clip,
                brightness=current_gui_params.get("brightness"),
                contrast=current_gui_params.get("contrast"),
                saturation=current_gui_params.get("saturation"),
                hue=current_gui_params.get("hue"),
                roi=selected_roi,
                use_cuda=actual_use_cuda # Base color adjust can be CUDA
            )
        else:
            raise ValueError(f"Neplatný režim: {mode}")

        if config.stop_processing_event.is_set():
            return False 
            
        processed_clip.write_videofile(
            output_path, 
            codec=config.VIDEO_CODEC, 
            audio_codec=config.AUDIO_CODEC
        )
        
        return True 

    except Exception as e:
        print(f"Chyba při zpracování videa: {e}") # Pro ladění
        return False 
    finally:
        if clip:
            clip.close()
        if processed_clip:
            processed_clip.close()
        config.stop_processing_event.clear()
