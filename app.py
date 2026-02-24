"""
Video Composition API - v13 (OpenCV)
======================================
Avatar is UNTOUCHED — no scaling, no cropping, no repositioning.
Only operation: place B-roll image(s) BEHIND avatar.

v13: Complete rewrite using OpenCV frame-by-frame compositing.
- Replaces FFmpeg filter chains (colorkey/lumakey) with pixel-level masking
- cv2.threshold on grayscale for precise black detection
- Morphological operations for clean mask edges
- FFmpeg used ONLY for final audio mux (not for compositing)
- Full debug logging with frame dimensions at each step
"""

import os
import json
import uuid
import subprocess
import shutil
import requests
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

OUTPUT_DIR = "/tmp/reel-output"
TEMP_DIR = "/tmp/reel-temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Masking settings ─────────────────────────────────────────────
# Pixels with luminance (grayscale value) <= LUMA_THRESHOLD are "black background"
# Range: 0-255 (not 0.0-1.0 like FFmpeg lumakey)
#   20 = catches H.264 compressed black (#000000 → ~#141414 ≈ grayscale ~18)
#   Brown jacket has grayscale ~70-80 — well above threshold
#   Dark hair has grayscale ~30-40 — still above threshold
LUMA_THRESHOLD = 20

# Gaussian blur kernel size for mask edge smoothing (must be odd, 0=off)
MASK_BLUR = 5

# Morphological kernel size for cleaning mask edges
MORPH_KERNEL = 3


def download_file(url, local_path):
    """Download a file from URL to local path. Returns path or None."""
    logger.info(f"Downloading: {url[:100]}...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(local_path)
        if size < 1000:
            logger.warning(f"File too small ({size} bytes): {local_path}")
            return None
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def probe_video(video_path):
    """Get video dimensions, duration, FPS, and audio info via ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)

        video_stream = None
        has_audio = False
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_stream:
                video_stream = stream
            if stream.get('codec_type') == 'audio':
                has_audio = True

        if not video_stream:
            return None

        width = int(video_stream.get('width', 1080))
        height = int(video_stream.get('height', 1920))

        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = round(int(num) / max(int(den), 1), 2)
        else:
            fps = float(fps_str)
        fps = max(15, min(fps, 60))

        duration = float(data.get('format', {}).get('duration', 0))
        if duration <= 0:
            duration = float(video_stream.get('duration', 35))

        nb_frames = int(video_stream.get('nb_frames', 0))
        if nb_frames <= 0:
            nb_frames = int(fps * duration)

        info = {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'nb_frames': nb_frames,
            'has_audio': has_audio
        }
        logger.info(f"Probed video: {info}")
        return info

    except Exception as e:
        logger.error(f"Probe failed: {e}")
        return None


def prepare_broll_image(img_path, target_width):
    """
    Load and scale B-roll image to target_width, preserving aspect ratio.
    Returns BGR numpy array or None.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Failed to read image: {img_path}")
        return None

    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    new_h = new_h if new_h % 2 == 0 else new_h + 1

    scaled = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LANCZOS4)
    logger.info(f"B-roll: {img_path} — {w}x{h} → {target_width}x{new_h}")
    return scaled


def compose_frame(avatar_frame, bg_frame, luma_threshold, mask_blur, morph_kernel):
    """
    Composite one avatar frame over one background frame.

    1. Convert avatar to grayscale (luminance)
    2. Binary mask: bright pixels (avatar) = 255, dark pixels (bg) = 0
    3. Morphological close → fills small holes in avatar (clothing texture)
    4. Morphological open → removes small noise in background region
    5. Gaussian blur → soft edges
    6. Alpha blend: output = bg * (1 - alpha) + avatar * alpha
    """
    gray = cv2.cvtColor(avatar_frame, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, luma_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if mask_blur > 0:
        blur_k = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    alpha = mask.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)

    composited = (bg_frame.astype(np.float32) * (1.0 - alpha_3ch) +
                  avatar_frame.astype(np.float32) * alpha_3ch)
    return composited.astype(np.uint8)


def build_background_frame(canvas_h, canvas_w, broll_image, image_y):
    """
    Black canvas with B-roll image placed at image_y, centered horizontally.
    """
    bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if broll_image is None:
        return bg

    img_h, img_w = broll_image.shape[:2]

    x_offset = max(0, (canvas_w - img_w) // 2)

    y_start = max(0, image_y)
    y_end = min(canvas_h, image_y + img_h)
    x_start = x_offset
    x_end = min(canvas_w, x_offset + img_w)

    src_y_start = max(0, -image_y)
    src_y_end = src_y_start + (y_end - y_start)
    src_x_end = x_end - x_start

    if y_end > y_start and x_end > x_start:
        bg[y_start:y_end, x_start:x_end] = broll_image[src_y_start:src_y_end, 0:src_x_end]

    return bg


def compose_video(avatar_path, image_paths, video_info, output_path,
                  luma_threshold=LUMA_THRESHOLD, mask_blur=MASK_BLUR,
                  morph_kernel=MORPH_KERNEL):
    """
    Main compositing: read avatar frame-by-frame, composite with B-roll,
    write output. Audio muxed separately with FFmpeg.
    """
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    duration = video_info['duration']
    has_audio = video_info.get('has_audio', False)

    image_y = round(height / 6)
    image_y = image_y if image_y % 2 == 0 else image_y + 1

    logger.info(f"Compositing: {width}x{height}, image_y={image_y}, fps={fps}, "
                f"dur={duration:.1f}s, thresh={luma_threshold}, "
                f"blur={mask_blur}, morph={morph_kernel}")

    # ── Prepare B-roll images ──
    broll_images = []
    for img_path in image_paths:
        scaled = prepare_broll_image(img_path, width)
        if scaled is not None:
            broll_images.append(scaled)

    num_images = len(broll_images)
    if num_images == 0:
        logger.warning("No valid B-roll images — avatar over black only")

    # ── Open avatar ──
    cap = cv2.VideoCapture(avatar_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open avatar video: {avatar_path}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"VideoCapture opened: {actual_w}x{actual_h}, "
                f"fps={actual_fps}, frames={total_frames}")

    # Use actual dimensions from capture (most reliable)
    if actual_w > 0 and actual_h > 0:
        if actual_w != width or actual_h != height:
            logger.info(f"Dimensions differ from probe! Using capture: {actual_w}x{actual_h}")
            width = actual_w
            height = actual_h
            image_y = round(height / 6)
            image_y = image_y if image_y % 2 == 0 else image_y + 1
            # Re-scale images
            broll_images = []
            for img_path in image_paths:
                scaled = prepare_broll_image(img_path, width)
                if scaled is not None:
                    broll_images.append(scaled)
            num_images = len(broll_images)

    if actual_fps > 0:
        fps = actual_fps

    # ── Video writer ──
    temp_video = output_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create VideoWriter")

    # ── Frame loop ──
    frame_idx = 0
    frames_written = 0
    image_dur = duration / max(num_images, 1)

    # Pre-build background frames for each B-roll image (cache)
    bg_cache = {}
    for i, broll in enumerate(broll_images):
        bg_cache[i] = build_background_frame(height, width, broll, image_y)
    # Also cache empty background
    bg_empty = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        ret, avatar_frame = cap.read()
        if not ret:
            break

        # Pick current B-roll based on time
        current_time = frame_idx / fps
        if num_images > 0:
            img_index = min(int(current_time / image_dur), num_images - 1)
            bg_frame = bg_cache[img_index]
        else:
            bg_frame = bg_empty

        composited = compose_frame(
            avatar_frame, bg_frame,
            luma_threshold, mask_blur, morph_kernel
        )

        writer.write(composited)
        frames_written += 1
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = (frame_idx / max(total_frames, 1)) * 100
            logger.info(f"Progress: {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()
    logger.info(f"Wrote {frames_written} frames to temp video")

    # ── Mux audio with FFmpeg ──
    if has_audio:
        logger.info("Muxing audio from avatar...")
        mux_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', avatar_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-shortest',
            output_path
        ]
        r = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logger.error(f"Audio mux failed: {r.stderr[-1000:]}")
            shutil.copy2(temp_video, output_path)
    else:
        encode_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        r = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logger.error(f"Encode failed: {r.stderr[-1000:]}")
            shutil.copy2(temp_video, output_path)

    try:
        os.remove(temp_video)
    except:
        pass

    return frames_written


# ═══════════════════════════════════════════════════════════════════
# Flask routes
# ═══════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True, timeout=5)
        ffmpeg_ver = result.stdout.split('\n')[0] if result.stdout else 'unknown'

        return jsonify({
            'status': 'healthy',
            'api_version': 'v13-opencv',
            'ffmpeg': ffmpeg_ver,
            'opencv': cv2.__version__,
            'numpy': np.__version__,
            'defaults': {
                'luma_threshold': LUMA_THRESHOLD,
                'mask_blur': MASK_BLUR,
                'morph_kernel': MORPH_KERNEL
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/compose', methods=['POST'])
def compose():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'error': 'No JSON body'}), 400

        avatar_url = data.get('avatar_video_url', '')
        images = data.get('images', [])

        if not avatar_url:
            return jsonify({'status': 'error', 'error': 'avatar_video_url required'}), 400

        # Tuning overrides (all optional)
        luma_threshold = int(data.get('luma_threshold', LUMA_THRESHOLD))
        mask_blur = int(data.get('mask_blur', MASK_BLUR))
        morph_kernel = int(data.get('morph_kernel', MORPH_KERNEL))

        # Clamp to safe ranges
        luma_threshold = max(5, min(luma_threshold, 80))
        mask_blur = max(0, min(mask_blur, 15))
        morph_kernel = max(1, min(morph_kernel, 9))

        job_id = str(uuid.uuid4())[:8]
        job_temp = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_temp, exist_ok=True)

        logger.info(f"[{job_id}] Starting: {len(images)} images, "
                    f"luma={luma_threshold}, blur={mask_blur}, morph={morph_kernel}")

        # ── Download avatar ──
        avatar_path = os.path.join(job_temp, 'avatar.mp4')
        dl = download_file(avatar_url, avatar_path)
        if not dl:
            return jsonify({'status': 'error', 'error': 'Failed to download avatar video'}), 500

        # ── Probe ──
        video_info = probe_video(avatar_path)
        if not video_info:
            return jsonify({'status': 'error', 'error': 'Failed to probe avatar video'}), 500

        if data.get('duration'):
            video_info['duration'] = float(data['duration'])

        # ── Download images (max 6) ──
        image_paths = []
        for i, img_url in enumerate(images[:6]):
            ext = '.png' if '.png' in img_url.lower() else '.jpg'
            img_path = os.path.join(job_temp, f'img_{i}{ext}')
            dl = download_file(img_url, img_path)
            if dl:
                image_paths.append(dl)

        logger.info(f"[{job_id}] Downloaded {len(image_paths)}/{len(images)} images")

        # ── Compose ──
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')

        frames = compose_video(
            avatar_path, image_paths, video_info, output_path,
            luma_threshold=luma_threshold,
            mask_blur=mask_blur,
            morph_kernel=morph_kernel
        )

        if not os.path.exists(output_path):
            return jsonify({'status': 'error', 'error': 'Output file not created'}), 500

        file_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] SUCCESS: {file_size} bytes, {frames} frames")

        # Cleanup
        try:
            shutil.rmtree(job_temp, ignore_errors=True)
        except:
            pass

        base_url = request.host_url.rstrip('/')
        return jsonify({
            'status': 'success',
            'video_url': f"{base_url}/output/{job_id}.mp4",
            'job_id': job_id,
            'duration': round(video_info['duration'], 1),
            'file_size': file_size,
            'frames_composited': frames,
            'images_used': len(image_paths),
            'video_dimensions': f"{video_info['width']}x{video_info['height']}",
            'fps': video_info['fps'],
            'settings': {
                'luma_threshold': luma_threshold,
                'mask_blur': mask_blur,
                'morph_kernel': morph_kernel
            }
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
