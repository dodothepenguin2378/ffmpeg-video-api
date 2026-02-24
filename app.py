"""
FFmpeg Video Composition API - v12
===================================
Avatar is UNTOUCHED — no scaling, no cropping, no repositioning.
Only operation: place B-roll image(s) BEHIND avatar.

Key improvements over v11:
- lumakey instead of colorkey (handles H.264 compression artifacts better)
- Dynamic dimension detection via ffprobe (no hardcoded 1080x1920)
- Dynamic IMAGE_Y calculation (1/6 of actual video height)
- Preserves original FPS from avatar video
- Better error handling and logging
"""

import os
import json
import uuid
import subprocess
import requests
import logging
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

OUTPUT_DIR = "/tmp/reel-output"
TEMP_DIR = "/tmp/reel-temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lumakey settings ──────────────────────────────────────────────
# threshold: luminance below this → fully transparent
#   0.08 = catches compressed black (#000000 → ~#141414) but NOT dark clothing
#   Brown jacket (#8B4513) has luminance ~0.28 — well above threshold
#   Dark hair has luminance ~0.10-0.15 — still above threshold
# tolerance: range around threshold for partial transparency (soft edge)
# softness: additional edge softness
LUMAKEY_THRESHOLD = 0.08
LUMAKEY_TOLERANCE = 0.02
LUMAKEY_SOFTNESS  = 0.05


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
    """
    Get video dimensions, duration, and FPS via ffprobe.
    Returns dict with width, height, duration, fps.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            return None

        width = int(video_stream.get('width', 1080))
        height = int(video_stream.get('height', 1920))

        # FPS: parse r_frame_rate (e.g., "24/1" or "30000/1001")
        fps_str = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = round(int(num) / max(int(den), 1), 2)
        else:
            fps = float(fps_str)
        # Clamp to reasonable range
        fps = max(15, min(fps, 60))

        # Duration from format (more reliable)
        duration = float(data.get('format', {}).get('duration', 0))
        if duration <= 0:
            duration = float(video_stream.get('duration', 35))

        info = {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps
        }
        logger.info(f"Probed video: {info}")
        return info

    except Exception as e:
        logger.error(f"Probe failed: {e}")
        return None


def build_ffmpeg_command(avatar_path, image_paths, video_info, output_path):
    """
    Build FFmpeg command for compositing.
    
    Pipeline:
    1. Black canvas (same size as avatar)
    2. Scale each image to full width, natural height (no crop, no pad)
    3. Place image(s) on canvas at Y = height/6
    4. Apply lumakey to avatar → black bg becomes transparent
    5. Overlay keyed avatar on top at 0:0 (untouched position)
    """
    width = video_info['width']
    height = video_info['height']
    duration = video_info['duration']
    fps = video_info['fps']

    # Dynamic Y position: ~1/6 from top
    image_y = round(height / 6)
    # Make sure it's even (FFmpeg likes even numbers)
    image_y = image_y if image_y % 2 == 0 else image_y + 1

    logger.info(f"Layout: {width}x{height}, image_y={image_y}, fps={fps}, dur={duration:.1f}s")

    inputs = []
    filter_parts = []
    input_idx = 0

    # ── Input 0: Avatar video (untouched) ──
    inputs.extend(['-i', avatar_path])
    avatar_idx = input_idx
    input_idx += 1

    # ── Input 1+: B-roll images ──
    image_indices = []
    for img_path in image_paths:
        inputs.extend(['-loop', '1', '-t', str(duration), '-i', img_path])
        image_indices.append(input_idx)
        input_idx += 1

    # ── STEP 1: Black canvas matching avatar dimensions ──
    filter_parts.append(
        f"color=c=black:s={width}x{height}:d={duration}:r={fps}[bg]"
    )

    # ── STEP 2: Scale images → full width, natural height ──
    num_images = len(image_indices)
    if num_images > 0:
        image_dur = duration / num_images

        for i, img_idx in enumerate(image_indices):
            # scale to exact width, height auto-calculated preserving aspect ratio
            # -2 ensures height is even (required by many codecs)
            filter_parts.append(
                f"[{img_idx}:v]scale={width}:-2,setsar=1[img{i}]"
            )

        # ── STEP 3: Place images on canvas with timing ──
        prev = "bg"
        for i in range(num_images):
            start_t = i * image_dur
            end_t = (i + 1) * image_dur
            out_label = f"bg_img{i}"
            filter_parts.append(
                f"[{prev}][img{i}]overlay=0:{image_y}:"
                f"enable='between(t,{start_t:.3f},{end_t:.3f})'[{out_label}]"
            )
            prev = out_label

        canvas = prev
    else:
        canvas = "bg"

    # ── STEP 4: Lumakey on avatar → black becomes transparent ──
    # lumakey works on luminance (0=black, 1=white)
    # threshold=0.08 catches H.264 compressed black (not pure #000000)
    # tolerance=0.02 gives a small transition zone
    # softness=0.05 smooths edges slightly
    #
    # Why lumakey > colorkey for black backgrounds:
    #   colorkey matches exact RGB color — H.264 compression shifts black to
    #   slightly colored values (#030201, #050304) which colorkey misses at low
    #   similarity or catches too much at high similarity.
    #   lumakey matches on brightness regardless of color shift.
    filter_parts.append(
        f"[{avatar_idx}:v]lumakey="
        f"threshold={LUMAKEY_THRESHOLD}:"
        f"tolerance={LUMAKEY_TOLERANCE}:"
        f"softness={LUMAKEY_SOFTNESS}"
        f"[avatar_keyed]"
    )

    # ── STEP 5: Overlay keyed avatar on canvas+images ──
    # Position 0:0 — avatar is NOT moved, NOT scaled, NOTHING changed
    # shortest=1 ensures output ends when avatar ends
    filter_parts.append(
        f"[{canvas}][avatar_keyed]overlay=0:0:shortest=1[out]"
    )

    filter_complex = ";\n".join(filter_parts)

    cmd = ['ffmpeg', '-y']
    cmd.extend(inputs)
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', f'{avatar_idx}:a?',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-r', str(fps),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-t', str(duration),
        output_path
    ])
    return cmd


@app.route('/health', methods=['GET'])
def health():
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True, timeout=5)
        version = result.stdout.split('\n')[0] if result.stdout else 'unknown'
        return jsonify({
            'status': 'healthy',
            'ffmpeg': version,
            'api_version': 'v12',
            'lumakey': {
                'threshold': LUMAKEY_THRESHOLD,
                'tolerance': LUMAKEY_TOLERANCE,
                'softness': LUMAKEY_SOFTNESS
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

        # Optional overrides for lumakey tuning
        luma_threshold = float(data.get('lumakey_threshold', LUMAKEY_THRESHOLD))
        luma_tolerance = float(data.get('lumakey_tolerance', LUMAKEY_TOLERANCE))
        luma_softness  = float(data.get('lumakey_softness', LUMAKEY_SOFTNESS))

        job_id = str(uuid.uuid4())[:8]
        job_temp = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_temp, exist_ok=True)

        logger.info(f"[{job_id}] Starting: {len(images)} images, lumakey={luma_threshold}/{luma_tolerance}/{luma_softness}")

        # ── Download avatar ──
        avatar_path = os.path.join(job_temp, 'avatar.mp4')
        dl = download_file(avatar_url, avatar_path)
        if not dl:
            return jsonify({'status': 'error', 'error': 'Failed to download avatar video'}), 500

        # ── Probe avatar for dimensions, fps, duration ──
        video_info = probe_video(avatar_path)
        if not video_info:
            return jsonify({'status': 'error', 'error': 'Failed to probe avatar video'}), 500

        # Allow manual duration override (but prefer probed value)
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

        # ── Build and run FFmpeg ──
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')

        # Temporarily override global settings if custom values provided
        global LUMAKEY_THRESHOLD, LUMAKEY_TOLERANCE, LUMAKEY_SOFTNESS
        orig_t, orig_tol, orig_s = LUMAKEY_THRESHOLD, LUMAKEY_TOLERANCE, LUMAKEY_SOFTNESS
        LUMAKEY_THRESHOLD = luma_threshold
        LUMAKEY_TOLERANCE = luma_tolerance
        LUMAKEY_SOFTNESS = luma_softness

        cmd = build_ffmpeg_command(avatar_path, image_paths, video_info, output_path)

        LUMAKEY_THRESHOLD, LUMAKEY_TOLERANCE, LUMAKEY_SOFTNESS = orig_t, orig_tol, orig_s

        # Log the filter_complex for debugging
        fc_idx = cmd.index('-filter_complex') + 1
        logger.info(f"[{job_id}] filter_complex:\n{cmd[fc_idx]}")

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if r.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg FAILED (code {r.returncode}):\n{r.stderr[-2000:]}")
            return jsonify({
                'status': 'error',
                'error': f'FFmpeg failed (code {r.returncode})',
                'stderr': r.stderr[-500:]
            }), 500

        if not os.path.exists(output_path):
            return jsonify({'status': 'error', 'error': 'Output file not created'}), 500

        file_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] SUCCESS: {file_size} bytes")

        # Cleanup temp files
        try:
            import shutil
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
            'images_used': len(image_paths),
            'video_dimensions': f"{video_info['width']}x{video_info['height']}",
            'fps': video_info['fps'],
            'lumakey': {
                'threshold': luma_threshold,
                'tolerance': luma_tolerance,
                'softness': luma_softness
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
