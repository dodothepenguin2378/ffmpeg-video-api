"""
FFmpeg Video Composition API - v4
Simple layout: images behind, avatar in front (lower half), overlapping.

Layout (1080x1920):
┌─────────────────────┐
│                     │
│   IMAGE (fills      │
│   top ~60%)         │
│                     │
│  ┌───────────────┐  │
│  │  AVATAR       │  │  overlaps images
│  │  (black bg    │  │
│  │   from HeyGen)│  │
│  │               │  │
│  └───────────────┘  │
└─────────────────────┘

No title. No subtitles. Just images + avatar.
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

WIDTH = 1080
HEIGHT = 1920
FPS = 24
IMAGE_HEIGHT = 1150
AVATAR_HEIGHT = 870
AVATAR_TOP = 1050


def download_file(url, local_path):
    logger.info(f"Downloading: {url[:80]}...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded: {os.path.getsize(local_path)} bytes")
    return local_path


def build_ffmpeg_command(avatar_path, image_paths, duration, output_path):
    inputs = []
    filter_parts = []
    input_idx = 0

    # Input 0: avatar video
    inputs.extend(['-i', avatar_path])
    avatar_input = input_idx
    input_idx += 1

    # Input 1+: images
    image_inputs = []
    for img_path in image_paths:
        inputs.extend(['-loop', '1', '-t', str(duration), '-i', img_path])
        image_inputs.append(input_idx)
        input_idx += 1

    # Black background
    filter_parts.append(
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={duration}:r={FPS}[bg]"
    )

    # Images in top area, cycling
    num_images = len(image_inputs)
    if num_images > 0:
        image_dur = duration / num_images
        for i, img_input in enumerate(image_inputs):
            filter_parts.append(
                f"[{img_input}:v]scale={WIDTH}:{IMAGE_HEIGHT}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1[img{i}]"
            )
        prev = "bg"
        for i in range(num_images):
            start_t = i * image_dur
            end_t = (i + 1) * image_dur
            out = f"bg_img{i}"
            filter_parts.append(
                f"[{prev}][img{i}]overlay=0:0:"
                f"enable='between(t,{start_t:.2f},{end_t:.2f})'[{out}]"
            )
            prev = out
        canvas = prev
    else:
        canvas = "bg"

    # Avatar scaled and overlaid
    filter_parts.append(
        f"[{avatar_input}:v]"
        f"scale={WIDTH}:{AVATAR_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{AVATAR_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1[avatar]"
    )
    filter_parts.append(
        f"[{canvas}][avatar]overlay=0:{AVATAR_TOP}:shortest=1[out]"
    )

    filter_complex = ";\n".join(filter_parts)

    cmd = ['ffmpeg', '-y']
    cmd.extend(inputs)
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', f'{avatar_input}:a?',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-r', str(FPS), '-pix_fmt', 'yuv420p',
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
        return jsonify({'status': 'healthy', 'ffmpeg': version, 'api_version': 'v4'})
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
        duration = float(data.get('duration', 35))

        if not avatar_url:
            return jsonify({'status': 'error', 'error': 'avatar_video_url is required'}), 400

        job_id = str(uuid.uuid4())[:8]
        job_temp = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_temp, exist_ok=True)

        logger.info(f"[{job_id}] Starting: {len(images)} images, {duration}s")

        # Download avatar
        avatar_path = os.path.join(job_temp, 'avatar.mp4')
        download_file(avatar_url, avatar_path)

        # Probe actual duration
        try:
            probe = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', avatar_path
            ], capture_output=True, text=True, timeout=15)
            probe_data = json.loads(probe.stdout)
            actual = float(probe_data.get('format', {}).get('duration', duration))
            if actual > 0:
                duration = actual
        except:
            pass

        # Download images (max 6)
        image_paths = []
        for i, img_url in enumerate(images[:6]):
            try:
                ext = '.png' if '.png' in img_url.lower() else '.jpg'
                img_path = os.path.join(job_temp, f'img_{i}{ext}')
                download_file(img_url, img_path)
                image_paths.append(img_path)
            except Exception as e:
                logger.warning(f"[{job_id}] Image {i} failed: {e}")

        # Fallback placeholder
        if not image_paths:
            ph = os.path.join(job_temp, 'placeholder.png')
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', f'color=c=#1a1a2e:s={WIDTH}x{IMAGE_HEIGHT}:d=1',
                '-frames:v', '1', ph
            ], capture_output=True, timeout=10)
            image_paths = [ph]

        # Compose
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
        cmd = build_ffmpeg_command(avatar_path, image_paths, duration, output_path)

        logger.info(f"[{job_id}] Running FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg STDERR: {result.stderr[-2000:]}")
            return jsonify({
                'status': 'error',
                'error': f'FFmpeg failed (code {result.returncode})',
                'stderr': result.stderr[-500:]
            }), 500

        if not os.path.exists(output_path):
            return jsonify({'status': 'error', 'error': 'Output not created'}), 500

        file_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] Done: {file_size} bytes")

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
            'duration': round(duration, 1),
            'file_size': file_size
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
