"""
FFmpeg Video Composition API - v6
Reference layout: image fills top half, avatar large in bottom half,
avatar head overlaps image significantly.
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

# Layout matching reference:
# Image: top 50% (0-960px), fills full width
# Avatar: bottom 60% (770-1920px), fills full width
# Overlap: ~190px where avatar head goes over image
IMAGE_HEIGHT = 960
AVATAR_HEIGHT = 1150
AVATAR_TOP = 770


def download_file(url, local_path):
    logger.info(f"Downloading: {url[:100]}...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(local_path)
        logger.info(f"Downloaded: {size} bytes -> {local_path}")
        if size < 1000:
            logger.warning(f"File suspiciously small: {size} bytes")
            return None
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


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

    # Images — fill full width, crop to fit height
    num_images = len(image_inputs)
    if num_images > 0:
        image_dur = duration / num_images
        for i, img_input in enumerate(image_inputs):
            filter_parts.append(
                f"[{img_input}:v]"
                f"scale={WIDTH}:{IMAGE_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{IMAGE_HEIGHT},"
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

    # Avatar — fill full width, crop to fit
    filter_parts.append(
        f"[{avatar_input}:v]"
        f"scale={WIDTH}:{AVATAR_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{AVATAR_HEIGHT},"
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
        return jsonify({'status': 'healthy', 'ffmpeg': version, 'api_version': 'v6'})
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
            return jsonify({'status': 'error', 'error': 'avatar_video_url required'}), 400

        job_id = str(uuid.uuid4())[:8]
        job_temp = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_temp, exist_ok=True)

        logger.info(f"[{job_id}] Starting: {len(images)} images, {duration}s")

        # Download avatar
        avatar_path = os.path.join(job_temp, 'avatar.mp4')
        result = download_file(avatar_url, avatar_path)
        if not result:
            return jsonify({'status': 'error', 'error': 'Failed to download avatar video'}), 500

        # Probe actual duration
        try:
            probe = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', avatar_path
            ], capture_output=True, text=True, timeout=15)
            actual = float(json.loads(probe.stdout).get('format', {}).get('duration', duration))
            if actual > 0:
                duration = actual
                logger.info(f"[{job_id}] Avatar duration: {duration:.1f}s")
        except:
            pass

        # Download images (max 6)
        image_paths = []
        for i, img_url in enumerate(images[:6]):
            ext = '.png' if '.png' in img_url.lower() else '.jpg'
            img_path = os.path.join(job_temp, f'img_{i}{ext}')
            result = download_file(img_url, img_path)
            if result:
                image_paths.append(result)
            else:
                logger.warning(f"[{job_id}] Image {i} failed to download")

        # If no images downloaded, log it but continue (avatar only)
        if not image_paths:
            logger.warning(f"[{job_id}] NO images downloaded! Video will be avatar-only.")

        # Compose
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
        cmd = build_ffmpeg_command(avatar_path, image_paths, duration, output_path)

        logger.info(f"[{job_id}] Running FFmpeg with {len(image_paths)} images...")
        ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if ffmpeg_result.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg STDERR: {ffmpeg_result.stderr[-2000:]}")
            return jsonify({
                'status': 'error',
                'error': f'FFmpeg failed (code {ffmpeg_result.returncode})',
                'stderr': ffmpeg_result.stderr[-500:]
            }), 500

        if not os.path.exists(output_path):
            return jsonify({'status': 'error', 'error': 'Output not created'}), 500

        file_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] Done: {file_size} bytes, {len(image_paths)} images used")

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
            'file_size': file_size,
            'images_used': len(image_paths)
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
