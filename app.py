"""
FFmpeg Video Composition API - v2
Reel Builder: composes news reel from avatar video + images

Layout (1080x1920):
┌─────────────────────┐
│   IMAGE (top 60%)   │  0-1150px
│   scaled/centered   │
├─── overlap zone ────┤  1050-1150px
│   AVATAR (bottom    │  1050-1920px (45%, 100px overlap)
│   45%, chroma key)  │
│                     │
│   [SUBTITLES]       │  ~1750px
└─────────────────────┘
Title banner: first 5 seconds, top area
"""

import os
import json
import uuid
import subprocess
import requests
import math
import textwrap
import logging
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

OUTPUT_DIR = "/tmp/reel-output"
TEMP_DIR = "/tmp/reel-temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Video dimensions
WIDTH = 1080
HEIGHT = 1920
FPS = 24

# Layout zones
IMAGE_HEIGHT = 1150          # Top 60%
AVATAR_HEIGHT = 870          # Bottom 45%
AVATAR_TOP = 1050            # Where avatar starts (100px overlap with image)
SUBTITLE_Y = 1750            # Subtitle position
TITLE_BANNER_DURATION = 5    # Title shown for first 5 seconds


def download_file(url, local_path):
    """Download a file from URL to local path."""
    logger.info(f"Downloading: {url[:80]}...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded to: {local_path} ({os.path.getsize(local_path)} bytes)")
    return local_path


def generate_subtitle_chunks(script, duration, words_per_chunk=4):
    """Split script into subtitle chunks with timing."""
    if not script:
        return []
    words = script.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunks.append(' '.join(chunk_words))

    if not chunks:
        return []

    chunk_duration = duration / len(chunks)
    timed_chunks = []
    for i, text in enumerate(chunks):
        timed_chunks.append({
            'text': text,
            'start': i * chunk_duration,
            'end': (i + 1) * chunk_duration
        })
    return timed_chunks


def build_ffmpeg_command(avatar_path, image_paths, duration, title, script, output_path):
    """
    Build the FFmpeg command for video composition.

    Strategy:
    1. Black background canvas (1080x1920)
    2. Images cycle in top 60% area
    3. Avatar overlaid in bottom 45% with chroma key (green screen removal)
    4. Title banner for first 5 seconds
    5. Subtitles throughout
    """
    job_id = os.path.basename(output_path).replace('.mp4', '')

    # --- Build filter complex ---
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

    # === STEP 1: Create black background ===
    filter_parts.append(
        f"color=c=black:s={WIDTH}x{HEIGHT}:d={duration}:r={FPS}[bg]"
    )

    # === STEP 2: Process images (cycle through them) ===
    num_images = len(image_inputs)
    if num_images > 0:
        image_dur = duration / num_images

        for i, img_input in enumerate(image_inputs):
            start_t = i * image_dur
            end_t = (i + 1) * image_dur

            # Scale image to fill top area (1080x1150), maintaining aspect ratio
            filter_parts.append(
                f"[{img_input}:v]scale={WIDTH}:{IMAGE_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{IMAGE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1[img{i}]"
            )

        # Concatenate images with timing using overlay + enable
        # Start by overlaying first image on background
        prev = "bg"
        for i in range(num_images):
            start_t = i * image_dur
            end_t = (i + 1) * image_dur
            out_label = f"bg_img{i}"
            filter_parts.append(
                f"[{prev}][img{i}]overlay=0:0:enable='between(t,{start_t:.2f},{end_t:.2f})'[{out_label}]"
            )
            prev = out_label

        canvas_with_images = prev
    else:
        canvas_with_images = "bg"

    # === STEP 3: Process avatar with chroma key ===
    # Remove green screen background, scale to bottom portion
    filter_parts.append(
        f"[{avatar_input}:v]"
        f"chromakey=0x008000:0.25:0.08,"  # Green screen removal (HeyGen #008000)
        f"chromakey=0x00FF00:0.25:0.08,"  # Also try pure green just in case
        f"scale={WIDTH}:{AVATAR_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{AVATAR_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black@0,"
        f"setsar=1"
        f"[avatar_clean]"
    )

    # Overlay avatar on canvas
    filter_parts.append(
        f"[{canvas_with_images}][avatar_clean]overlay=0:{AVATAR_TOP}:shortest=1[with_avatar]"
    )

    # === STEP 4: Add gradient overlay between image and avatar zones ===
    # Creates a smooth blend at the overlap zone
    filter_parts.append(
        f"color=c=black:s={WIDTH}x200:d={duration}:r={FPS},"
        f"format=yuva420p,"
        f"geq=lum='0':a='if(lt(Y,100),255*Y/100,255*(200-Y)/100)'"
        f"[gradient]"
    )
    filter_parts.append(
        f"[with_avatar][gradient]overlay=0:{AVATAR_TOP - 100}[with_grad]"
    )

    # === STEP 5: Title banner (first 5 seconds) ===
    current = "with_grad"
    if title:
        # Wrap title text for display
        safe_title = title.replace("'", "'\\''").replace(":", "\\:")
        if len(safe_title) > 50:
            safe_title = safe_title[:47] + "..."

        # Semi-transparent banner background
        filter_parts.append(
            f"[{current}]drawbox=x=0:y=40:w={WIDTH}:h=100:"
            f"color=black@0.6:t=fill:"
            f"enable='between(t,0,{TITLE_BANNER_DURATION})'[with_banner]"
        )
        filter_parts.append(
            f"[with_banner]drawtext="
            f"text='{safe_title}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=36:fontcolor=white:"
            f"x=(w-text_w)/2:y=65:"
            f"enable='between(t,0,{TITLE_BANNER_DURATION})'"
            f"[with_title]"
        )
        current = "with_title"

    # === STEP 6: Subtitles ===
    subtitle_chunks = generate_subtitle_chunks(script, duration, words_per_chunk=4)
    for i, chunk in enumerate(subtitle_chunks):
        safe_text = chunk['text'].replace("'", "'\\''").replace(":", "\\:").replace("%", "%%")
        if len(safe_text) > 60:
            safe_text = safe_text[:57] + "..."

        out_label = f"sub{i}"
        # Subtitle background box
        filter_parts.append(
            f"[{current}]drawtext="
            f"text='{safe_text}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=42:fontcolor=white:"
            f"borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y={SUBTITLE_Y}:"
            f"enable='between(t,{chunk['start']:.3f},{chunk['end']:.3f})'"
            f"[{out_label}]"
        )
        current = out_label

    # Final output label
    final_video = current

    # Build the full filter_complex string
    filter_complex = ";\n".join(filter_parts)

    # === Build command ===
    cmd = ['ffmpeg', '-y']
    cmd.extend(inputs)
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', f'[{final_video}]',
        '-map', f'{avatar_input}:a?',  # Audio from avatar video
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-r', str(FPS),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-t', str(duration),
        output_path
    ])

    return cmd


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint (pinged by UptimeRobot)."""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True, timeout=5)
        version = result.stdout.split('\n')[0] if result.stdout else 'unknown'
        return jsonify({
            'status': 'healthy',
            'ffmpeg': version,
            'api_version': 'v2'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/compose', methods=['POST'])
def compose():
    """
    Main composition endpoint.

    Expected JSON body:
    {
        "avatar_video_url": "https://...",
        "images": ["https://...", ...],
        "duration": 35,
        "script": "Full narration text...",
        "title": "Story Title"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'error': 'No JSON body'}), 400

        avatar_url = data.get('avatar_video_url', '')
        images = data.get('images', [])
        duration = float(data.get('duration', 35))
        script = data.get('script', '')
        title = data.get('title', '')

        if not avatar_url:
            return jsonify({'status': 'error', 'error': 'avatar_video_url is required'}), 400

        job_id = str(uuid.uuid4())[:8]
        job_temp = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_temp, exist_ok=True)

        logger.info(f"[{job_id}] Starting composition: {len(images)} images, {duration}s")

        # Download avatar video
        avatar_path = os.path.join(job_temp, 'avatar.mp4')
        download_file(avatar_url, avatar_path)

        # Probe avatar for actual duration
        try:
            probe = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', avatar_path
            ], capture_output=True, text=True, timeout=15)
            probe_data = json.loads(probe.stdout)
            actual_duration = float(probe_data.get('format', {}).get('duration', duration))
            if actual_duration > 0:
                duration = actual_duration
                logger.info(f"[{job_id}] Avatar actual duration: {duration:.1f}s")
        except Exception as e:
            logger.warning(f"[{job_id}] Could not probe avatar duration: {e}")

        # Download images
        image_paths = []
        for i, img_url in enumerate(images[:6]):  # Max 6 images
            try:
                ext = '.jpg'
                if '.png' in img_url.lower():
                    ext = '.png'
                img_path = os.path.join(job_temp, f'img_{i}{ext}')
                download_file(img_url, img_path)
                image_paths.append(img_path)
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to download image {i}: {e}")

        # If no images downloaded, create a dark placeholder
        if not image_paths:
            logger.warning(f"[{job_id}] No images available, creating placeholder")
            placeholder = os.path.join(job_temp, 'placeholder.png')
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', f'color=c=#1a1a2e:s={WIDTH}x{IMAGE_HEIGHT}:d=1',
                '-frames:v', '1', placeholder
            ], capture_output=True, timeout=10)
            image_paths = [placeholder]

        # Build and run FFmpeg command
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
        cmd = build_ffmpeg_command(
            avatar_path, image_paths, duration, title, script, output_path
        )

        logger.info(f"[{job_id}] Running FFmpeg...")
        logger.info(f"[{job_id}] Command length: {len(' '.join(cmd))} chars")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            logger.error(f"[{job_id}] FFmpeg STDERR: {result.stderr[-2000:]}")
            return jsonify({
                'status': 'error',
                'error': f'FFmpeg failed (code {result.returncode})',
                'stderr': result.stderr[-500:]
            }), 500

        if not os.path.exists(output_path):
            return jsonify({
                'status': 'error',
                'error': 'Output file not created'
            }), 500

        file_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] Composition complete: {file_size} bytes")

        # Clean up temp files
        try:
            import shutil
            shutil.rmtree(job_temp, ignore_errors=True)
        except:
            pass

        # Build video URL
        base_url = request.host_url.rstrip('/')
        video_url = f"{base_url}/output/{job_id}.mp4"

        return jsonify({
            'status': 'success',
            'video_url': video_url,
            'job_id': job_id,
            'duration': round(duration, 1),
            'file_size': file_size
        })

    except Exception as e:
        logger.error(f"Composition error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    """Serve rendered video files."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
