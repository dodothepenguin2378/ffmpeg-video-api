import os
import uuid
import subprocess
import requests
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

WORK_DIR = "/tmp/ffmpeg_work"
OUTPUT_DIR = "/tmp/ffmpeg_output"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-cleanup old files (older than 1 hour)
def cleanup_old_files():
    while True:
        time.sleep(600)  # every 10 min
        now = time.time()
        for d in [WORK_DIR, OUTPUT_DIR]:
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                try:
                    if now - os.path.getmtime(fp) > 3600:
                        os.remove(fp)
                except:
                    pass

threading.Thread(target=cleanup_old_files, daemon=True).start()


def download_file(url, path):
    """Download a file from URL to local path."""
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return path


@app.route('/health', methods=['GET'])
def health():
    """Health check + FFmpeg version."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        version = result.stdout.split('\n')[0]
        return jsonify({"status": "ok", "ffmpeg": version})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/compose', methods=['POST'])
def compose_video():
    """
    Compose a reel video:
    - Black background 1080x1920
    - Images in top half (cycling)
    - Avatar video in bottom half (chroma key green screen removal)
    - Audio from avatar video
    - Subtitles overlay
    
    POST JSON:
    {
        "avatar_video_url": "https://...",
        "images": ["https://img1.jpg", "https://img2.jpg"],
        "duration": 30,
        "script": "Full text for subtitles",
        "title": "Breaking News Title"
    }
    """
    try:
        data = request.json
        avatar_url = data['avatar_video_url']
        images = data.get('images', [])
        duration = float(data.get('duration', 30))
        script = data.get('script', '')
        title = data.get('title', '')
        
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(WORK_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Download avatar video
        avatar_path = os.path.join(job_dir, 'avatar.mp4')
        download_file(avatar_url, avatar_path)
        
        # Get actual avatar video duration using ffprobe
        probe_result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', avatar_path],
            capture_output=True, text=True
        )
        probe_data = json.loads(probe_result.stdout)
        actual_duration = float(probe_data.get('format', {}).get('duration', duration))
        duration = actual_duration
        
        # Download images
        image_paths = []
        for i, img_url in enumerate(images[:6]):  # max 6 images
            try:
                img_path = os.path.join(job_dir, f'img_{i}.jpg')
                download_file(img_url, img_path)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Failed to download image {i}: {e}")
        
        # Build FFmpeg command
        output_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
        
        # Build the filter complex
        inputs = []
        filter_parts = []
        
        # Input 0: black background
        inputs.extend(['-f', 'lavfi', '-i', f'color=c=black:s=1080x1920:d={duration}:r=30'])
        
        # Input 1: avatar video
        inputs.extend(['-i', avatar_path])
        
        # Input 2+: images
        for img_path in image_paths:
            inputs.extend(['-loop', '1', '-t', str(duration), '-i', img_path])
        
        # --- FILTER COMPLEX ---
        filters = []
        
        # Process avatar: chroma key + scale to bottom half
        filters.append(
            f'[1:v]colorkey=0x00FF00:0.3:0.15,scale=1080:960:force_original_aspect_ratio=decrease,'
            f'pad=1080:960:(ow-iw)/2:(oh-ih)/2:color=black[avatar]'
        )
        
        if image_paths:
            # Calculate timing for each image
            img_duration = duration / len(image_paths)
            
            # Scale each image to fill top half
            for i in range(len(image_paths)):
                input_idx = i + 2
                filters.append(
                    f'[{input_idx}:v]scale=1080:960:force_original_aspect_ratio=increase,'
                    f'crop=1080:960,setpts=PTS-STARTPTS[img{i}]'
                )
            
            # Overlay images with timing using enable
            # Start with black background
            current = '0:v'
            for i in range(len(image_paths)):
                start_t = i * img_duration
                end_t = (i + 1) * img_duration
                out_label = f'bg{i}'
                filters.append(
                    f'[{current}][img{i}]overlay=0:0:'
                    f"enable='between(t,{start_t:.2f},{end_t:.2f})'[{out_label}]"
                )
                current = out_label
            
            # Overlay avatar on bottom half
            filters.append(
                f'[{current}][avatar]overlay=0:960:shortest=1[composed]'
            )
        else:
            # No images — just avatar on black background
            filters.append(
                '[0:v][avatar]overlay=0:960:shortest=1[composed]'
            )
        
        # Add title banner (first 5 seconds)
        if title:
            safe_title = title.replace("'", "\u2019").replace(":", " -").replace("\\", "")
            filters.append(
                f"[composed]drawtext=text='{safe_title}':"
                f"fontsize=42:fontcolor=white:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                f"x=(w-text_w)/2:y=50:"
                f"box=1:boxcolor=red@0.7:boxborderw=15:"
                f"enable='between(t,0,5)'[titled]"
            )
            current_out = 'titled'
        else:
            current_out = 'composed'
        
        # Add subtitles from script
        if script:
            words = script.split()
            words_per_chunk = 4
            chunks = []
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                chunk_text = ' '.join(chunk_words).upper()
                start = (i / len(words)) * duration
                end = (min(i + words_per_chunk, len(words)) / len(words)) * duration
                chunks.append((chunk_text, start, end))
            
            for i, (text, start, end) in enumerate(chunks):
                safe_text = text.replace("'", "\u2019").replace(":", " -").replace("\\", "").replace('"', '')
                in_label = current_out
                out_label = f'sub{i}'
                filters.append(
                    f"[{in_label}]drawtext=text='{safe_text}':"
                    f"fontsize=56:fontcolor=white:"
                    f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                    f"x=(w-text_w)/2:y=1750:"
                    f"box=1:boxcolor=black@0.6:boxborderw=10:"
                    f"borderw=2:bordercolor=black:"
                    f"enable='between(t,{start:.3f},{end:.3f})'[{out_label}]"
                )
                current_out = out_label
        
        filter_complex = ';'.join(filters)
        
        # Build full command
        cmd = ['ffmpeg', '-y']
        cmd.extend(inputs)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', f'[{current_out}]',
            '-map', '1:a?',  # audio from avatar video
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-shortest',
            output_path
        ])
        
        print(f"FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600  # 10 min timeout
        )
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            # Try simplified version without subtitles
            return jsonify({
                "status": "error",
                "error": f"FFmpeg failed: {result.stderr[-500:]}",
                "job_id": job_id
            }), 500
        
        # Get output file size
        file_size = os.path.getsize(output_path)
        
        # Return the video URL
        base_url = request.host_url.rstrip('/')
        video_url = f"{base_url}/output/{job_id}.mp4"
        
        return jsonify({
            "status": "completed",
            "video_url": video_url,
            "job_id": job_id,
            "duration": duration,
            "file_size": file_size
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    """Serve the rendered video file."""
    return send_from_directory(OUTPUT_DIR, filename, mimetype='video/mp4')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
