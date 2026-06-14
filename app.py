"""
Hopfield Media Retrieval — Flask backend.
All encoding, storage, and retrieval logic runs in Python.
The HTML template has ZERO JavaScript — pure form submissions.
"""

import os
import io
import json
import base64
import tempfile
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, send_file,
    redirect, url_for, flash
)

# Optional heavy imports — graceful fallback
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
except ImportError:
    librosa = None


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

PATTERN_SIDE = 64
PATTERN_SIZE = PATTERN_SIDE * PATTERN_SIDE   # 4096
BETA = 8.0
MAX_ITER = 300

IMAGE_EXTS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tiff'}
AUDIO_EXTS = {'wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'}
VIDEO_EXTS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
PDF_EXTS   = {'pdf'}


# ═══════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = 'hopfield-media-retrieval-demo'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB


# ═══════════════════════════════════════════════════════════════
#  MODERN HOPFIELD NETWORK  (Dense Associative Memory)
# ═══════════════════════════════════════════════════════════════

class ModernHopfield:
    """
    Softmax-attention retrieval.
    Capacity scales exponentially with pattern size —
    far beyond the classical 0.138·N limit.
    """

    def __init__(self, N, beta=8.0):
        self.N = N
        self.beta = beta
        self.patterns = []

    def store(self, pattern):
        self.patterns.append(np.array(pattern, dtype=np.float64))

    def retrieve(self, query, max_iter=300):
        K = len(self.patterns)
        if K == 0:
            return np.array(query, dtype=np.float64)

        state = np.array(query, dtype=np.float64)
        Xi = np.array(self.patterns)            # K × N

        for _ in range(max_iter):
            sim = self.beta * (Xi @ state) / self.N   # K
            sim -= sim.max()                           # stable softmax
            w = np.exp(sim)
            w /= w.sum()
            h = Xi.T @ w                               # N
            new_state = np.where(h >= 0, 1.0, -1.0)
            if np.array_equal(new_state, state):
                break
            state = new_state

        return state

    def find_closest(self, pattern):
        pattern = np.array(pattern, dtype=np.float64)
        best_idx, best_ov = -1, -1.0
        for i, p in enumerate(self.patterns):
            ov = float(np.mean(p == pattern))
            if ov > best_ov:
                best_ov = ov
                best_idx = i
        return best_idx, best_ov


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def downsample(arr, target):
    """Linearly interpolate 1-D array to exactly `target` elements."""
    if len(arr) == target:
        return np.asarray(arr, dtype=np.float64)
    idx = np.linspace(0, len(arr) - 1, target)
    return np.interp(idx, np.arange(len(arr)), arr)


def binarize(arr):
    """Normalise to [0,1], threshold at median → ±1."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.ones(len(arr))
    norm = (arr - mn) / (mx - mn)
    med = np.median(norm)
    return np.where(norm >= med, 1.0, -1.0)


def detect_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in IMAGE_EXTS: return 'image'
    if ext in AUDIO_EXTS: return 'audio'
    if ext in VIDEO_EXTS: return 'video'
    if ext in PDF_EXTS:   return 'pdf'
    return 'unknown'


# ═══════════════════════════════════════════════════════════════
#  ENCODERS — each returns a ±1 array of length side²
# ═══════════════════════════════════════════════════════════════

def encode_image(file_bytes, side):
    img = Image.open(io.BytesIO(file_bytes)).convert('L')
    img = img.resize((side, side), Image.Resampling.LANCZOS)
    return binarize(np.array(img).flatten())


def encode_audio(file_bytes, side):
    if librosa is None:
        raise RuntimeError("librosa is not installed — cannot encode audio")
    # Write to temp file so librosa can decode it
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        y, _ = librosa.load(tmp, sr=22050, duration=10.0)
        arr = downsample(y, side * side)
        return binarize(arr)
    finally:
        os.unlink(tmp)


def encode_video(file_bytes, side):
    if cv2 is None:
        raise RuntimeError("opencv is not installed — cannot encode video")
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot read video frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (side, side))
        return binarize(gray.flatten())
    finally:
        os.unlink(tmp)


def encode_pdf(file_bytes, side):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed — cannot encode PDF")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
    img = img.resize((side, side), Image.Resampling.LANCZOS)
    doc.close()
    return binarize(np.array(img).flatten())


def encode_file(filename, file_bytes, side):
    ftype = detect_type(filename)
    if ftype == 'image': return encode_image(file_bytes, side), ftype
    if ftype == 'audio': return encode_audio(file_bytes, side), ftype
    if ftype == 'video': return encode_video(file_bytes, side), ftype
    if ftype == 'pdf':   return encode_pdf(file_bytes, side), ftype
    raise ValueError(f"Unsupported file type: {filename}")


# ═══════════════════════════════════════════════════════════════
#  IN-MEMORY STORE
# ═══════════════════════════════════════════════════════════════

class Store:
    def __init__(self):
        self.entries = []
        self.network = ModernHopfield(PATTERN_SIZE, BETA)

    def reset(self):
        self.entries = []
        self.network = ModernHopfield(PATTERN_SIZE, BETA)

    def add(self, name, ftype, pattern, original_bytes):
        b64 = base64.b64encode(original_bytes).decode()
        self.entries.append({
            'name': name,
            'type': ftype,
            'pattern': pattern.tolist(),
            'original_b64': b64,
        })
        self.network.store(pattern)


store = Store()
last_result = None   # holds retrieval result for download


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html',
                           entries=store.entries,
                           result=last_result,
                           pattern_side=PATTERN_SIDE)


@app.route('/config', methods=['POST'])
def config_route():
    global PATTERN_SIDE, PATTERN_SIZE
    try:
        side = int(request.form.get('side', 64))
        side = max(8, min(256, side))
    except ValueError:
        side = 64
    PATTERN_SIDE = side
    PATTERN_SIZE = side * side
    store.reset()
    flash(f"Pattern size set to {side}×{side} = {PATTERN_SIZE}. Store cleared.", 'ok')
    return redirect(url_for('index'))


@app.route('/store', methods=['POST'])
def store_route():
    files = request.files.getlist('files')
    if not files:
        flash("No files selected.", 'err')
        return redirect(url_for('index'))

    for f in files:
        if not f.filename:
            continue
        try:
            data = f.read()
            pattern, ftype = encode_file(f.filename, data, PATTERN_SIDE)
            store.add(f.filename, ftype, pattern, data)
            flash(f"✓ Stored: {f.filename}  [{ftype}]", 'ok')
        except Exception as e:
            flash(f"✗ {f.filename}: {e}", 'err')

    return redirect(url_for('index'))


@app.route('/clear', methods=['POST'])
def clear_route():
    global last_result
    store.reset()
    last_result = None
    flash("Store cleared.", 'ok')
    return redirect(url_for('index'))


@app.route('/download_memory')
def download_memory():
    if not store.entries:
        flash("Nothing to download — store is empty.", 'err')
        return redirect(url_for('index'))

    payload = {
        'version': 1,
        'pattern_size': PATTERN_SIZE,
        'pattern_side': PATTERN_SIDE,
        'beta': BETA,
        'entries': store.entries,
    }
    data = json.dumps(payload).encode()
    return send_file(
        io.BytesIO(data),
        mimetype='application/json',
        as_attachment=True,
        download_name='hopfield_memory.json'
    )


@app.route('/retrieve', methods=['POST'])
def retrieve_route():
    global last_result

    mem_file = request.files.get('memory')
    qry_file = request.files.get('query')

    if not mem_file or not mem_file.filename:
        flash("Upload a memory file (.json).", 'err')
        return redirect(url_for('index'))
    if not qry_file or not qry_file.filename:
        flash("Upload a query file.", 'err')
        return redirect(url_for('index'))

    try:
        # ── Load memory ──
        mem = json.loads(mem_file.read())
        entries = mem['entries']
        if not entries:
            flash("Memory file has no entries.", 'err')
            return redirect(url_for('index'))

        N    = mem['pattern_size']
        side = mem.get('pattern_side', int(round(N ** 0.5)))
        beta = mem.get('beta', BETA)

        # ── Rebuild network ──
        net = ModernHopfield(N, beta)
        for e in entries:
            net.store(np.array(e['pattern']))

        # ── Encode query using memory's pattern side ──
        qdata = qry_file.read()
        qpattern, _ = encode_file(qry_file.filename, qdata, side)

        # ── Retrieve ──
        retrieved = net.retrieve(qpattern, MAX_ITER)
        idx, overlap = net.find_closest(retrieved)

        if idx >= 0:
            match = entries[idx]
            last_result = {
                'name': match['name'],
                'type': match['type'],
                'overlap': f"{overlap * 100:.1f}%",
                'original_b64': match['original_b64'],
            }
            flash(f"Match: {match['name']}  ({overlap*100:.1f}% overlap)", 'ok')
        else:
            last_result = None
            flash("No match found.", 'err')

    except Exception as e:
        flash(f"Retrieval error: {e}", 'err')
        last_result = None

    return redirect(url_for('index'))


@app.route('/download_retrieved')
def download_retrieved():
    if not last_result:
        flash("No result to download.", 'err')
        return redirect(url_for('index'))
    data = base64.b64decode(last_result['original_b64'])
    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name='retrieved_' + last_result['name']
    )


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)