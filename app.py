import os
import io
import json
import base64
import math
import tempfile
import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, send_file,
                   redirect, url_for, flash, session)

try:
    import fitz
except ImportError:
    fitz = None

try:
    import h5py
except ImportError:
    h5py = None


# ------------------------------------------------------------------
# FLASK SETUP
# ------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ------------------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------------------
ALPHA_MAX   = 0.12   # Modern Hopfield recommended load factor
SIDE_MIN    =  8
SIDE_MAX    = 256
BETA        =  8.0

my_stored_patterns = []   # list of np.ndarray, shape (N,), dtype int8 (+1/-1)
my_stored_files    = []   # list of dicts: name, type, original_b64
last_matches       = []   # list of result dicts (supports multi-retrieve)


# ------------------------------------------------------------------
# CAPACITY HELPERS
# ------------------------------------------------------------------

def optimal_side(num_files: int) -> int:
    """
    Return the smallest side length s such that
    alpha = num_files / (s*s) <= ALPHA_MAX,
    clamped to [SIDE_MIN, SIDE_MAX].
    """
    if num_files <= 0:
        return SIDE_MIN
    # s >= sqrt(K / alpha_max)
    s = math.ceil(math.sqrt(num_files / ALPHA_MAX))
    return max(SIDE_MIN, min(SIDE_MAX, s))


def max_files_for_side(side: int) -> int:
    """Upper bound of files for a given side (floor(alpha_max * N))."""
    return max(1, int(ALPHA_MAX * side * side))


# ------------------------------------------------------------------
# ENCODING HELPERS
# ------------------------------------------------------------------

def downsample_list(arr, target_size):
    if len(arr) == target_size:
        return list(arr)
    new_arr = []
    n = len(arr)
    for i in range(target_size):
        pos = (i / (target_size - 1)) * (n - 1)
        lo  = int(pos)
        hi  = min(lo + 1, n - 1)
        frac = pos - lo
        new_arr.append(arr[lo] * (1 - frac) + arr[hi] * frac)
    return new_arr


def make_binary(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.ones(len(arr), dtype=np.int8)
    norm = (arr - mn) / (mx - mn)
    median = float(np.median(norm))
    return np.where(norm >= median, np.int8(1), np.int8(-1))


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext in ('png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'):
        return 'image'
    if ext == 'py':
        return 'python'
    if ext == 'ipynb':
        return 'notebook'
    if ext in ('hdf5', 'h5'):
        return 'hdf5'
    if ext == 'pkl':
        return 'pickle'
    if ext == 'pdf':
        return 'pdf'
    return 'unknown'


# ------------------------------------------------------------------
# ENCODERS
# ------------------------------------------------------------------

def encode_image(file_bytes, side):
    img = Image.open(io.BytesIO(file_bytes)).convert('L')
    img = img.resize((side, side), Image.LANCZOS)
    return make_binary(list(img.getdata()))


def encode_pdf(file_bytes, side):
    if fitz is None:
        raise RuntimeError("PDF library (PyMuPDF) not installed on server.")
    doc  = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img  = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
    img  = img.resize((side, side), Image.LANCZOS)
    doc.close()
    return make_binary(list(img.getdata()))


def encode_raw_bytes(file_bytes, side):
    arr = list(file_bytes)
    if not arr:
        raise ValueError("File is empty.")
    data = downsample_list(arr, side * side)
    return make_binary(data)


def encode_hdf5(file_bytes, side):
    if h5py is None:
        raise RuntimeError("HDF5 library (h5py) not installed on server.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    tmp.write(file_bytes)
    tmp.close()
    try:
        numbers = []
        def grab(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                numbers.extend(obj[()].flatten().tolist())
        with h5py.File(tmp.name, 'r') as f:
            f.visititems(grab)
        if not numbers:
            raise ValueError("No numeric data found in HDF5 file.")
        data = downsample_list(numbers, side * side)
        return make_binary(data)
    finally:
        os.unlink(tmp.name)


def encode_file(filename, file_bytes, side):
    ftype = get_file_type(filename)
    if ftype == 'image':
        return encode_image(file_bytes, side), ftype
    if ftype in ('python', 'notebook', 'pickle'):
        return encode_raw_bytes(file_bytes, side), ftype
    if ftype == 'hdf5':
        return encode_hdf5(file_bytes, side), ftype
    if ftype == 'pdf':
        return encode_pdf(file_bytes, side), ftype
    raise ValueError(f"Unsupported file type: {ftype}")


# ------------------------------------------------------------------
# MEMORY FILE  (.npz, compressed, bit-packed)
# ------------------------------------------------------------------

def build_npz_bytes(patterns: list, files: list, side: int, beta: float) -> bytes:
    """
    Pack everything into a compressed .npz blob and return raw bytes.

    Layout inside the archive
    --------------------------
    patterns_packed   : uint8 array, shape (K, ceil(N/8))  — np.packbits rows
    names             : object array of str
    types             : object array of str
    originals_b64     : object array of str  (base-64 encoded raw file bytes)
    meta              : structured array with side, N, beta, K
    """
    if not patterns:
        raise ValueError("Nothing stored.")

    K = len(patterns)
    N = side * side

    # Stack into matrix and pack bits  (+1 → 1,  -1 → 0)
    mat = np.stack(patterns, axis=0).astype(np.int8)   # (K, N)
    bits = ((mat + 1) // 2).astype(np.uint8)           # 0/1
    packed = np.packbits(bits, axis=1)                  # (K, ceil(N/8))

    names   = np.array([f['name']         for f in files], dtype=object)
    types   = np.array([f['type']         for f in files], dtype=object)
    orig64  = np.array([f['original_b64'] for f in files], dtype=object)
    meta    = np.array([(side, N, beta, K)],
                       dtype=[('side','i4'),('N','i4'),('beta','f8'),('K','i4')])

    buf = io.BytesIO()
    np.savez_compressed(buf,
                        patterns_packed=packed,
                        names=names,
                        types=types,
                        originals_b64=orig64,
                        meta=meta)
    return buf.getvalue()


def load_npz_bytes(raw: bytes):
    """
    Load a memory .npz blob produced by build_npz_bytes.
    Returns (patterns, entries, side, beta)
      patterns : list of np.ndarray int8 (+1/-1), length K
      entries  : list of dicts  name/type/original_b64
    """
    buf = io.BytesIO(raw)
    data = np.load(buf, allow_pickle=True)

    meta   = data['meta'][0]
    side   = int(meta['side'])
    N      = int(meta['N'])
    beta   = float(meta['beta'])
    K      = int(meta['K'])

    packed  = data['patterns_packed']           # (K, ceil(N/8))
    bits    = np.unpackbits(packed, axis=1)     # (K, ceil(N/8)*8)
    bits    = bits[:, :N]                       # trim padding
    mat     = (bits.astype(np.int8) * 2) - 1   # 0/1 → -1/+1

    patterns = [mat[i] for i in range(K)]

    names  = data['names']
    types  = data['types']
    orig64 = data['originals_b64']

    entries = [
        {'name': str(names[i]), 'type': str(types[i]),
         'original_b64': str(orig64[i])}
        for i in range(K)
    ]

    return patterns, entries, side, beta


# ------------------------------------------------------------------
# HOPFIELD NETWORK LOGIC
# ------------------------------------------------------------------

def hopfield_retrieve(query_pattern: np.ndarray,
                      stored_patterns: list,
                      beta: float,
                      max_steps: int = 300) -> np.ndarray:
    """Modern (dense) Hopfield update — vectorised."""
    K = len(stored_patterns)
    if K == 0:
        return query_pattern.copy()

    mat   = np.stack(stored_patterns, axis=0).astype(float)  # (K, N)
    state = query_pattern.astype(float).copy()               # (N,)

    for _ in range(max_steps):
        sims   = beta * (mat @ state) / len(state)           # (K,)
        sims  -= sims.max()
        w      = np.exp(sims)
        w     /= w.sum()
        h      = mat.T @ w                                   # (N,)
        new_state = np.where(h >= 0, 1.0, -1.0)
        if np.array_equal(new_state, state):
            break
        state = new_state

    return state.astype(np.int8)


def find_closest_matches(retrieved_patterns: list,
                         stored_patterns: list,
                         entries: list,
                         N: int) -> list:
    """
    For each retrieved pattern find the stored pattern with highest overlap.
    Returns list of (entry_index, overlap_float).
    """
    mat = np.stack(stored_patterns, axis=0).astype(np.int8)  # (K, N)
    results = []
    for rp in retrieved_patterns:
        overlaps = (mat == rp.reshape(1, -1)).sum(axis=1) / N
        idx      = int(overlaps.argmax())
        results.append((idx, float(overlaps[idx])))
    return results


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------

@app.route('/')
def index():
    side = optimal_side(len(my_stored_files))
    cap  = max_files_for_side(side)
    return render_template('index.html',
                           entries=my_stored_files,
                           results=last_matches,
                           current_side=side,
                           capacity=cap,
                           alpha_max=ALPHA_MAX)


@app.route('/store', methods=['POST'])
def store_files():
    global my_stored_patterns, my_stored_files

    files = request.files.getlist('files')
    if not files or all(not f.filename for f in files):
        flash("No files selected.")
        return redirect('/')

    newly_added = 0
    for f in files:
        if not f.filename:
            continue
        try:
            data = f.read()
            # Recalculate side for the new total
            future_count = len(my_stored_files) + 1
            side = optimal_side(future_count)

            # If side changed, re-encode all existing patterns at new side
            if my_stored_files and side != optimal_side(len(my_stored_files)):
                _reencode_all(side)

            pattern, ftype = encode_file(f.filename, data, side)
            my_stored_patterns.append(pattern)
            my_stored_files.append({
                'name': f.filename,
                'type': ftype,
                'original_b64': base64.b64encode(data).decode()
            })
            newly_added += 1
            flash(f"Stored: {f.filename} ({ftype})")
        except Exception as e:
            flash(f"Error with {f.filename}: {e}")

    if not my_stored_files:
        return redirect('/')

    # Auto-download memory file
    try:
        side = optimal_side(len(my_stored_files))
        npz_bytes = build_npz_bytes(my_stored_patterns, my_stored_files,
                                    side, BETA)
        return send_file(
            io.BytesIO(npz_bytes),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='hopfield_memory.npz'
        )
    except Exception as e:
        flash(f"Memory file generation failed: {e}")
        return redirect('/')


def _reencode_all(new_side: int):
    """Re-encode every stored file at a new side length (in-place)."""
    global my_stored_patterns
    new_patterns = []
    for i, f in enumerate(my_stored_files):
        raw = base64.b64decode(f['original_b64'])
        pattern, _ = encode_file(f['name'], raw, new_side)
        new_patterns.append(pattern)
    my_stored_patterns = new_patterns


@app.route('/clear', methods=['POST'])
def clear_store():
    global my_stored_patterns, my_stored_files, last_matches
    my_stored_patterns = []
    my_stored_files    = []
    last_matches       = []
    flash("Store cleared.")
    return redirect('/')


@app.route('/download_memory')
def download_memory():
    if not my_stored_files:
        flash("Nothing stored yet.")
        return redirect('/')
    side = optimal_side(len(my_stored_files))
    npz_bytes = build_npz_bytes(my_stored_patterns, my_stored_files,
                                side, BETA)
    return send_file(
        io.BytesIO(npz_bytes),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='hopfield_memory.npz'
    )


@app.route('/retrieve', methods=['POST'])
def retrieve_file():
    global last_matches

    mem_file  = request.files.get('memory')
    qry_files = request.files.getlist('query')

    if not mem_file or not mem_file.filename:
        flash("Upload a memory file (.npz).")
        return redirect('/')
    if not qry_files or all(not f.filename for f in qry_files):
        flash("Upload at least one query file.")
        return redirect('/')

    try:
        stored_patterns, entries, side, beta = load_npz_bytes(mem_file.read())
        N = side * side
    except Exception as e:
        flash(f"Could not load memory file: {e}")
        return redirect('/')

    query_patterns = []
    query_names    = []
    for qf in qry_files:
        if not qf.filename:
            continue
        try:
            qdata   = qf.read()
            qpat, _ = encode_file(qf.filename, qdata, side)
            query_patterns.append(qpat)
            query_names.append(qf.filename)
        except Exception as e:
            flash(f"Could not encode {qf.filename}: {e}")

    if not query_patterns:
        return redirect('/')

    retrieved_patterns = [
        hopfield_retrieve(qp, stored_patterns, beta)
        for qp in query_patterns
    ]

    match_info = find_closest_matches(retrieved_patterns, stored_patterns,
                                      entries, N)

    last_matches = []
    for qi, (idx, overlap) in enumerate(match_info):
        match = entries[idx]
        last_matches.append({
            'query_name':   query_names[qi],
            'name':         match['name'],
            'type':         match['type'],
            'overlap':      f"{overlap * 100:.1f}%",
            'original_b64': match['original_b64'],
            'dl_index':     qi          # index for /download_retrieved/<n>
        })
        flash(f"Query '{query_names[qi]}' → matched '{match['name']}' "
              f"({overlap*100:.1f}% overlap)")

    return redirect('/')


@app.route('/download_retrieved/<int:idx>')
def download_retrieved(idx: int):
    if idx < 0 or idx >= len(last_matches):
        flash("Result not found.")
        return redirect('/')
    match = last_matches[idx]
    data  = base64.b64decode(match['original_b64'])
    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name='retrieved_' + match['name']
    )


if __name__ == '__main__':
    app.run(debug=True)
