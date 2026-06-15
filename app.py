import os
import io
import json
import base64
import math
import tempfile
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, redirect, url_for, flash

# Try to import the extra libraries. If they are not there, we just skip those file types.
try:
    import fitz # for pdf
except:
    fitz = None

try:
    import cv2 # for video
except:
    cv2 = None

try:
    import librosa # for audio
except:
    librosa = None


# ------------------------------------------------------------------
# FLASK SETUP
# ------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'supersecretkey'
# Max upload size 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


# ------------------------------------------------------------------
# GLOBAL VARIABLES (where we store everything in memory)
# ------------------------------------------------------------------
PATTERN_SIDE = 64
PATTERN_SIZE = 64 * 64 # 4096
BETA = 8.0

# This list will hold the actual patterns (lists of 1 and -1)
my_stored_patterns = []

# This list will hold the file info and the original file data so we can download it later
my_stored_files = []

# To hold the last result so the user can download it
last_match = None


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def downsample_list(arr, target_size):
    """Make an array a specific length by averaging nearby points."""
    if len(arr) == target_size:
        return arr
    
    new_arr = []
    for i in range(target_size):
        # find where we are in the old array
        pos = (i / (target_size - 1)) * (len(arr) - 1)
        low_idx = int(pos)
        high_idx = min(low_idx + 1, len(arr) - 1)
        fraction = pos - low_idx
        
        # blend between the two nearest points
        val = arr[low_idx] * (1 - fraction) + arr[high_idx] * fraction
        new_arr.append(val)
        
    return new_arr


def make_binary(arr):
    """Turn an array of numbers into an array of only 1 and -1."""
    arr = np.array(arr, dtype=float)
    
    # 1. normalize to 0 to 1
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 0.00001:
        return [1] * len(arr) # everything is the same
    
    norm_arr = (arr - mn) / (mx - mn)
    
    # 2. find the middle (median)
    sorted_arr = sorted(norm_arr)
    middle = sorted_arr[len(sorted_arr) // 2]
    
    # 3. if above middle, 1. if below middle, -1.
    result = []
    for val in norm_arr:
        if val >= middle:
            result.append(1)
        else:
            result.append(-1)
            
    return result


def get_file_type(filename):
    """Guess the file type from the name."""
    ext = filename.split('.')[-1].lower()
    
    if ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp']:
        return 'image'
    elif ext in ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a']:
        return 'audio'
    elif ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
        return 'video'
    elif ext in ['pdf']:
        return 'pdf'
    else:
        return 'unknown'


# ------------------------------------------------------------------
# ENCODERS (turn files into lists of 1 and -1)
# ------------------------------------------------------------------

def encode_image(file_bytes, side):
    img = Image.open(io.BytesIO(file_bytes)).convert('L') # L means grayscale
    img = img.resize((side, side))
    pixels = list(img.getdata()) # get pixel values as a list
    return make_binary(pixels)


def encode_audio(file_bytes, side):
    if librosa is None:
        raise Exception("Audio library not installed on server.")
    
    # librosa needs a file path, so we write to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file_bytes)
    tmp.close()
    
    try:
        y, sr = librosa.load(tmp.name, sr=22050, duration=10.0)
        data = downsample_list(list(y), side * side)
        return make_binary(data)
    finally:
        os.unlink(tmp.name) # delete temp file


def encode_video(file_bytes, side):
    if cv2 is None:
        raise Exception("Video library not installed on server.")
        
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.write(file_bytes)
    tmp.close()
    
    try:
        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # jump to the middle of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Could not read video frame")
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (side, side))
        pixels = list(gray.flatten())
        return make_binary(pixels)
    finally:
        os.unlink(tmp.name)


def encode_pdf(file_bytes, side):
    if fitz is None:
        raise Exception("PDF library not installed on server.")
        
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
    img = img.resize((side, side))
    doc.close()
    
    pixels = list(img.getdata())
    return make_binary(pixels)


def encode_file(filename, file_bytes, side):
    """Figure out what kind of file it is and encode it."""
    ftype = get_file_type(filename)
    
    if ftype == 'image':
        return encode_image(file_bytes, side), ftype
    elif ftype == 'audio':
        return encode_audio(file_bytes, side), ftype
    elif ftype == 'video':
        return encode_video(file_bytes, side), ftype
    elif ftype == 'pdf':
        return encode_pdf(file_bytes, side), ftype
    else:
        raise Exception("I don't know how to read this file type.")


# ------------------------------------------------------------------
# HOPFIELD NETWORK LOGIC (Written simply with loops)
# ------------------------------------------------------------------

def hopfield_retrieve(query_pattern):
    """
    This does the Modern Hopfield retrieval using basic loops.
    It looks at all stored patterns, compares them to the query,
    and tries to converge on the closest match.
    """
    global my_stored_patterns, BETA, PATTERN_SIZE
    
    K = len(my_stored_patterns)
    if K == 0:
        return query_pattern
        
    N = PATTERN_SIZE
    state = list(query_pattern) # make a copy
    
    # Iterate up to 300 times
    for step in range(300):
        changed = False
        
        # 1. Calculate similarities (softmax attention)
        similarities = [0.0] * K
        for mu in range(K):
            dot_product = 0
            for i in range(N):
                dot_product += my_stored_patterns[mu][i] * state[i]
            similarities[mu] = BETA * dot_product / N
            
        # Stable softmax (subtract max so math.exp doesn't crash)
        max_sim = max(similarities)
        weights = [0.0] * K
        sum_weights = 0.0
        for mu in range(K):
            weights[mu] = math.exp(similarities[mu] - max_sim)
            sum_weights += weights[mu]
            
        for mu in range(K):
            weights[mu] = weights[mu] / sum_weights
            
        # 2. Update the state based on weights
        for i in range(N):
            h = 0.0
            for mu in range(K):
                h += weights[mu] * my_stored_patterns[mu][i]
                
            if h >= 0:
                new_val = 1
            else:
                new_val = -1
                
            if new_val != state[i]:
                changed = True
                state[i] = new_val
                
        # If nothing changed, we found a stable memory!
        if not changed:
            break
            
    return state


def find_closest_match(retrieved_pattern):
    """Compare the retrieved pattern to all stored patterns to find the best overlap."""
    global my_stored_patterns, my_stored_files, PATTERN_SIZE
    
    best_index = -1
    best_overlap = -1.0
    
    for i in range(len(my_stored_patterns)):
        matches = 0
        for j in range(PATTERN_SIZE):
            if my_stored_patterns[i][j] == retrieved_pattern[j]:
                matches += 1
                
        overlap = matches / PATTERN_SIZE
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = i
            
    return best_index, best_overlap


# ------------------------------------------------------------------
# WEBSITE ROUTES
# ------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', 
                           entries=my_stored_files, 
                           result=last_match, 
                           pattern_side=PATTERN_SIDE)


@app.route('/config', methods=['POST'])
def set_config():
    global PATTERN_SIDE, PATTERN_SIZE, my_stored_patterns, my_stored_files
    
    try:
        side = int(request.form.get('side', 64))
        if side < 8: side = 8
        if side > 256: side = 256
    except:
        side = 64
        
    PATTERN_SIDE = side
    PATTERN_SIZE = side * side
    
    # Reset everything because size changed
    my_stored_patterns = []
    my_stored_files = []
    flash(f"Pattern size set to {PATTERN_SIDE}x{PATTERN_SIDE} = {PATTERN_SIZE}. Store cleared.")
    return redirect('/')


@app.route('/store', methods=['POST'])
def store_files():
    global my_stored_patterns, my_stored_files
    
    files = request.files.getlist('files')
    if not files:
        flash("No files selected.")
        return redirect('/')
        
    for f in files:
        if not f.filename:
            continue
        try:
            data = f.read()
            pattern, ftype = encode_file(f.filename, data, PATTERN_SIDE)
            
            # Save the pattern
            my_stored_patterns.append(pattern)
            
            # Save the original file so we can return it later
            b64_data = base64.b64encode(data).decode()
            my_stored_files.append({
                'name': f.filename,
                'type': ftype,
                'original_b64': b64_data
            })
            
            flash(f"Stored: {f.filename} ({ftype})")
        except Exception as e:
            flash(f"Error with {f.filename}: {e}")
            
    return redirect('/')


@app.route('/clear', methods=['POST'])
def clear_store():
    global my_stored_patterns, my_stored_files, last_match
    my_stored_patterns = []
    my_stored_files = []
    last_match = None
    flash("Store cleared.")
    return redirect('/')


@app.route('/download_memory')
def download_memory():
    global my_stored_patterns, my_stored_files, PATTERN_SIZE, PATTERN_SIDE, BETA
    
    if not my_stored_files:
        flash("Nothing to download.")
        return redirect('/')
        
    # Build a big dictionary to save as JSON
    payload = {
        'pattern_size': PATTERN_SIZE,
        'pattern_side': PATTERN_SIDE,
        'beta': BETA,
        'entries': []
    }
    
    for i in range(len(my_stored_files)):
        payload['entries'].append({
            'name': my_stored_files[i]['name'],
            'type': my_stored_files[i]['type'],
            'pattern': my_stored_patterns[i],
            'original_b64': my_stored_files[i]['original_b64']
        })
        
    data = json.dumps(payload).encode()
    return send_file(
        io.BytesIO(data),
        mimetype='application/json',
        as_attachment=True,
        download_name='hopfield_memory.json'
    )


@app.route('/retrieve', methods=['POST'])
def retrieve_file():
    global last_match
    
    mem_file = request.files.get('memory')
    qry_file = request.files.get('query')
    
    if not mem_file or not mem_file.filename:
        flash("Upload a memory file (.json).")
        return redirect('/')
    if not qry_file or not qry_file.filename:
        flash("Upload a query file.")
        return redirect('/')
        
    try:
        # 1. Load memory
        mem = json.loads(mem_file.read())
        entries = mem['entries']
        if not entries:
            flash("Memory file is empty.")
            return redirect('/')
            
        N = mem['pattern_size']
        side = mem.get('pattern_side', int(math.sqrt(N)))
        beta = mem.get('beta', BETA)
        
        # Put patterns back into global list temporarily
        my_stored_patterns = [e['pattern'] for e in entries]
        
        # 2. Encode the damaged query file
        qdata = qry_file.read()
        qpattern, _ = encode_file(qry_file.filename, qdata, side)
        
        # 3. Run Hopfield retrieval
        retrieved = hopfield_retrieve(qpattern)
        
        # 4. Find which stored file it is closest to
        idx, overlap = find_closest_match(retrieved)
        
        if idx >= 0:
            match = entries[idx]
            last_match = {
                'name': match['name'],
                'type': match['type'],
                'overlap': f"{overlap * 100:.1f}%",
                'original_b64': match['original_b64']
            }
            flash(f"Found match: {match['name']} ({overlap*100:.1f}% overlap)")
        else:
            last_match = None
            flash("No match found.")
            
    except Exception as e:
        flash(f"Error during retrieval: {e}")
        last_match = None
        
    return redirect('/')


@app.route('/download_retrieved')
def download_retrieved():
    global last_match
    if not last_match:
        flash("No result to download.")
        return redirect('/')
        
    data = base64.b64decode(last_match['original_b64'])
    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name='retrieved_' + last_match['name']
    )
