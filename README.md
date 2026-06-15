# Hopfield File Recovery System

This is a simple project made using Flask and Hopfield Networks.

Website Link:

https://file-recovery-system.onrender.com/

The project can store different types of files as patterns in a Hopfield Network and later try to recover the original file from a damaged version.

## Files Supported

* Images
* Audio files
* Videos
* PDF files

## How to Use

### Storing Files

1. Open the website.
2. Choose the pattern size.
3. Upload one or more files.
4. Click **Store**.
5. Download the generated memory file.

### Retrieving Files

1. Upload the memory file.
2. Upload a damaged/noisy version of a stored file.
3. Click **Retrieve**.
4. Download the recovered file.

## Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Idea Behind the Project

The project uses a Modern Hopfield Network to store patterns generated from files. When a damaged file is uploaded, the network finds the closest stored pattern and returns the corresponding original file.

## Made Using

* Python
* Flask
* NumPy
* OpenCV
* Pillow
* PyMuPDF
* Librosa
