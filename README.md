# Hopfield File Recovery System

This is a small project made using Flask and Hopfield Networks.

Website:

https://file-recovery-system.onrender.com/

The website can store files in a Hopfield Network and later retrieve the original file from a damaged or noisy version.

## Supported Files

* Images
* Audio files
* Videos
* PDF files

## How to Use

### Store Files

1. Open the website.
2. Choose the pattern size.
3. Upload one or more files.
4. Click **Store**.
5. Download the memory file.

### Retrieve Files

1. Upload the memory file.
2. Upload a damaged version of one of the stored files.
3. Click **Retrieve**.
4. Download the recovered file.

## Libraries Used

* Flask
* NumPy
* Pillow
* OpenCV
* Librosa
* PyMuPDF

## About the Project

The project uses a Modern Hopfield Network to store binary patterns generated from files. During retrieval, the uploaded query file is converted into a pattern and matched with the stored memories. The closest stored memory is then returned to the user.

## Try it Here

https://file-recovery-system.onrender.com/
