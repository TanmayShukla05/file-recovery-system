# Hopfield Media Retrieval

Store images, audio, video, and PDFs as binary patterns in a
Modern Hopfield Network. Then upload a damaged/noisy version
and retrieve the original file back.

All logic is Python (Flask). The webpage has **zero JavaScript**.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Deploy on Render (free)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) and sign up with GitHub.
3. Click **New → Web Service**.
4. Connect your GitHub repo.
5. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free
6. Click **Deploy**. Your site is live at the URL Render gives you.

## How to use

### Store
1. Set the pattern side length (default 64 → 4096).
2. Select files (image, audio, video, PDF).
3. Click **Store**.
4. Click **Download Memory File** to save `hopfield_memory.json`.

### Retrieve
1. Upload the **memory file** (.json).
2. Upload a **damaged or noisy version** of a stored file.
3. Click **Retrieve**.
4. Click **Download Retrieved File** to get the original back.

## Supported file types

| Type  | Extensions                     | How it's encoded                  |
|-------|--------------------------------|-----------------------------------|
| Image | png, jpg, jpeg, bmp, gif, webp | Grayscale → resize → binarize     |
| Audio | wav, mp3, ogg, flac, aac, m4a  | Waveform → downsample → binarize  |
| Video | mp4, avi, mov, mkv, webm       | Middle frame → resize → binarize  |
| PDF   | pdf                            | Render page 1 → resize → binarize |

## Network

Modern Hopfield (Dense Associative Memory). Capacity far exceeds
the classical 0.138·N limit. Retrieval uses softmax attention.