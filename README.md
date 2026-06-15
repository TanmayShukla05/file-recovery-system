# File Recovery System

Website: https://file-recovery-system.onrender.com/

This is a project that uses a Hopfield network to store files and get them back if they get damaged.

## How to use

1. Set the pattern size (default is 64x64).
2. Upload some files (images, audio, video, pdf) and click Store.
3. Click "Download Memory File" to save a file with all the patterns.
4. Later, upload that memory file and a damaged version of one of the files.
5. Click Retrieve and it will find the closest match. You can download the original file back.

## Supported file types

- Images: png, jpg, jpeg, bmp, gif, webp
- Audio: wav, mp3, ogg, flac, aac, m4a
- Video: mp4, avi, mov, mkv, webm
- PDF: pdf

## How it works

It turns files into simple lists of 1s and -1s. It stores these lists in a Modern Hopfield Network. When you upload a damaged file, it also turns that into 1s and -1s. The network updates the damaged list until it settles on the closest stored memory. Then it gives you the original file back.
