# File Recovery System

Website: https://file-recovery-system.onrender.com/

This is a project that uses a Hopfield network to store files and get them back if they get damaged.

## How to use

1. Set the pattern size (default is 64x64).
2. Upload some files and click Store.
3. Click "Download Memory File" to save a file with all the patterns.
4. Later, upload that memory file and a damaged version of one of the files.
5. Click Retrieve and it will find the closest match. You can download the original file back.

## Supported file types

- Images: png, jpg, jpeg, bmp, gif, webp
- Python code: py
- Jupyter Notebooks: ipynb
- Data files: hdf5, h5, pkl
- Documents: pdf

## How it works

It turns files into simple lists of 1s and -1s. For code and pickle files, it reads the raw bytes. For HDF5 files, it pulls out the numbers stored inside. It stores these lists in a Modern Hopfield Network. When you upload a damaged file, the network tries to fix the pattern and finds the closest match. Then it gives you the original file back.
