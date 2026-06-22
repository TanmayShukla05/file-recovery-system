# File Recovery System

Website: https://file-recovery-system.onrender.com/

This project uses a Modern Hopfield Network to store files and recover them
if they get damaged, accidentally edited, or partially lost.

## How to use

1. Upload the original files and click **Store**.
   A `hopfield_memory.npz` file will download automatically — save it.
2. Later, if a file becomes corrupted or altered, come back to the site.
3. Upload `hopfield_memory.npz` and one or more damaged files, then click **Retrieve**.
4. Each matched file gets its own download link so you can recover several at once.

## Supported file types

- Images: PNG, JPG, JPEG, BMP, GIF, WEBP
- Python code: PY
- Jupyter Notebooks: IPYNB
- Data files: HDF5, H5, PKL
- Documents: PDF

## How it works

### Encoding
Files are converted to ±1 binary vectors.
- **Images / PDFs** — resized to a square greyscale thumbnail, then median-thresholded.
- **Code / Notebooks / Pickle** — raw bytes are read and resampled to the target length.
- **HDF5** — all numeric datasets are extracted, flattened, and resampled.

### Automatic pattern-size selection
The network sets the vector length automatically so that the load factor
α = (number of files) / (pattern size) stays ≤ 0.12.
This keeps the network in a regime where it can reliably recall every stored
pattern without cross-talk. Uploading more files causes the network to grow
the pattern size and re-encode everything at higher resolution.

### Memory file format (.npz)
Patterns are bit-packed with `np.packbits` (1 bit per entry vs 8–64 for naive
storage) and then compressed with `np.savez_compressed` (zlib).
The archive contains:

| Array | Contents |
|---|---|
| `patterns_packed` | bit-packed ±1 matrix, shape `(K, ⌈N/8⌉)` |
| `names` | original filenames |
| `types` | file-type strings |
| `originals_b64` | base-64 encoded raw file bytes |
| `meta` | side, N, beta, K |

### Retrieval
The damaged file is encoded into a ±1 vector and fed into the Modern Hopfield
update rule (up to 300 steps). The converged state is compared against every
stored pattern using Hamming overlap, and the best match is returned.
Multiple query files can be submitted at once; each gets its own result.
