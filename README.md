# Circle Mosaic Stitcher

A Streamlit app for assembling VHX microscope tile sets into a single circular mosaic. Detects the row layout of a circular scan pattern, stitches tiles horizontally within each row using LoFTR feature matching, stacks the rows vertically, and exports the result — either as one large image or as a grid of tiles with neighbor metadata.

## Requirements

Listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | web app framework |
| `opencv-python-headless` | image loading, resizing, RANSAC affine estimation |
| `torch` | backend for the LoFTR matcher |
| `kornia` | provides the LoFTR feature-matching model |
| `Pillow` | image I/O, JPEG comment / PNG text-chunk metadata embedding |
| `numpy` | array operations |
| `tifffile` | optional BigTIFF export (falls back to JPG if missing) |

Python 3.9+ recommended. No GPU required — the app pins `torch` to CPU threads.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## App workflow

1. **Upload tiles** — either multi-select image files or upload a ZIP of a folder. Files are sorted by filename, so name your tiles so that sort order matches scan order.
2. **Layout detection** — algorithmically figures out how many tiles belong in each row of the circle by checking for black background at row edges. Adjustable thresholds; row counts can also be overridden manually.
3. **Stitch** — horizontal stitching within each row (LoFTR + RANSAC affine), then vertical stacking of rows into the final mosaic.
4. **Result** — preview the mosaic, download it as a single TIFF/JPG/PNG, or export it as a tiled grid with neighbor metadata (see below).

## Tiled export metadata format

The "Export as tiled JPGs with neighbor metadata" step slices the final mosaic into a grid of square tiles (default 3000×3000px, configurable). Edge tiles are clipped to the mosaic bounds rather than padded, so the last row/column of tiles may be smaller than `tile_size`.

Each tile's metadata is written in **three places**, so it survives even if one gets stripped:

### 1. Embedded in the image file itself
- **JPEG**: written to the standard JPEG comment segment (readable via `PIL.Image.open(path).info["comment"]`, or any tool that reads JPEG COM markers, e.g. `exiftool`).
- **PNG**: written as a PNG text chunk under the key `neighbors` (readable via `PIL.Image.open(path).info["neighbors"]`, or `exiftool`).

### 2. A per-tile JSON sidecar
For every `tile_r{row}_c{col}.jpg`, a matching `tile_r{row}_c{col}.json` is written with the same content.

### 3. A global `manifest.json`
Lists every tile in one place.

**Schema** (identical structure used in the embedded comment, the sidecar, and each entry of the manifest's `tiles` array):

```json
{
  "filename": "tile_r002_c004.jpg",
  "row": 2,
  "col": 4,
  "x": 12000,
  "y": 6000,
  "width": 3000,
  "height": 3000,
  "mosaic_width": 21453,
  "mosaic_height": 14892,
  "grid_rows": 5,
  "grid_cols": 8,
  "neighbors": {
    "top":    "tile_r001_c004.jpg",
    "bottom": "tile_r003_c004.jpg",
    "left":   "tile_r002_c003.jpg",
    "right":  "tile_r002_c005.jpg"
  }
}
```

Field reference:

| Field | Meaning |
|---|---|
| `filename` | this tile's own filename |
| `row`, `col` | zero-indexed grid position |
| `x`, `y` | top-left pixel offset of this tile within the full mosaic |
| `width`, `height` | actual tile dimensions in px (may be smaller than `tile_size` at edges) |
| `mosaic_width`, `mosaic_height` | full mosaic dimensions, for reassembly reference |
| `grid_rows`, `grid_cols` | total grid dimensions |
| `neighbors.top/bottom/left/right` | adjacent tile's filename, or `null` if this tile is on that edge of the grid |

The top-level `manifest.json` additionally includes `tile_size` (the requested slice size) and `format` (`.jpg` or `.png`).

All of this is bundled into a single `mosaic_tiles_{tile_size}px.zip` for download.

## Hosting on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Choose your repository, the branch (e.g. `main`), and the main file path (`app.py`).
5. Click **Deploy**.

Streamlit Cloud reads `requirements.txt` automatically and installs everything before starting the app. First deploy can take a few minutes since `torch` and `kornia` are large.

**Note:** the free tier has limited CPU/RAM. LoFTR matching on large tiles can be slow or hit memory limits on the free tier — fine for testing and small mosaics, but for large production runs consider a paid tier or self-hosting (e.g. a VM with more RAM, run via `streamlit run app.py --server.port 80`).
