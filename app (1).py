import streamlit as st
import os
import cv2
import torch
import numpy as np
import kornia.feature as KF
from PIL import Image
import tempfile
import zipfile
import io
import json
import math

st.set_page_config(page_title="Circle Mosaic Stitcher", layout="wide", page_icon="🔬")

# Use available CPU threads more efficiently for LoFTR
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass

# ─── Styling (LIGHT MODE, blue accents) ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #ffffff; color: #1a1d27; }

    h1 { font-size: 1.6rem !important; font-weight: 600 !important;
         letter-spacing: -0.02em; color: #1a1d27 !important; }
    h2, h3 { font-weight: 500 !important; color: #2c2f3a !important; }

    .metric-box {
        background: #f4f6fb;
        border: 1px solid #dde3f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #1e6ad4;
    }

    .status-ok   { color: #1f9d55; font-weight: 500; }
    .status-warn { color: #b8860b; font-weight: 500; }
    .status-err  { color: #d23c3c; font-weight: 500; }

    .stButton > button {
        background: #1e6ad4;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
        transition: background 0.15s;
    }
    .stButton > button:hover { background: #2478e8; }
    .stButton > button:disabled { background: #9db8e0; color: #f0f0f0; }

    /* Force slider track/thumb to blue, regardless of theme */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #1e6ad4 !important;
        border-color: #1e6ad4 !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: #1e6ad4 !important;
    }

    div[data-testid="stExpander"] {
        background: #f8f9fc;
        border: 1px solid #e2e6f0;
        border-radius: 8px;
    }

    .param-card {
        background: #f4f6fb;
        border: 1px solid #dde3f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0 18px 0;
    }
    .param-card .param-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e6ad4;
        margin-bottom: 4px;
    }
    .param-help {
        font-size: 0.8rem;
        color: #5a5f70;
        line-height: 1.45;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── LoFTR cache ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_loftr(pretrained="outdoor"):
    device = torch.device("cpu")
    matcher = KF.LoFTR(pretrained=pretrained).to(device)
    matcher.eval()
    return matcher, device

# ─── Core functions (adapted from main.py) ───────────────────────────────────

def sharpen_for_matching(img_bgr):
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
    return cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0)

def prep_for_matching_bgr(img_bgr, device, max_size=1024):
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        img_resized = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
    else:
        img_resized = img_bgr.copy()
    sharpened = sharpen_for_matching(img_resized)
    h_r, w_r = sharpened.shape[:2]
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    t = torch.from_numpy(gray).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0).to(device)
    return img_resized, t, w/float(w_r), h/float(h_r)

def loftr_match(tA, tB, matcher):
    with torch.inference_mode():
        out = matcher({"image0": tA, "image1": tB})
    return out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy()

def estimate_affine_ransac(mk0, mk1, thresh=5.0):
    if mk0.shape[0] < 3:
        return None, None
    A, mask = cv2.estimateAffinePartial2D(mk0, mk1, method=cv2.RANSAC,
                                          ransacReprojThreshold=thresh, confidence=0.99)
    return A, mask

def load_image_any(path):
    """Load jpg/png/tif robustly."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        pil = Image.open(path)
        if pil.mode not in ('RGB', 'L'):
            pil = pil.convert('RGB')
        elif pil.mode == 'L':
            pil = pil.convert('RGB')
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)

def compress_image(img_bgr, scale=0.25):
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), cv2.INTER_AREA)

def check_corner_region(mosaic_np, y_pos, x_start, n_tiles, tile_w, tile_h,
                         row_idx, region_size=20, black_threshold=70, black_frac_thresh=0.8):
    h, w = mosaic_np.shape[:2]
    check_right = (row_idx % 2 == 0)
    y_bottom = min(y_pos + tile_h, h) - 1
    x_corner = (min(x_start + n_tiles*tile_w, w) - 1) if check_right else x_start
    half = region_size // 2
    y1, y2 = max(y_bottom-half,0), min(y_bottom+half+1,h)
    x1, x2 = max(x_corner-half,0), min(x_corner+half+1,w)
    region = mosaic_np[y1:y2, x1:x2]
    if region.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    frac = np.mean(gray < black_threshold)
    return frac >= black_frac_thresh, frac

def determine_row_counts(compressed_imgs, tile_size, canvas_size,
                          black_threshold=70, black_frac_thresh=0.8, region_size=20):
    canvas_w, canvas_h = canvas_size
    tile_w, tile_h = tile_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0,0,0))
    total = len(compressed_imgs)
    half_target = total / 2.0
    half_counts, img_index = [], 0
    min_tiles_first, prev_tiles = 0, 0

    for row_idx in range(canvas_h // tile_h):
        if img_index >= total:
            break
        current, valid_row, y_pos = 2, False, row_idx * tile_h

        while not valid_row:
            if current * tile_w > canvas_w:
                break
            x_start = max((canvas_w - current*tile_w)//2, 0)
            batch = compressed_imgs[img_index: img_index+current]
            if len(batch) < current:
                current = len(batch)
                batch = compressed_imgs[img_index: img_index+current]
                if current == 0:
                    break

            disp = batch[::-1] if row_idx%2==1 else batch
            tmp = canvas.copy()
            for j, img in enumerate(disp):
                pil = Image.fromarray(cv2.cvtColor(
                    cv2.resize(img, tile_size, cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                tmp.paste(pil, (x_start + j*tile_w, y_pos))

            mosaic_np = np.array(tmp.convert("RGB"))[:,:,::-1]
            corner_ok, frac = check_corner_region(mosaic_np, y_pos, x_start, current,
                                                   tile_w, tile_h, row_idx,
                                                   region_size, black_threshold, black_frac_thresh)

            if row_idx > 0:
                remaining = total - img_index
                is_last = remaining <= prev_tiles
                valid_count = is_last or (abs(current-prev_tiles)%2==0 and current>=2)
                valid_min = is_last or row_idx==(canvas_h//tile_h)-1 or current>=min_tiles_first
                valid_row = corner_ok and valid_count and valid_min
            else:
                valid_row = corner_ok

            if valid_row:
                canvas = tmp
                half_counts.append(current)
                if row_idx == 0:
                    min_tiles_first = current
                prev_tiles = current
                img_index += len(batch)
                break
            else:
                current += 1
                if img_index + current > total:
                    current = total - img_index
                    if current <= 0:
                        break

        if img_index >= half_target:
            break

    bottom = half_counts[::-1]
    row_counts = half_counts + (bottom[1:] if len(half_counts)%2==1 else bottom)
    return row_counts

def build_layout_preview(compressed_imgs, row_counts, tile_size=(128,128)):
    tile_w, tile_h = tile_size
    canvas_w = max(row_counts) * tile_w
    canvas_h = len(row_counts) * tile_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    idx = 0
    for row_idx, count in enumerate(row_counts):
        batch = compressed_imgs[idx:idx+count]
        idx += count
        if row_idx%2==1:
            batch = batch[::-1]
        row_w = count * tile_w
        x_start = (canvas_w - row_w) // 2
        for j, img in enumerate(batch):
            thumb = cv2.resize(img, tile_size, cv2.INTER_AREA)
            x = x_start + j*tile_w
            canvas[row_idx*tile_h:(row_idx+1)*tile_h, x:x+tile_w] = thumb
    return canvas

def stitch_row(imgs_bgr, matcher, device, row_name="row",
               max_match_size=1024, min_inliers=10,
               default_overlap_ratio=0.1, match_downscale=0.25,
               max_overlap_pct=0.3, progress_cb=None):
    if len(imgs_bgr) == 0:
        return None
    if len(imgs_bgr) == 1:
        return imgs_bgr[0].copy()

    Ws = [im.shape[1] for im in imgs_bgr]
    target_h = imgs_bgr[0].shape[0]
    pano = imgs_bgr[0].copy()
    pano_w = Ws[0]
    prev_w = Ws[0]
    overlaps = []

    for i in range(1, len(imgs_bgr)):
        prev_img = imgs_bgr[i-1]
        next_img = imgs_bgr[i]
        next_w = Ws[i]

        strip_w = int(prev_w * 0.3)
        prev_strip = prev_img[:, -strip_w:]
        next_strip = next_img[:, :strip_w]

        def ds(s, sc):
            if sc == 1.0: return s
            h,w = s.shape[:2]
            return cv2.resize(s, (int(w*sc), int(h*sc)), cv2.INTER_AREA)

        pm = ds(prev_strip, match_downscale)
        nm = ds(next_strip, match_downscale)
        _, pt, spx, spy = prep_for_matching_bgr(pm, device, max_match_size)
        _, nt, snx, sny = prep_for_matching_bgr(nm, device, max_match_size)
        fsx = spx/match_downscale; fsy = spy/match_downscale
        fnx = snx/match_downscale; fny = sny/match_downscale

        mk0, mk1 = loftr_match(pt, nt, matcher)

        dx, overlap = None, None
        if mk0.shape[0] >= 3:
            m0 = mk0.copy(); m1 = mk1.copy()
            m0[:,0] *= fsx; m0[:,1] *= fsy
            m1[:,0] *= fnx; m1[:,1] *= fny
            m0[:,0] += (prev_w - strip_w)
            A, mask = estimate_affine_ransac(m0, m1)
            if A is not None and mask is not None:
                inliers = int(mask.ravel().sum())
                if inliers >= min_inliers:
                    tx, ty = float(A[0,2]), float(A[1,2])
                    dx_est = -tx
                    if abs(ty) <= 0.2*prev_img.shape[0] and 0 < dx_est <= prev_w:
                        ov = prev_w - dx_est
                        if ov <= max_overlap_pct * prev_w:
                            dx, overlap = dx_est, ov

        if overlap is None or overlap <= 0:
            overlap = (sum(overlaps)/len(overlaps)) if overlaps else prev_w*default_overlap_ratio
            overlap = max(0, min(prev_w*max_overlap_pct, overlap))
            dx = prev_w - overlap

        overlaps.append(overlap)
        new_w = pano_w + next_w - int(overlap)
        border = max(0, next_w - int(overlap))
        pano = cv2.copyMakeBorder(pano, 0,0,0,border, cv2.BORDER_CONSTANT, value=0)
        pano[:, pano_w-int(overlap):pano_w-int(overlap)+next_w] = next_img
        pano_w = new_w
        prev_w = next_w

        if progress_cb:
            progress_cb(i, len(imgs_bgr)-1, overlap)

    return pano[:target_h, :]

def stack_rows_loftr(row_panos, matcher, device,
                     max_match_size=1024, min_inliers=10,
                     default_overlap_ratio=0.1, match_downscale=0.25,
                     progress_cb=None):
    valid = [p for p in row_panos if p is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].copy()

    mosaic = valid[0].copy()
    mosaic_h, mosaic_w = mosaic.shape[:2]
    overlaps = []

    for i in range(1, len(valid)):
        next_row = valid[i]
        next_h, next_w = next_row.shape[:2]
        single_h = next_h

        top_strip = mosaic[-single_h:, :]
        top_cx, bot_cx = top_strip.shape[1]//2, next_row.shape[1]//2
        tcrop = int(top_strip.shape[1]*0.10)
        bcrop = int(next_row.shape[1]*0.10)
        tc = top_strip[:, top_cx-tcrop:top_cx+tcrop]
        bc = next_row[:,  bot_cx-bcrop:bot_cx+bcrop]

        def ds(s, sc):
            if sc==1.0: return s
            h,w=s.shape[:2]
            return cv2.resize(s,(int(w*sc),int(h*sc)),cv2.INTER_AREA)

        tm = ds(tc, match_downscale)
        bm = ds(bc, match_downscale)
        _, tt, stx, sty = prep_for_matching_bgr(tm, device, max_match_size)
        _, bt, sbx, sby = prep_for_matching_bgr(bm, device, max_match_size)
        fstx=stx/match_downscale; fsty=sty/match_downscale
        fsbx=sbx/match_downscale; fsby=sby/match_downscale

        mk0, mk1 = loftr_match(tt, bt, matcher)

        overlap = None
        if mk0.shape[0] >= 3:
            m0=mk0.copy(); m1=mk1.copy()
            m0[:,0]*=fstx; m0[:,1]*=fsty
            m1[:,0]*=fsbx; m1[:,1]*=fsby
            A, mask = estimate_affine_ransac(m0, m1)
            if A is not None and mask is not None:
                inliers = int(mask.ravel().sum())
                if inliers >= min_inliers:
                    ty, tx = float(A[1,2]), float(A[0,2])
                    ov = single_h - ty
                    ov = max(0, min(single_h, ov))
                    if abs(tx) < 0.3*tc.shape[1] and 0 < ov < single_h*0.5:
                        overlap = ov

        if overlap is None or overlap <= 0:
            overlap = (sum(overlaps)/len(overlaps)) if overlaps else single_h*default_overlap_ratio
            overlap = max(0, min(single_h*0.4, overlap))

        overlaps.append(overlap)
        new_h = mosaic_h + next_h - int(overlap)
        new_w = max(mosaic_w, next_w)
        new_mosaic = np.zeros((new_h, new_w, 3), dtype=mosaic.dtype)
        xoe = (new_w-mosaic_w)//2
        new_mosaic[:mosaic_h, xoe:xoe+mosaic_w] = mosaic
        y_start = mosaic_h - int(overlap)
        xon = (new_w-next_w)//2
        roi = new_mosaic[y_start:y_start+next_h, xon:xon+next_w]
        mask_next = np.any(next_row>0, axis=2, keepdims=True)
        new_mosaic[y_start:y_start+next_h, xon:xon+next_w] = np.where(mask_next, next_row, roi)
        mosaic = new_mosaic
        mosaic_h, mosaic_w = mosaic.shape[:2]

        if progress_cb:
            progress_cb(i, len(valid)-1)

    return mosaic

def img_to_bytes(img_bgr, fmt=".jpg", quality=90):
    if fmt == ".tif":
        try:
            import tifffile
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            tifffile.imwrite(buf, rgb, compression="lzw", bigtiff=True)
            return buf.getvalue(), "image/tiff"
        except ImportError:
            fmt = ".jpg"
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt==".jpg" else []
    _, buf = cv2.imencode(fmt, img_bgr, enc_params)
    mime = "image/jpeg" if fmt==".jpg" else "image/png"
    return buf.tobytes(), mime

def extract_zip_to_images(zip_bytes):
    """Extract a zip of tile images into a temp dir, return sorted (path, name) list."""
    tmpdir = tempfile.mkdtemp()
    valid_ext = {".jpg",".jpeg",".png",".tif",".tiff"}
    paths = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if not name or name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_ext:
                continue
            out_path = os.path.join(tmpdir, name)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            paths.append((out_path, name))
    paths.sort(key=lambda p: p[1])
    return paths

# ─── Grid-tile export with neighbor metadata ─────────────────────────────────

def export_grid_tiles(final_mosaic, tile_size=3000, fmt=".jpg", quality=90):
    """
    Slice final_mosaic into a grid of tile_size x tile_size chunks (edge tiles
    may be smaller). For each tile, embed a JSON metadata blob (mosaic
    position + neighbor filenames) directly into the image file, plus write a
    matching .json sidecar, plus a global manifest.json.

    Returns a dict: {"manifest": manifest_dict, "files": {filename: bytes}}
    """
    h, w = final_mosaic.shape[:2]
    n_cols = math.ceil(w / tile_size)
    n_rows = math.ceil(h / tile_size)

    def tile_name(r, c):
        return f"tile_r{r:03d}_c{c:03d}{fmt}"

    manifest = {
        "mosaic_width": int(w),
        "mosaic_height": int(h),
        "tile_size": int(tile_size),
        "rows": int(n_rows),
        "cols": int(n_cols),
        "format": fmt,
        "tiles": [],
    }

    files = {}

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * tile_size, min((r + 1) * tile_size, h)
            x0, x1 = c * tile_size, min((c + 1) * tile_size, w)
            tile_bgr = final_mosaic[y0:y1, x0:x1]

            neighbors = {
                "top":    tile_name(r - 1, c) if r > 0 else None,
                "bottom": tile_name(r + 1, c) if r < n_rows - 1 else None,
                "left":   tile_name(r, c - 1) if c > 0 else None,
                "right":  tile_name(r, c + 1) if c < n_cols - 1 else None,
            }

            name = tile_name(r, c)
            meta = {
                "filename": name,
                "row": r,
                "col": c,
                "x": int(x0),
                "y": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "mosaic_width": int(w),
                "mosaic_height": int(h),
                "grid_rows": int(n_rows),
                "grid_cols": int(n_cols),
                "neighbors": neighbors,
            }
            manifest["tiles"].append(meta)

            # Encode the tile with the metadata embedded in the file itself.
            rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            comment_bytes = json.dumps(meta).encode("utf-8")

            buf = io.BytesIO()
            if fmt == ".jpg":
                pil_img.save(buf, format="JPEG", quality=quality, comment=comment_bytes)
            elif fmt == ".png":
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                pnginfo.add_text("neighbors", json.dumps(meta))
                pil_img.save(buf, format="PNG", pnginfo=pnginfo)
            else:
                pil_img.save(buf, format="TIFF")
            files[name] = buf.getvalue()

            # Sidecar JSON, in case the embedded comment gets stripped later.
            sidecar_name = f"tile_r{r:03d}_c{c:03d}.json"
            files[sidecar_name] = json.dumps(meta, indent=2).encode("utf-8")

    files["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    return {"manifest": manifest, "files": files}

def build_tiles_zip(files_dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files_dict.items():
            zf.writestr(name, data)
    return buf.getvalue()

# ─── Session state init ───────────────────────────────────────────────────────
for k,v in [
    ("stage", "upload"),
    ("fullres_imgs", None), ("fullres_names", None),
    ("compressed_imgs", None),
    ("row_counts", None),
    ("layout_preview", None),
    ("row_panos", None),
    ("final_mosaic", None),
    ("busy_h", False),
    ("busy_v", False),
    ("tiles_zip", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

def param_card(title, help_text):
    st.markdown(f"""
    <div class="param-card">
        <div class="param-title">{title}</div>
        <div class="param-help">{help_text}</div>
    """, unsafe_allow_html=True)

def param_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🔬 Circle Mosaic Stitcher")
st.caption("VHX microscope tile assembly — upload tiles, detect layout, stitch with LoFTR")

# ─── Stage 1: Upload ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 1 · Upload tiles")

st.markdown(
    '<div class="param-help">Streamlit can\'t browse a folder directly from the file picker, '
    'but you can upload an entire folder at once by zipping it first and using the ZIP option below — '
    'or just multi-select all the files if you prefer.</div>',
    unsafe_allow_html=True
)

upload_mode = st.radio("Upload method", ["Multiple files", "ZIP of a folder"], horizontal=True)

loaded_files = []  # list of (path_or_None, name, bytes_or_None)

if upload_mode == "Multiple files":
    uploaded = st.file_uploader(
        "Upload all tile images (JPG, PNG, or TIF) — they will be sorted by filename",
        type=["jpg","jpeg","png","tif","tiff"],
        accept_multiple_files=True
    )
    if uploaded:
        uploaded_sorted = sorted(uploaded, key=lambda f: f.name)
        n = len(uploaded_sorted)
        st.metric("Tiles uploaded", n)

        if st.button(f"Load {n} tiles →"):
            with st.spinner("Loading images…"):
                tmpdir = tempfile.mkdtemp()
                fullres, names = [], []
                for uf in uploaded_sorted:
                    path = os.path.join(tmpdir, uf.name)
                    with open(path,"wb") as f:
                        f.write(uf.read())
                    img = load_image_any(path)
                    if img is not None:
                        fullres.append(img)
                        names.append(uf.name)
                compressed = [compress_image(im, 0.25) for im in fullres]

            st.session_state.fullres_imgs   = fullres
            st.session_state.fullres_names  = names
            st.session_state.compressed_imgs = compressed
            st.session_state.stage = "layout"
            st.success(f"Loaded {len(fullres)} images. First tile: {fullres[0].shape[1]}×{fullres[0].shape[0]}px")

else:
    zfile = st.file_uploader("Upload a ZIP file containing the tile folder", type=["zip"])
    if zfile:
        if st.button("Load tiles from ZIP →"):
            with st.spinner("Unzipping and loading images…"):
                paths = extract_zip_to_images(zfile.read())
                fullres, names = [], []
                for path, name in paths:
                    img = load_image_any(path)
                    if img is not None:
                        fullres.append(img)
                        names.append(name)
                compressed = [compress_image(im, 0.25) for im in fullres]

            if not fullres:
                st.error("No valid image files (jpg/png/tif) found inside the ZIP.")
            else:
                st.session_state.fullres_imgs   = fullres
                st.session_state.fullres_names  = names
                st.session_state.compressed_imgs = compressed
                st.session_state.stage = "layout"
                st.success(f"Loaded {len(fullres)} images. First tile: {fullres[0].shape[1]}×{fullres[0].shape[0]}px")

# ─── Stage 2: Layout detection ───────────────────────────────────────────────
if st.session_state.stage in ("layout","stitch","done") and st.session_state.compressed_imgs:
    st.markdown("---")
    st.markdown("## 2 · Layout detection")
    st.caption("Figures out how many tiles sit in each row of the circular mosaic by looking for "
               "black background at the edge of each candidate row.")

    compressed = st.session_state.compressed_imgs
    total = len(compressed)

    param_card("Layout detection parameters",
        "<b>Canvas width / Tile size</b> — set these to roughly match your real layout scale; "
        "bigger canvas width lets more tiles fit per row before the algorithm gives up. Usually leave at defaults.<br><br>"
        "<b>Black threshold</b> (pixel value 0–255) — how dark a pixel must be to count as 'background black'. "
        "<u>Lower it</u> if your background isn't pure black (e.g. dark grey) and rows are being cut off too early. "
        "<u>Raise it</u> if noise/shadows in tiles are being mistaken for background, causing rows to end too soon.<br><br>"
        "<b>Black fraction threshold</b> — what fraction of the checked corner region must be 'black' before a row is "
        "considered complete (i.e. the circle's edge has been reached). <u>Lower it</u> (e.g. 0.6) if rows are stopping "
        "one tile too early — it'll accept a row as 'done' with less black showing. <u>Raise it</u> (e.g. 0.9) if rows "
        "are running too long / overshooting into the next row's space, because it'll demand more solid black before stopping.<br><br>"
        "<b>Corner region size</b> — size in px of the patch sampled to check for black. Increase if detection looks noisy/jumpy; "
        "decrease for finer-grained row endings."
    )
    c1, c2 = st.columns(2)
    with c1:
        canvas_w = st.select_slider("Canvas width (px)", [1024,2048,4096,8192], value=4096)
        tile_size_px = st.select_slider("Tile size (px)", [64,128,256], value=128)
    with c2:
        black_thresh = st.slider("Black threshold", 30, 120, 70)
        black_frac = st.slider("Black fraction threshold", 0.5, 1.0, 0.8, 0.05)
    region_sz = st.slider("Corner region size", 10, 40, 20)
    param_card_end()

    if st.button("Detect row layout"):
        with st.spinner("Detecting circle layout…"):
            estimated_rows = (total // (canvas_w // tile_size_px)) + 4
            canvas_h_px = estimated_rows * tile_size_px
            row_counts = determine_row_counts(
                compressed,
                tile_size=(tile_size_px, tile_size_px),
                canvas_size=(canvas_w, canvas_h_px),
                black_threshold=black_thresh,
                black_frac_thresh=black_frac,
                region_size=region_sz
            )
            preview = build_layout_preview(compressed, row_counts,
                                           tile_size=(tile_size_px, tile_size_px))
            st.session_state.row_counts     = row_counts
            st.session_state.layout_preview = preview
            st.session_state.stage = "layout"

    if st.session_state.row_counts:
        rc = st.session_state.row_counts
        st.markdown(f"""
        <div class="metric-box">
        Rows detected: {len(rc)} &nbsp;|&nbsp;
        Total tiles: {sum(rc)} / {total} &nbsp;|&nbsp;
        Row counts: {rc}
        </div>""", unsafe_allow_html=True)

        if sum(rc) != total:
            st.warning(f"⚠️ Layout places {sum(rc)} tiles but {total} uploaded. Adjust parameters and re-detect.")
        else:
            st.markdown('<span class="status-ok">✓ All tiles accounted for</span>', unsafe_allow_html=True)

        if st.session_state.layout_preview is not None:
            preview_rgb = cv2.cvtColor(st.session_state.layout_preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="Layout preview (compressed thumbnails)", width='stretch')

        with st.expander("✏️ Override row counts manually"):
            rc_str = st.text_input("Row counts (comma-separated)", value=",".join(map(str,rc)))
            if st.button("Apply override"):
                try:
                    new_rc = [int(x.strip()) for x in rc_str.split(",")]
                    if sum(new_rc) != total:
                        st.error(f"Sum {sum(new_rc)} ≠ {total} tiles. Fix the counts.")
                    else:
                        st.session_state.row_counts = new_rc
                        preview = build_layout_preview(compressed, new_rc,
                                                       tile_size=(tile_size_px,tile_size_px))
                        st.session_state.layout_preview = preview
                        st.success("Row counts updated.")
                        st.rerun()
                except ValueError:
                    st.error("Invalid format — use comma-separated integers.")

# ─── Stage 3: Stitching ──────────────────────────────────────────────────────
if st.session_state.row_counts and st.session_state.stage in ("layout","stitch","done"):
    st.markdown("---")
    st.markdown("## 3 · Stitch")

    rc = st.session_state.row_counts
    fullres = st.session_state.fullres_imgs

    st.markdown("### Horizontal stitching (within each row)")
    param_card("Horizontal stitching parameters",
        "<b>Min inliers (H)</b> — minimum number of confidently-matched feature points required before trusting the "
        "computed overlap between two adjacent tiles. Raise it if rows look 'glitchy'/misaligned from bad matches; "
        "lower it if tiles are falling back to the default overlap too often (slow/inaccurate matches).<br><br>"
        "<b>Fallback overlap ratio (H)</b> — overlap (as a fraction of tile width) used when matching fails. Set this "
        "close to your microscope's known real overlap percentage.<br><br>"
        "<b>Max overlap cap (H)</b> — upper limit on how much overlap is allowed, to reject clearly-wrong matches.<br><br>"
        "<b>Match downscale (H)</b> — resolution tiles are shrunk to before feature matching. <u>Lower = faster but less "
        "precise alignment.</u> This is the biggest lever for speed.<br><br>"
        "<b>Max match size (H)</b> — hard cap on matching resolution in pixels. Lower this for a speed boost on large tiles."
    )
    c1, c2 = st.columns(2)
    with c1:
        h_min_inliers   = st.slider("Min inliers (H)", 5, 50, 10)
        h_overlap_ratio = st.slider("Fallback overlap ratio (H)", 0.05, 0.3, 0.1, 0.01)
        h_max_overlap   = st.slider("Max overlap cap (H) %", 10, 50, 30) / 100
    with c2:
        h_match_ds       = st.slider("Match downscale (H)", 0.1, 1.0, 0.2, 0.05,
                                      help="Lower = much faster matching, at some cost to alignment precision.")
        h_max_match_size = st.select_slider("Max match size (H)", [512,1024,2048], value=512,
                                      help="Lower = faster. 512 is usually plenty for alignment purposes.")
    param_card_end()

    run_rows_disabled = st.session_state.busy_h
    run_rows = st.button("▶ Stitch rows horizontally", disabled=run_rows_disabled, key="btn_h")

    if run_rows and not st.session_state.busy_h:
        st.session_state.busy_h = True
        matcher, device = get_loftr()
        row_panos = []
        img_index = 0
        prog = st.progress(0)
        status = st.empty()

        for row_idx, count in enumerate(rc):
            prog.progress(row_idx / len(rc))
            status.markdown(f"Stitching row {row_idx+1}/{len(rc)} ({count} tiles)…")

            batch = fullres[img_index:img_index+count]
            img_index += count
            if row_idx%2==1:
                batch = batch[::-1]

            pano = stitch_row(
                batch, matcher, device,
                row_name=f"row_{row_idx}",
                max_match_size=h_max_match_size,
                min_inliers=h_min_inliers,
                default_overlap_ratio=h_overlap_ratio,
                match_downscale=h_match_ds,
                max_overlap_pct=h_max_overlap
            )
            row_panos.append(pano)

        prog.progress(1.0)
        status.markdown('<span class="status-ok">✓ All rows stitched</span>', unsafe_allow_html=True)
        st.session_state.row_panos = row_panos
        st.session_state.stage = "stitch"
        st.session_state.busy_h = False
        st.rerun()

    if st.session_state.row_panos:
        st.markdown("### Row panoramas")
        row_panos = st.session_state.row_panos

        cols = st.columns(min(4, len(row_panos)))
        for i, pano in enumerate(row_panos):
            if pano is not None:
                thumb = cv2.resize(pano, (300, int(300*pano.shape[0]/pano.shape[1])), cv2.INTER_AREA)
                cols[i%4].image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
                                caption=f"Row {i} — {pano.shape[1]}×{pano.shape[0]}px",
                                width='stretch')

        st.markdown("### Vertical stitching (stacking rows)")
        param_card("Vertical stitching parameters",
            "<b>Min inliers (V)</b> — same idea as horizontal, but for matching a row's bottom edge to the next row's "
            "top edge. Raise if vertical seams look misaligned.<br><br>"
            "<b>Fallback overlap ratio (V)</b> — vertical overlap fraction used when matching fails.<br><br>"
            "<b>Match downscale (V) / Max match size (V)</b> — same speed/precision tradeoff as horizontal. Lower values "
            "make this step noticeably faster — this is usually the slowest part of the pipeline since each row is "
            "large, so lowering these two is the most effective way to speed it up."
        )
        c1, c2 = st.columns(2)
        with c1:
            v_min_inliers   = st.slider("Min inliers (V)", 5, 50, 10)
            v_overlap_ratio = st.slider("Fallback overlap ratio (V)", 0.05, 0.3, 0.1, 0.01)
        with c2:
            v_match_ds       = st.slider("Match downscale (V)", 0.1, 1.0, 0.2, 0.05)
            v_max_match_size = st.select_slider("Max match size (V)", [512,1024,2048], value=512)
        param_card_end()

        run_v_disabled = st.session_state.busy_v
        run_v = st.button("▶ Stack rows vertically", disabled=run_v_disabled, key="btn_v")

        if run_v and not st.session_state.busy_v:
            st.session_state.busy_v = True
            matcher, device = get_loftr()
            prog = st.progress(0)
            status = st.empty()

            def vcb(i, total):
                prog.progress(i/total)
                status.markdown(f"Aligning row {i}/{total}…")

            final = stack_rows_loftr(
                st.session_state.row_panos, matcher, device,
                max_match_size=v_max_match_size,
                min_inliers=v_min_inliers,
                default_overlap_ratio=v_overlap_ratio,
                match_downscale=v_match_ds,
                progress_cb=vcb
            )
            prog.progress(1.0)
            status.markdown('<span class="status-ok">✓ Mosaic complete</span>', unsafe_allow_html=True)
            st.session_state.final_mosaic = final
            st.session_state.stage = "done"
            st.session_state.busy_v = False
            st.rerun()

# ─── Stage 4: Result ─────────────────────────────────────────────────────────
if st.session_state.final_mosaic is not None:
    st.markdown("---")
    st.markdown("## 4 · Result")

    final = st.session_state.final_mosaic
    h, w = final.shape[:2]

    col1, col2, col3 = st.columns(3)
    col1.metric("Width", f"{w:,} px")
    col2.metric("Height", f"{h:,} px")
    col3.metric("Size", f"{w*h/1e6:.1f} Mpx")

    max_display = 1400
    scale = min(1.0, max_display/w)
    preview = cv2.resize(final, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
             caption="Final mosaic (scaled for display)", width='stretch')

    st.markdown("### ⬇️ Download")
    param_card("Output settings",
        "<b>Output format</b> — TIFF is lossless and best for archival/further analysis but produces large files; "
        "JPG is much smaller but lossy; PNG is lossless and smaller than TIFF but slower to write for very large images.<br><br>"
        "<b>JPEG quality</b> — only applies if JPG is selected; higher = better quality, larger file."
    )
    out_fmt = st.selectbox("Output format", [".tif (BigTIFF, lossless)", ".jpg (compressed)", ".png (lossless)"])
    out_fmt = ".tif" if "tif" in out_fmt else (".jpg" if "jpg" in out_fmt else ".png")
    jpg_quality = st.slider("JPEG quality", 70, 100, 90) if out_fmt==".jpg" else 90
    param_card_end()

    data, mime = img_to_bytes(final, out_fmt, jpg_quality)
    st.download_button(
        f"Download final mosaic ({out_fmt})",
        data, f"final_mosaic{out_fmt}", mime,
        width='stretch'
    )

    st.markdown("### 🧩 Export as tiled JPGs with neighbor metadata")
    param_card("Tile export settings",
        "Slices the final mosaic into a grid of square tiles (edge tiles may be smaller than the "
        "requested size, since they're clipped to the mosaic bounds). Each tile file has its "
        "position and neighbor filenames embedded directly in the image (JPEG comment / PNG text "
        "chunk), plus a matching <code>.json</code> sidecar, plus a global <code>manifest.json</code> "
        "listing every tile and its neighbors. Everything is bundled into one ZIP.<br><br>"
        "<b>Tile size</b> — width/height of each grid tile in pixels (default 3000)."
    )
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        grid_tile_size = st.number_input("Tile size (px)", min_value=256, max_value=10000, value=3000, step=100)
    with tc2:
        grid_fmt = st.selectbox("Tile format", [".jpg", ".png"], index=0)
    with tc3:
        grid_quality = st.slider("JPEG quality (tiles)", 70, 100, 90) if grid_fmt == ".jpg" else 90

    n_cols_preview = math.ceil(w / grid_tile_size)
    n_rows_preview = math.ceil(h / grid_tile_size)
    st.markdown(f"""
    <div class="metric-box">
    Grid: {n_rows_preview} rows × {n_cols_preview} cols &nbsp;=&nbsp; {n_rows_preview*n_cols_preview} tiles
    </div>""", unsafe_allow_html=True)
    param_card_end()

    if st.button("Build tiled export (ZIP)"):
        with st.spinner(f"Slicing into {n_rows_preview*n_cols_preview} tiles and embedding metadata…"):
            result = export_grid_tiles(final, tile_size=int(grid_tile_size),
                                        fmt=grid_fmt, quality=grid_quality)
            zip_bytes = build_tiles_zip(result["files"])
            st.session_state.tiles_zip = zip_bytes
        st.success(f"Built {len(result['manifest']['tiles'])} tiles + manifest.json")

    if st.session_state.tiles_zip is not None:
        st.download_button(
            "Download tiled export (ZIP)",
            st.session_state.tiles_zip,
            f"mosaic_tiles_{int(grid_tile_size)}px.zip",
            "application/zip",
            width='stretch'
        )

    if st.button("🔄 Start over"):
        for k in ["stage","fullres_imgs","fullres_names","compressed_imgs",
                  "row_counts","layout_preview","row_panos","final_mosaic","tiles_zip"]:
            st.session_state[k] = None
        st.session_state.stage = "upload"
        st.session_state.busy_h = False
        st.session_state.busy_v = False
        st.rerun()
