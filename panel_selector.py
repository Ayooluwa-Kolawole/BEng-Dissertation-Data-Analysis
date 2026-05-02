"""
Panel Impregnation Analyser — Interactive ROI Selector
=======================================================
Opens each panel image in a window. Trace around the panel face with
a freehand outline by clicking and dragging. Press Confirm (or Enter)
to accept and move to the next image. At the end, analysis runs
automatically and results are saved alongside the images.

Usage
-----
    python panel_selector.py                        # analyses all images in current folder
    python panel_selector.py path/to/photos/        # analyses all images in given folder
    python panel_selector.py a.jpg b.jpg c.jpg      # specific files

Requirements (all standard on macOS with Python)
-----------
    pip install opencv-python-headless Pillow numpy matplotlib
"""

import sys
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── tuneable ────────────────────────────────────────────────────────────────
BRIGHTNESS_THRESH = 175      # V-channel cutoff: above = shiny, at/below = matte
MAX_DISPLAY_SIZE  = (1100, 800)  # max window size (px) — image is scaled to fit
# ────────────────────────────────────────────────────────────────────────────


def find_images(args):
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    paths = []
    if not args:
        folder = Path('.')
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    else:
        for a in args:
            p = Path(a)
            if p.is_dir():
                paths += sorted(x for x in p.iterdir() if x.suffix.lower() in exts)
            elif p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    return paths


def classify(img_bgr, polygon, thresh=BRIGHTNESS_THRESH):
    """Return (pct_matte, pct_shiny, panel_mask, shiny_mask, matte_mask).

    polygon: iterable of (x, y) points in original image coordinates.
    """
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v    = hsv[:, :, 2]
    hue  = hsv[:, :, 0]
    sat  = hsv[:, :, 1]

    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    mask_u8 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask_u8, [pts], 255)
    mask = mask_u8.astype(bool)

    # Exclude blue tape, white labels, dark flash/borders
    blue_tape   = (hue >= 95) & (hue <= 135) & (sat > 80)
    white_label = (v > 210) & (sat < 20)
    dark_border = v < 40
    panel_mask  = mask & ~blue_tape & ~white_label & ~dark_border

    shiny = (v > thresh) & panel_mask
    matte = (v <= thresh) & panel_mask
    total = panel_mask.sum()
    if total == 0:
        return 0.0, 0.0, panel_mask, shiny, matte
    return 100 * matte.sum() / total, 100 * shiny.sum() / total, panel_mask, shiny, matte


def save_overlay(img_bgr, panel_mask, shiny, matte, pct_m, pct_s, out_path):
    rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy().astype(float)
    overlay[shiny]        = [255, 140, 0]
    overlay[matte]        = [0, 160, 100]
    overlay[~panel_mask] *= 0.25
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(rgb);     ax1.set_title('Original', fontsize=11);     ax1.axis('off')
    ax2.imshow(overlay); ax2.set_title(f'Matte {pct_m:.1f}%   Shiny {pct_s:.1f}%', fontsize=11)
    ax2.axis('off')
    ax2.legend(handles=[Patch(color=[0, 160/255, 100/255], label=f'Matte {pct_m:.1f}%'),
                        Patch(color=[1, 140/255, 0],       label=f'Shiny {pct_s:.1f}%')],
               loc='lower right', fontsize=9)
    plt.suptitle(Path(out_path).stem.replace('_classified', ''), fontsize=10, color='dimgray')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()


def save_summary_chart(results, output_dir):
    names   = list(results.keys())
    mattes  = [results[n]['pct_matte'] for n in names]
    shinys  = [results[n]['pct_shiny'] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.3), 5))
    x = np.arange(len(names))
    ax.bar(x, mattes, color='#1D9E75', label='Matte (good impregnation)')
    ax.bar(x, shinys, bottom=mattes, color='#EF9F27', label='Shiny (poor impregnation)')

    for i, (m, s) in enumerate(zip(mattes, shinys)):
        ax.text(i, m / 2, f'{m:.1f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Percentage of panel face area (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Impregnation Quality — All Panels', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = Path(output_dir) / 'impregnation_summary.png'
    plt.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close()
    return out


# ── GUI ─────────────────────────────────────────────────────────────────────

class ROISelector:
    """Tkinter window for drawing a freehand outline on one image."""

    def __init__(self, root, img_bgr, title, initial_thresh=BRIGHTNESS_THRESH):
        self.root     = root
        self.img_bgr  = img_bgr
        self.title    = title
        self.result   = None   # (polygon, threshold) once confirmed
        self._points  = []     # display-coord points for the in-progress path
        self._path_id = None
        self._pending = None   # full-resolution polygon, or None

        root.title(f'Trace panel face — {title}')
        root.resizable(True, True)

        # ── scale image to fit display ──────────────────────────────────────
        ih, iw = img_bgr.shape[:2]
        mw, mh = MAX_DISPLAY_SIZE
        scale  = min(mw / iw, mh / ih, 1.0)
        self.scale = scale
        dw, dh = int(iw * scale), int(ih * scale)

        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(img_rgb).resize((dw, dh), Image.LANCZOS)
        self._display_rgb = np.asarray(pil_img)         # for live preview overlay
        self._tk_img      = ImageTk.PhotoImage(pil_img) # original (no overlay)
        self._preview_tk  = None                        # holds overlay PhotoImage

        # ── layout ──────────────────────────────────────────────────────────
        top = tk.Frame(root, bg='#1a1a1a')
        top.pack(fill='x', padx=0, pady=0)

        instr = tk.Label(
            top,
            text='Click and drag to trace the panel face outline. Redraw as many times as needed.',
            bg='#1a1a1a', fg='#cccccc', font=('Helvetica', 12), pady=8
        )
        instr.pack(side='left', padx=12)

        self.status_var = tk.StringVar(value='No selection yet')
        status_lbl = tk.Label(top, textvariable=self.status_var,
                              bg='#1a1a1a', fg='#EF9F27', font=('Helvetica', 11, 'bold'), pady=8)
        status_lbl.pack(side='right', padx=12)

        # ── slider row ──────────────────────────────────────────────────────
        slider_row = tk.Frame(root, bg='#1a1a1a')
        slider_row.pack(fill='x')

        tk.Label(slider_row, text='Shine threshold',
                 bg='#1a1a1a', fg='#cccccc',
                 font=('Helvetica', 11)).pack(side='left', padx=(12, 6), pady=(0, 6))

        self.thresh_var = tk.IntVar(value=initial_thresh)
        slider = tk.Scale(slider_row, from_=100, to=230, orient='horizontal',
                          variable=self.thresh_var, command=self._on_thresh_change,
                          bg='#1a1a1a', fg='#cccccc', troughcolor='#333',
                          activebackground='#EF9F27', highlightthickness=0,
                          length=300, showvalue=True, sliderrelief='flat')
        slider.pack(side='left', pady=(0, 6))

        tk.Label(slider_row,
                 text='lower = more shiny detected   •   higher = more matte detected',
                 bg='#1a1a1a', fg='#888',
                 font=('Helvetica', 10)).pack(side='left', padx=12, pady=(0, 6))

        # canvas
        self.canvas = tk.Canvas(root, width=dw, height=dh,
                                cursor='crosshair', bg='black', highlightthickness=0)
        self.canvas.pack()
        self._image_id = self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

        # bottom bar
        bot = tk.Frame(root, bg='#1a1a1a')
        bot.pack(fill='x')

        self.confirm_btn = tk.Button(
            bot, text='✓  Confirm & Next  (Enter)',
            command=self._confirm,
            state='disabled',
            bg='#1D9E75', fg='white', activebackground='#15735a',
            font=('Helvetica', 12, 'bold'),
            relief='flat', pady=8, padx=20
        )
        self.confirm_btn.pack(side='right', padx=12, pady=8)

        skip_btn = tk.Button(
            bot, text='Skip this image',
            command=self._skip,
            bg='#333', fg='#aaa', activebackground='#444',
            font=('Helvetica', 11), relief='flat', pady=8, padx=12
        )
        skip_btn.pack(side='right', padx=4, pady=8)

        prog_lbl = tk.Label(bot, text=title, bg='#1a1a1a', fg='#888',
                            font=('Helvetica', 10))
        prog_lbl.pack(side='left', padx=12)

        # bindings
        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        root.bind('<Return>', lambda e: self._confirm())
        root.bind('<Escape>', lambda e: self._skip())
        root.protocol('WM_DELETE_WINDOW', self._skip)

    # ── drawing ──────────────────────────────────────────────────────────────

    def _on_press(self, event):
        self._points = [(event.x, event.y)]
        if self._path_id:
            self.canvas.delete(self._path_id)
            self._path_id = None
        self._pending = None
        self.canvas.itemconfig(self._image_id, image=self._tk_img)
        self.confirm_btn.config(state='disabled')

    def _on_drag(self, event):
        if not self._points:
            return
        last = self._points[-1]
        if (event.x - last[0]) ** 2 + (event.y - last[1]) ** 2 < 4:
            return  # skip near-duplicate points to keep the path light
        self._points.append((event.x, event.y))
        if self._path_id:
            self.canvas.delete(self._path_id)
        flat = [c for pt in self._points for c in pt]
        self._path_id = self.canvas.create_line(
            *flat, fill='#FF3333', width=2, smooth=True
        )

    def _on_release(self, _event):
        if len(self._points) < 3:
            self.status_var.set('Outline too short — try again')
            return

        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        if max(xs) - min(xs) < 10 or max(ys) - min(ys) < 10:
            self.status_var.set('Too small — try again')
            return

        # convert display coords → original image coords
        s = self.scale
        ih, iw = self.img_bgr.shape[:2]
        polygon = [
            (max(0, min(iw, int(x / s))), max(0, min(ih, int(y / s))))
            for x, y in self._points
        ]
        self._pending = polygon

        # redraw as a closed polygon (drawn on top of the overlay)
        if self._path_id:
            self.canvas.delete(self._path_id)
        flat = [c for pt in self._points for c in pt]
        self._path_id = self.canvas.create_polygon(
            *flat, outline='#FF3333', fill='', width=2
        )

        self._render_preview()
        self.confirm_btn.config(state='normal')

    def _on_thresh_change(self, _val):
        if self._pending is not None:
            self._render_preview()

    def _render_preview(self):
        """Recompute classification using the slider threshold and paint
        a coloured overlay onto the canvas image."""
        if self._pending is None or len(self._points) < 3:
            return

        thresh = int(self.thresh_var.get())
        pts = np.asarray(self._points, dtype=np.int32).reshape(-1, 1, 2)

        rgb = self._display_rgb
        dh, dw = rgb.shape[:2]
        mask_u8 = np.zeros((dh, dw), np.uint8)
        cv2.fillPoly(mask_u8, [pts], 255)
        mask = mask_u8.astype(bool)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v, hue, sat = hsv[:, :, 2], hsv[:, :, 0], hsv[:, :, 1]

        blue_tape   = (hue >= 95) & (hue <= 135) & (sat > 80)
        white_label = (v > 210) & (sat < 20)
        dark_border = v < 40
        panel_mask  = mask & ~blue_tape & ~white_label & ~dark_border

        shiny = (v > thresh) & panel_mask
        matte = (v <= thresh) & panel_mask

        overlay = rgb.astype(float)
        overlay[shiny]        = [255, 140, 0]
        overlay[matte]        = [0, 160, 100]
        overlay[~panel_mask] *= 0.35
        overlay = overlay.clip(0, 255).astype(np.uint8)

        self._preview_tk = ImageTk.PhotoImage(Image.fromarray(overlay))
        self.canvas.itemconfig(self._image_id, image=self._preview_tk)

        total = panel_mask.sum()
        if total > 0:
            pm = 100 * matte.sum() / total
            ps = 100 * shiny.sum() / total
            self.status_var.set(
                f'Matte {pm:.1f}%  |  Shiny {ps:.1f}%  — Enter to confirm'
            )
        else:
            self.status_var.set('Empty selection — try again')

    def _confirm(self):
        if self._pending is not None:
            self.result = (self._pending, int(self.thresh_var.get()))
        self.root.quit()

    def _skip(self):
        self.result = None
        self.root.quit()


def get_roi(img_bgr, title, initial_thresh=BRIGHTNESS_THRESH):
    """Show selector window; return (polygon, threshold) or None if skipped."""
    root = tk.Tk()
    root.configure(bg='#1a1a1a')
    app  = ROISelector(root, img_bgr, title, initial_thresh=initial_thresh)
    root.mainloop()
    result = app.result
    try:
        root.destroy()
    except tk.TclError:
        pass
    return result


# ── main ────────────────────────────────────────────────────────────────────

def main():
    image_paths = find_images(sys.argv[1:])
    if not image_paths:
        print('No images found. Pass a folder path or list of image files.')
        sys.exit(1)

    output_dir = image_paths[0].parent
    results    = {}
    last_thresh = BRIGHTNESS_THRESH

    print(f'\nFound {len(image_paths)} image(s). A window will open for each one.\n')

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f'  [skip] Could not read {path.name}')
            continue

        title = f'Image {i+1}/{len(image_paths)} — {path.name}'
        print(f'  Opening: {path.name}')

        selection = get_roi(img, title, initial_thresh=last_thresh)
        if selection is None:
            print('  [skipped]')
            continue
        polygon, thresh = selection
        last_thresh = thresh

        pct_m, pct_s, pmask, shiny, matte = classify(img, polygon, thresh=thresh)
        print(f'  → Matte {pct_m:.1f}%   Shiny {pct_s:.1f}%   (threshold {thresh})')

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        results[path.stem] = {
            'pct_matte': round(pct_m, 2),
            'pct_shiny': round(pct_s, 2),
            'bbox':      bbox,
            'polygon':   polygon,
            'threshold': thresh,
            'path':      str(path),
        }

        out_img = output_dir / f'{path.stem}_classified.png'
        save_overlay(img, pmask, shiny, matte, pct_m, pct_s, out_img)
        print(f'  Saved: {out_img.name}')

    if not results:
        print('\nNo images were analysed.')
        return

    # summary chart
    chart_path = save_summary_chart(results, output_dir)
    print(f'\nSummary chart saved: {chart_path.name}')

    # CSV
    import csv
    csv_path = output_dir / 'impregnation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'pct_matte', 'pct_shiny', 'bbox'])
        w.writeheader()
        for name, r in results.items():
            w.writerow({'filename': name, 'pct_matte': r['pct_matte'],
                        'pct_shiny': r['pct_shiny'], 'bbox': r['bbox']})
    print(f'CSV saved:           {csv_path.name}')

    print('\n── Results ─────────────────────────────────────')
    print(f'  {"Panel":<30} {"Matte":>8}  {"Shiny":>8}')
    print('  ' + '─' * 50)
    for name, r in results.items():
        print(f'  {name:<30} {r["pct_matte"]:>7.1f}%  {r["pct_shiny"]:>7.1f}%')
    print()


if __name__ == '__main__':
    main()
