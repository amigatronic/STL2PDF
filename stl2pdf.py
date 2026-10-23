#!/usr/bin/env python3
"""Generate dimensioned STL reports as PDF, interactive HTML, or both.

Requirements:
    pip install -r requirements.txt

Run without arguments to open the desktop GUI; pass an input path for command-line mode.

Examples:
    python stl2pdf.py model.stl
    python stl2pdf.py model.stl --format both --paper A3
    python stl2pdf.py models/ --format html --paper A5 -o reports/
    python stl2pdf.py model.stl --format pdf --paper A1 --landscape
    python stl2pdf.py model.stl --merge combined.pdf --paper A4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

DEFAULT_COLOR = "#4a90d9"


def settings_path() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home()
    return root / "stl2pdf" / "settings.json"


def load_saved_color(default: str = DEFAULT_COLOR) -> str:
    """Return the model color saved in the previous session, or the default."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        color = data.get("color")
        if isinstance(color, str):
            matplotlib.colors.to_rgb(color)
            return color
    except (OSError, ValueError, AttributeError):
        pass
    return default


def save_color(color: str) -> None:
    """Persist the model color for the next session; failures are ignored."""
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"color": color}), encoding="utf-8")
    except OSError:
        pass


PAPER_SIZES = {
    "A5": (5.83, 8.27),
    "A4": (8.27, 11.69),
    "A3": (11.69, 16.54),
    "A1": (23.39, 33.11),
}

VIEWS = [
    ("Axonometric I", 30, -45),
    ("Axonometric II", 30, 45),
    ("Axonometric III", 55, -135),
    ("Interactive 3D", 25, -55),
]


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("The file does not contain a triangular mesh.")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty or len(loaded.faces) == 0:
        raise ValueError("The file contains an empty or invalid mesh.")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("The mesh contains non-finite vertex coordinates.")
    loaded.remove_unreferenced_vertices()
    loaded.fix_normals()
    return loaded


def format_number(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:,.3f}"


def mesh_metrics(mesh: trimesh.Trimesh) -> dict:
    ext = np.asarray(mesh.extents, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    return {
        "extents": ext,
        "bounds": bounds,
        "area": float(mesh.area),
        "volume": abs(float(mesh.volume)) if mesh.is_watertight else None,
        "watertight": bool(mesh.is_watertight),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "components": int(mesh.body_count),
    }


def smooth_polydata(mesh: trimesh.Trimesh, feature_angle: float = 35.0) -> pv.PolyData:
    """Convert to PolyData with per-vertex normals; vertices are split on sharp edges."""
    faces = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)
    return poly.compute_normals(
        cell_normals=False, point_normals=True,
        split_vertices=True, feature_angle=feature_angle,
    )


def shade_vertices(normals: np.ndarray, color: str) -> np.ndarray:
    """Precompute per-vertex RGB (0-255) with two fixed lights, independent of the viewer's shader."""
    key = np.array([0.35, -0.45, 0.82])
    fill = np.array([-0.55, 0.60, 0.30])
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    intensity = 0.30 + 0.55 * np.clip(normals @ key, 0.0, 1.0) + 0.25 * np.clip(normals @ fill, 0.0, 1.0)
    base = np.array(matplotlib.colors.to_rgb(color))
    rgb = np.clip(intensity[:, None] * base[None, :], 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(int)


def render_view_images(mesh: trimesh.Trimesh, views, color: str, size=(1400, 1400)) -> list[np.ndarray]:
    """Render all views off-screen with a real z-buffer (VTK) and return RGB images."""
    poly = smooth_polydata(mesh)

    mn, mx = np.asarray(mesh.bounds, dtype=float)
    center = (mn + mx) / 2.0
    diag = float(np.linalg.norm(mx - mn))
    corners = np.array(list(itertools.product(*zip(mn, mx)))) - center
    aspect = size[0] / size[1]
    world_up = np.array([0.0, 0.0, 1.0])

    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    images = []
    try:
        plotter.set_background("white")
        plotter.add_mesh(poly, color=color, smooth_shading=True,
                         ambient=0.25, diffuse=0.8, specular=0.15)
        plotter.enable_parallel_projection()
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass

        for _title, elev, azim in views:
            e, a = np.radians(elev), np.radians(azim)
            direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            forward = -direction
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)

            half_w = float(np.abs(corners @ right).max())
            half_h = float(np.abs(corners @ up).max())
            scale = max(half_h, half_w / aspect, 1e-6) * 1.08

            plotter.camera_position = [
                tuple(center + direction * diag * 2.0),
                tuple(center),
                tuple(world_up),
            ]
            plotter.camera.parallel_scale = scale
            plotter.reset_camera_clipping_range()
            images.append(plotter.screenshot(return_img=True))
    finally:
        plotter.close()
    return images


def add_dimension_box(ax, mesh, color="#333333") -> None:
    """Draw a thin bounding box and dimension labels around the model."""
    mn, mx = mesh.bounds
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=color, linewidth=0.55, alpha=0.65)
    ext = mx - mn
    ax.text(mx[0], mn[1], mn[2], f" X {format_number(ext[0])}", fontsize=6, color=color)
    ax.text(mn[0], mx[1], mn[2], f" Y {format_number(ext[1])}", fontsize=6, color=color)
    ax.text(mn[0], mn[1], mx[2], f" Z {format_number(ext[2])}", fontsize=6, color=color)


def build_pdf_page(path: Path, color: str, paper: str, landscape: bool):
    mesh = load_mesh(path)
    metrics = mesh_metrics(mesh)
    width, height = PAPER_SIZES[paper]
    if landscape:
        width, height = height, width

    images = render_view_images(mesh, VIEWS, color)

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.suptitle(path.name, fontsize=16 if paper in ("A5", "A4") else 22, fontweight="bold", y=0.975)
    positions = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    for (title, _elev, _azim), position, image in zip(VIEWS, positions, images):
        ax = fig.add_subplot(*position)
        ax.imshow(image, interpolation="lanczos")
        ax.set_title(title, fontsize=10, pad=5, fontweight="bold")
        ax.axis("off")

    ext = metrics["extents"]
    mn, mx = metrics["bounds"]
    unit_note = "STL coordinates (STL files do not define a unit)"
    details = [
        f"Overall dimensions X x Y x Z: {format_number(ext[0])} x {format_number(ext[1])} x {format_number(ext[2])}",
        f"Bounds X [{format_number(mn[0])}, {format_number(mx[0])}]  Y [{format_number(mn[1])}, {format_number(mx[1])}]  Z [{format_number(mn[2])}, {format_number(mx[2])}]",
        f"Triangles: {metrics['faces']:,}    Vertices: {metrics['vertices']:,}    Connected components: {metrics['components']}",
        f"Watertight: {'Yes' if metrics['watertight'] else 'No'}    Surface area: {format_number(metrics['area'])} square file-units",
    ]
    if metrics["volume"] is not None:
        details.append(f"Volume: {format_number(metrics['volume'])} cubic file-units")
    details.append(unit_note)
    fig.text(0.055, 0.055, "\n".join(details), fontsize=8 if paper in ("A5", "A4") else 11,
             va="bottom", family="monospace", linespacing=1.5)
    interactive_name = f"{path.stem}.html"
    fig.text(0.97, 0.018, f"Open interactive 3D report: {interactive_name}",
             ha="right", va="bottom", fontsize=7, color="#1558a6",
             url=interactive_name)
    fig.subplots_adjust(top=0.925, bottom=0.225, left=0.045, right=0.955, hspace=0.16, wspace=0.10)
    return fig


def build_html(path: Path, color: str) -> str:
    mesh = load_mesh(path)
    metrics = mesh_metrics(mesh)
    poly = smooth_polydata(mesh)
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]
    vertex_colors = shade_vertices(np.asarray(poly.point_data["Normals"]), color)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[view[0] for view in VIEWS],
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    camera_specs = [
        dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        dict(eye=dict(x=-1.5, y=-1.5, z=1.8)),
        dict(eye=dict(x=1.7, y=-1.7, z=1.35)),
    ]
    for index, camera in enumerate(camera_specs):
        row, col = index // 2 + 1, index % 2 + 1
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            vertexcolor=vertex_colors, opacity=1.0, flatshading=False,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            name=path.name, showscale=False,
            hovertemplate="X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
        ), row=row, col=col)
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        center = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
        fig.layout[scene_name].update(
            camera={**camera, "center": {"x": 0, "y": 0, "z": 0}, "projection": {"type": "orthographic"}},
            aspectmode="data",
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="#f7f9fc",
            xaxis=dict(showbackground=True, backgroundcolor="#f7f9fc", gridcolor="#dce2ea"),
            yaxis=dict(showbackground=True, backgroundcolor="#f7f9fc", gridcolor="#dce2ea"),
            zaxis=dict(showbackground=True, backgroundcolor="#f7f9fc", gridcolor="#dce2ea"),
        )
    fig.update_layout(
        title=dict(text=path.name, x=0.5),
        height=1050,
        margin=dict(l=10, r=10, t=65, b=15),
        paper_bgcolor="white",
        showlegend=False,
        font=dict(family="Arial, sans-serif", color="#202a36"),
    )
    html_plot = fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
    ext = metrics["extents"]
    mn, mx = metrics["bounds"]
    volume = format_number(metrics["volume"]) if metrics["volume"] is not None else "Unavailable (mesh is not watertight)"
    details = f"""
    <section class="metrics">
      <h2>Dimensions and mesh information</h2>
      <div class="metric-grid">
        <div><span>Overall X</span><strong>{format_number(ext[0])}</strong></div>
        <div><span>Overall Y</span><strong>{format_number(ext[1])}</strong></div>
        <div><span>Overall Z</span><strong>{format_number(ext[2])}</strong></div>
        <div><span>Surface area</span><strong>{format_number(metrics['area'])}</strong></div>
        <div><span>Volume</span><strong>{volume}</strong></div>
        <div><span>Watertight</span><strong>{'Yes' if metrics['watertight'] else 'No'}</strong></div>
        <div><span>Triangles</span><strong>{metrics['faces']:,}</strong></div>
        <div><span>Vertices</span><strong>{metrics['vertices']:,}</strong></div>
        <div><span>Connected components</span><strong>{metrics['components']}</strong></div>
      </div>
      <p class="note">STL files do not store a measurement unit. All dimensions and derived quantities use the model's raw coordinate units.</p>
      <p class="note">Bounding coordinates: X [{format_number(mn[0])}, {format_number(mx[0])}], Y [{format_number(mn[1])}, {format_number(mx[1])}], Z [{format_number(mn[2])}, {format_number(mx[2])}].</p>
    </section>
    """
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>STL Report - {path.name}</title>
<style>
:root {{ color-scheme: light; font-family: Arial, sans-serif; color: #202a36; background: #eef2f7; }}
body {{ margin: 0; }} header {{ padding: 18px 24px; background: #172334; color: white; }}
header h1 {{ margin: 0; font-size: 1.35rem; overflow-wrap: anywhere; }} main {{ max-width: 1500px; margin: 18px auto; padding: 0 14px; }}
.viewer {{ background: white; border-radius: 12px; box-shadow: 0 3px 18px #1b2c4212; overflow: hidden; }}
.metrics {{ margin-top: 18px; background: white; border-radius: 12px; padding: 18px; box-shadow: 0 3px 18px #1b2c4212; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 10px; }}
.metric-grid div {{ padding: 12px; background: #f5f7fa; border-radius: 8px; }}
.metric-grid span {{ display: block; color: #5c6878; font-size: .82rem; margin-bottom: 6px; }}
.metric-grid strong {{ font-size: 1.05rem; overflow-wrap: anywhere; }} .note {{ color: #586575; font-size: .9rem; }}
@media print {{ body {{ background: white; }} header {{ background: white; color: black; }} main {{ margin: 0; max-width: none; }} .viewer,.metrics {{ box-shadow: none; }} }}
</style></head><body><header><h1>STL Technical Report</h1></header><main><div class="viewer">{html_plot}</div>{details}</main></body></html>"""


def discover_files(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".stl"), key=lambda p: p.name.lower())
    return [source]


def generate_reports(
    files: list[Path],
    output_dir: Path,
    output_format: str,
    paper: str,
    landscape: bool,
    color: str,
    merge_path: Path | None = None,
    progress_callback=None,
    log_callback=None,
) -> tuple[int, list[Path]]:
    """Generate reports and optionally notify a graphical interface."""
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    generated: list[Path] = []
    merged_pdf = PdfPages(merge_path) if merge_path else None
    try:
        for index, path in enumerate(files, start=1):
            if log_callback:
                log_callback(f"Processing {path.name} ({index}/{len(files)})...")
            try:
                if output_format in ("pdf", "both"):
                    figure = build_pdf_page(path, color, paper, landscape)
                    try:
                        if merged_pdf:
                            merged_pdf.savefig(figure, facecolor="white")
                        else:
                            pdf_path = output_dir / f"{path.stem}.pdf"
                            figure.savefig(pdf_path, facecolor="white")
                            generated.append(pdf_path)
                            if log_callback:
                                log_callback(f"Created PDF: {pdf_path}")
                    finally:
                        plt.close(figure)
                if output_format in ("html", "both"):
                    html_path = output_dir / f"{path.stem}.html"
                    html_path.write_text(build_html(path, color), encoding="utf-8")
                    generated.append(html_path)
                    if log_callback:
                        log_callback(f"Created HTML: {html_path}")
            except Exception as exc:
                failures += 1
                if log_callback:
                    log_callback(f"ERROR: {path.name}: {exc}")
            if progress_callback:
                progress_callback(index, len(files))
    finally:
        if merged_pdf:
            merged_pdf.close()
            generated.append(merge_path)
            if log_callback:
                log_callback(f"Created merged PDF: {merge_path}")
    return failures, generated


class STL2PDFApp:
    """Tkinter desktop interface for STL report generation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("STL Technical Report Generator")
        self.root.configure(bg="#eef2f7")

        self.files: list[Path] = []
        self.color = load_saved_color()
        self.busy = False

        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Documents"))
        self.format_var = tk.StringVar(value="both")
        self.paper_var = tk.StringVar(value="A4")
        self.orientation_var = tk.StringVar(value="Portrait")
        self.merge_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready. Select one or more STL files to begin.")

        self._build_ui()
        self._fit_window()

    def _fit_window(self):
        """Size the window from the real content height so the footer is never clipped."""
        self.root.update_idletasks()
        max_h = max(400, self.root.winfo_screenheight() - 80)
        need_w = self.root.winfo_reqwidth()
        need_h = min(self.root.winfo_reqheight(), max_h)
        self.root.minsize(need_w, need_h)
        self.root.geometry(f"{max(850, need_w)}x{min(max(720, need_h), max_h)}")

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("Card.TLabelframe", background="#ffffff", padding=12)
        style.configure("Card.TLabelframe.Label", background="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#eef2f7", font=("Segoe UI", 9))
        style.configure("Card.TLabel", background="#ffffff", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#172334", foreground="#ffffff", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", background="#172334", foreground="#d7e0eb", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))

        header = tk.Frame(self.root, bg="#172334", padx=22, pady=17)
        header.pack(fill="x")
        ttk.Label(header, text="STL Technical Report Generator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Create dimensioned PDF sheets and interactive 3D HTML reports", style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))

        # Footer is packed first (bottom) so it always keeps its space.
        footer = ttk.Frame(self.root, padding=(16, 0, 16, 14))
        footer.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 7))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.generate_button = ttk.Button(footer, text="Generate reports", style="Accent.TButton", command=self.start_generation)
        self.generate_button.pack(side="right", padx=(10, 0))

        body = ttk.Frame(self.root, padding=(16, 14, 16, 10))
        body.pack(fill="both", expand=True)

        file_card = ttk.LabelFrame(body, text="Input models", style="Card.TLabelframe")
        file_card.pack(fill="both", expand=True, pady=(0, 10))
        file_buttons = ttk.Frame(file_card)
        file_buttons.pack(fill="x")
        ttk.Button(file_buttons, text="Add STL files...", command=self.add_files).pack(side="left")
        ttk.Button(file_buttons, text="Add folder...", command=self.add_folder).pack(side="left", padx=(8, 0))
        ttk.Button(file_buttons, text="Remove selected", command=self.remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(file_buttons, text="Clear list", command=self.clear_files).pack(side="right")

        list_frame = ttk.Frame(file_card)
        list_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=7, font=("Consolas", 9),
                                    bg="#f8fafc", fg="#202a36", selectbackground="#c8def8", relief="flat",
                                    highlightthickness=1, highlightbackground="#d7dee8")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.file_count_var = tk.StringVar(value="0 files selected")
        ttk.Label(file_card, textvariable=self.file_count_var, style="Card.TLabel").pack(anchor="e", pady=(5, 0))

        output_card = ttk.LabelFrame(body, text="Output", style="Card.TLabelframe")
        output_card.pack(fill="x", pady=(0, 10))
        out_row = ttk.Frame(output_card)
        out_row.pack(fill="x")
        ttk.Label(out_row, text="Output folder", style="Card.TLabel", width=14).pack(side="left")
        self.output_entry = ttk.Entry(out_row, textvariable=self.output_dir_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(out_row, text="Browse...", command=self.choose_output).pack(side="right")

        settings_card = ttk.LabelFrame(body, text="Report settings", style="Card.TLabelframe")
        settings_card.pack(fill="x", pady=(0, 10))
        settings_grid = ttk.Frame(settings_card)
        settings_grid.pack(fill="x")

        ttk.Label(settings_grid, text="Output format", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        format_box = ttk.Combobox(settings_grid, textvariable=self.format_var, values=("pdf", "html", "both"), state="readonly", width=12)
        format_box.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Label(settings_grid, text="Paper size", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(24, 8), pady=5)
        ttk.Combobox(settings_grid, textvariable=self.paper_var, values=tuple(PAPER_SIZES), state="readonly", width=8).grid(row=0, column=3, sticky="w", pady=5)
        ttk.Label(settings_grid, text="Orientation", style="Card.TLabel").grid(row=0, column=4, sticky="w", padx=(24, 8), pady=5)
        ttk.Combobox(settings_grid, textvariable=self.orientation_var, values=("Portrait", "Landscape"), state="readonly", width=12).grid(row=0, column=5, sticky="w", pady=5)

        color_row = ttk.Frame(settings_card)
        color_row.pack(fill="x", pady=(8, 0))
        ttk.Label(color_row, text="Model color", style="Card.TLabel").pack(side="left")
        self.color_swatch = tk.Label(color_row, text="   ", bg=self.color, relief="solid", borderwidth=1)
        self.color_swatch.pack(side="left", padx=(10, 6))
        ttk.Button(color_row, text="Choose color...", command=self.choose_color).pack(side="left")
        ttk.Checkbutton(color_row, text="Combine PDF pages into one file", variable=self.merge_var, command=self._toggle_merge).pack(side="left", padx=(20, 0))
        self.merge_path: Path | None = None
        self.merge_button = ttk.Button(color_row, text="Choose merged PDF...", command=self.choose_merge_path, state="disabled")
        self.merge_button.pack(side="left", padx=(8, 0))

        log_card = ttk.LabelFrame(body, text="Progress and messages", style="Card.TLabelframe")
        log_card.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_card, height=5, wrap="word", font=("Consolas", 9), bg="#f8fafc", fg="#263445", relief="flat", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_files(self):
        self.file_list.delete(0, "end")
        for path in self.files:
            self.file_list.insert("end", str(path))
        self.file_count_var.set(f"{len(self.files)} file(s) selected")

    def add_files(self):
        selected = filedialog.askopenfilenames(title="Select STL models", filetypes=(("STL files", "*.stl *.STL"), ("All files", "*.*")))
        self._add_paths(Path(item) for item in selected)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select a folder containing STL models")
        if folder:
            paths = sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == ".stl"), key=lambda p: p.name.lower())
            self._add_paths(paths)
            if not paths:
                messagebox.showinfo("No STL files", "The selected folder does not contain any STL files.", parent=self.root)

    def _add_paths(self, paths):
        known = {str(p.resolve()).casefold() for p in self.files}
        for path in paths:
            if path.is_file() and path.suffix.lower() == ".stl" and str(path.resolve()).casefold() not in known:
                self.files.append(path)
                known.add(str(path.resolve()).casefold())
        self.files.sort(key=lambda p: str(p).casefold())
        self._refresh_files()

    def remove_selected(self):
        selected = set(self.file_list.curselection())
        self.files = [path for index, path in enumerate(self.files) if index not in selected]
        self._refresh_files()

    def clear_files(self):
        self.files.clear()
        self._refresh_files()

    def choose_output(self):
        folder = filedialog.askdirectory(title="Choose output folder", initialdir=self.output_dir_var.get() or str(Path.home()))
        if folder:
            self.output_dir_var.set(folder)

    def choose_color(self):
        chosen = colorchooser.askcolor(color=self.color, title="Choose model color", parent=self.root)
        if chosen and chosen[1]:
            self.color = chosen[1]
            self.color_swatch.configure(bg=self.color)
            save_color(self.color)

    def _toggle_merge(self):
        self.merge_button.configure(state="normal" if self.merge_var.get() else "disabled")

    def choose_merge_path(self):
        selected = filedialog.asksaveasfilename(title="Save combined PDF", defaultextension=".pdf", filetypes=(("PDF files", "*.pdf"),), initialfile="combined_report.pdf")
        if selected:
            self.merge_path = Path(selected)
            self.merge_button.configure(text=f"Merged: {self.merge_path.name}")

    def start_generation(self):
        if self.busy:
            return
        if not self.files:
            messagebox.showwarning("No input models", "Add at least one STL file before generating reports.", parent=self.root)
            return
        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not self.output_dir_var.get().strip():
            messagebox.showwarning("Output folder required", "Choose an output folder.", parent=self.root)
            return
        output_format = self.format_var.get()
        merge_path = self.merge_path if self.merge_var.get() else None
        if self.merge_var.get() and output_format == "html":
            messagebox.showwarning("Invalid output settings", "Combining pages is available only when PDF output is enabled.", parent=self.root)
            return
        if self.merge_var.get() and merge_path is None:
            self.choose_merge_path()
            if self.merge_path is None:
                return
            merge_path = self.merge_path
        if merge_path and output_format == "both":
            # HTML reports are still emitted individually; only PDF pages are merged.
            pass
        try:
            matplotlib.colors.to_rgb(self.color)
        except ValueError:
            messagebox.showerror("Invalid color", "Choose a valid model color.", parent=self.root)
            return

        files = list(self.files)
        self.busy = True
        self.generate_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(files))
        self.status_var.set("Generating reports...")
        self._log(f"Starting report generation for {len(files)} model(s).")
        args = (files, output_dir, output_format, self.paper_var.get(), self.orientation_var.get() == "Landscape", self.color, merge_path)
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, files, output_dir, output_format, paper, landscape, color, merge_path):
        def progress(current, total):
            self.root.after(0, lambda: self.progress.configure(value=current, maximum=total))
        def log(message):
            self.root.after(0, lambda m=message: self._log(m))
        try:
            failures, generated = generate_reports(files, output_dir, output_format, paper, landscape, color,
                                                   merge_path=merge_path, progress_callback=progress, log_callback=log)
            self.root.after(0, lambda: self._finish(failures, len(files), generated))
        except Exception as exc:
            error_text = str(exc)
            self.root.after(0, lambda: self._fatal_error(error_text))

    def _finish(self, failures: int, total: int, generated: list[Path]):
        self.busy = False
        self.generate_button.configure(state="normal")
        if failures:
            self.status_var.set(f"Finished with {failures} error(s) out of {total} model(s).")
            self._log(f"Finished with errors. {len(generated)} output file(s) created.")
            messagebox.showwarning("Generation completed with errors", f"{failures} of {total} model(s) could not be processed. See the log for details.", parent=self.root)
        else:
            self.status_var.set(f"Completed successfully. {len(generated)} output file(s) created.")
            self._log("All reports generated successfully.")
            messagebox.showinfo("Generation complete", f"Created {len(generated)} output file(s).\n\nOutput folder:\n{self.output_dir_var.get()}", parent=self.root)

    def _fatal_error(self, message: str):
        self.busy = False
        self.generate_button.configure(state="normal")
        self.status_var.set("Generation failed.")
        self._log(f"FATAL ERROR: {message}")
        messagebox.showerror("Generation failed", message, parent=self.root)


def launch_gui() -> int:
    root = tk.Tk()
    STL2PDFApp(root)
    root.mainloop()
    return 0


def main() -> int:
    # Launch the desktop GUI by default; command-line mode remains available when arguments are supplied.
    if len(sys.argv) == 1:
        return launch_gui()

    parser = argparse.ArgumentParser(description="Generate technical reports from STL files.")
    parser.add_argument("input", help="An STL file or a directory containing STL files")
    parser.add_argument("-o", "--outdir", help="Output directory; defaults to the input directory")
    parser.add_argument("--format", choices=("pdf", "html", "both"), default="both", help="Output format (default: both)")
    parser.add_argument("--paper", choices=tuple(PAPER_SIZES), default="A4", help="PDF paper size (default: A4)")
    parser.add_argument("--landscape", action="store_true", help="Use landscape orientation for the PDF")
    parser.add_argument("--merge", metavar="FILE.pdf", help="Combine all input models into one multi-page PDF")
    parser.add_argument("--color", default="#4a90d9", help="Model color in hexadecimal notation (default: #4a90d9)")
    args = parser.parse_args()

    try:
        matplotlib.colors.to_rgb(args.color)
    except ValueError:
        parser.error("--color must be a valid Matplotlib color, for example #4a90d9")

    source = Path(args.input)
    if not source.exists():
        parser.error(f"Input path does not exist: {source}")
    files = discover_files(source)
    if not files:
        parser.error("No STL files found in the input path.")
    if args.merge and args.format == "html":
        parser.error("--merge can only be used when PDF output is enabled.")

    output_dir = Path(args.outdir) if args.outdir else (source if source.is_dir() else source.parent)
    failures, _ = generate_reports(files, output_dir, args.format, args.paper, args.landscape, args.color,
                                   merge_path=Path(args.merge) if args.merge else None,
                                   log_callback=print)
    return 1 if failures == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
