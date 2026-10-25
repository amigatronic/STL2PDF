"""Tests for stl2pdf.

Rendering tests need an OpenGL-capable machine and are skipped unless the
STL2PDF_RENDER_TESTS environment variable is set to 1.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh

# Make the script importable whether the tests sit next to it or in a tests/ subfolder.
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import stl2pdf  # noqa: E402

COLOR = "#4a90d9"


def make_box(path: Path, extents=(10.0, 20.0, 30.0)) -> Path:
    trimesh.creation.box(extents=extents).export(path)
    return path


def run_reports(files, out_dir, output_format="html", merge_path=None, log=None):
    return stl2pdf.generate_reports(
        files, out_dir, output_format, "A4", False, COLOR, merge_path=merge_path, log_callback=log
    )


def count_pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", path.read_bytes()))


@pytest.fixture
def blank_pdf_pages(monkeypatch):
    """Replace PDF page rendering with a blank figure so no OpenGL is needed."""

    def fake_build_pdf_page(model, color, paper, landscape, stem=None):
        return plt.figure(figsize=(2, 2))

    monkeypatch.setattr(stl2pdf, "build_pdf_page", fake_build_pdf_page)


def test_mesh_metrics_for_box(tmp_path):
    mesh = stl2pdf.load_mesh(make_box(tmp_path / "box.stl"))
    metrics = stl2pdf.mesh_metrics(mesh)
    assert np.allclose(metrics["extents"], (10, 20, 30))
    assert metrics["faces"] == 12
    assert metrics["vertices"] == 8
    assert metrics["components"] == 1
    assert metrics["watertight"] is True
    assert metrics["volume"] == pytest.approx(6000.0)
    assert metrics["area"] == pytest.approx(2200.0)


def test_load_mesh_rejects_empty_mesh(tmp_path):
    path = tmp_path / "empty.stl"
    path.write_text("solid empty\nendsolid empty\n")
    with pytest.raises(Exception):
        stl2pdf.load_mesh(path)


def test_discover_files_recursive_flag(tmp_path):
    make_box(tmp_path / "a.stl")
    make_box(tmp_path / "B.STL")
    (tmp_path / "notes.txt").write_text("not a model")
    sub = tmp_path / "sub"
    sub.mkdir()
    make_box(sub / "c.stl")

    flat = stl2pdf.discover_files(tmp_path)
    deep = stl2pdf.discover_files(tmp_path, recursive=True)
    assert [p.name for p in flat] == ["a.stl", "B.STL"]
    assert sorted(p.name for p in deep) == ["B.STL", "a.stl", "c.stl"]


def test_discover_files_single_file_is_returned_as_is(tmp_path):
    path = make_box(tmp_path / "one.stl")
    assert stl2pdf.discover_files(path) == [path]


def test_unique_stem_handles_collisions_case_insensitively():
    used: set[str] = set()
    assert stl2pdf.unique_stem("part", used) == "part"
    assert stl2pdf.unique_stem("Part", used) == "Part_2"
    assert stl2pdf.unique_stem("part", used) == "part_3"


def test_shade_vertices_returns_valid_rgb():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rgb = stl2pdf.shade_vertices(normals, COLOR)
    assert rgb.shape == (3, 3)
    assert rgb.min() >= 0 and rgb.max() <= 255
    assert rgb[0].sum() > rgb[1].sum()  # facing the key light is brighter than facing away


def test_color_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert stl2pdf.load_saved_color() == stl2pdf.DEFAULT_COLOR
    stl2pdf.save_color("#ff8800")
    assert stl2pdf.load_saved_color() == "#ff8800"


@pytest.mark.parametrize("content", ["not json", '{"color": "notacolor"}', "[]"])
def test_invalid_settings_fall_back_to_default(tmp_path, monkeypatch, content):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = stl2pdf.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text(content)
    assert stl2pdf.load_saved_color() == stl2pdf.DEFAULT_COLOR


def test_build_html_contains_dimensions(tmp_path):
    model = stl2pdf.prepare_model(make_box(tmp_path / "box.stl"))
    html = stl2pdf.build_html(model, COLOR)
    assert html.startswith("<!doctype html>")
    assert "box.stl" in html
    for expected in ("10.000", "20.000", "30.000", "6,000.000"):
        assert expected in html


def test_html_generation_and_failure_count(tmp_path):
    good = make_box(tmp_path / "good.stl")
    bad = tmp_path / "bad.stl"
    bad.write_text("solid empty\nendsolid empty\n")
    out_dir = tmp_path / "out"
    messages: list[str] = []

    failures, generated = run_reports([bad, good], out_dir, log=messages.append)

    assert failures == 1
    assert generated == [out_dir / "good.html"]
    assert (out_dir / "good.html").is_file()
    assert any(m.startswith("ERROR: bad.stl") for m in messages)


def test_same_stem_in_different_folders_does_not_overwrite(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    first = make_box(tmp_path / "x" / "part.stl")
    second = make_box(tmp_path / "y" / "part.stl", extents=(5, 5, 5))
    out_dir = tmp_path / "out"

    failures, generated = run_reports([first, second], out_dir)

    assert failures == 0
    assert sorted(p.name for p in generated) == ["part.html", "part_2.html"]


def test_model_is_prepared_once_per_file_for_both_formats(tmp_path, monkeypatch, blank_pdf_pages):
    calls = []
    real_prepare = stl2pdf.prepare_model

    def counting_prepare(path):
        calls.append(path)
        return real_prepare(path)

    monkeypatch.setattr(stl2pdf, "prepare_model", counting_prepare)
    files = [make_box(tmp_path / "a.stl"), make_box(tmp_path / "b.stl")]

    failures, generated = run_reports(files, tmp_path / "out", output_format="both")

    assert failures == 0
    assert len(calls) == 2
    assert len(generated) == 4


def test_merge_creates_one_pdf_with_all_pages_and_missing_parent(tmp_path, blank_pdf_pages):
    files = [make_box(tmp_path / f"m{i}.stl") for i in range(3)]
    merged = tmp_path / "does" / "not" / "exist" / "all.pdf"

    failures, generated = run_reports(files, tmp_path / "out", output_format="pdf", merge_path=merged)

    assert failures == 0
    assert generated == [merged]
    assert count_pdf_pages(merged) == 3


def test_merge_without_any_page_leaves_no_file(tmp_path, blank_pdf_pages):
    bad = tmp_path / "bad.stl"
    bad.write_text("solid empty\nendsolid empty\n")
    merged = tmp_path / "all.pdf"

    failures, generated = run_reports([bad], tmp_path / "out", output_format="pdf", merge_path=merged)

    assert failures == 1
    assert generated == []
    assert not merged.exists()


def test_merge_with_html_only_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_reports([make_box(tmp_path / "a.stl")], tmp_path / "out", output_format="html", merge_path=tmp_path / "x.pdf")


@pytest.mark.skipif(os.environ.get("STL2PDF_RENDER_TESTS") != "1", reason="set STL2PDF_RENDER_TESTS=1 to run rendering tests")
def test_full_pdf_and_html_rendering(tmp_path):
    path = make_box(tmp_path / "box.stl")
    out_dir = tmp_path / "out"

    failures, generated = run_reports([path], out_dir, output_format="both")

    assert failures == 0
    assert (out_dir / "box.pdf").stat().st_size > 10_000
    assert (out_dir / "box.html").is_file()
    assert count_pdf_pages(out_dir / "box.pdf") == 1


@pytest.mark.skipif(os.environ.get("STL2PDF_RENDER_TESTS") != "1", reason="set STL2PDF_RENDER_TESTS=1 to run rendering tests")
def test_each_view_is_rendered_from_a_different_angle(tmp_path):
    model = stl2pdf.prepare_model(make_box(tmp_path / "box.stl", extents=(80, 35, 20)))
    images = stl2pdf.render_view_images(model, stl2pdf.VIEWS, COLOR, size=(300, 300))
    assert len(images) == len(stl2pdf.VIEWS)
    assert len({image.tobytes() for image in images}) == len(images)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
