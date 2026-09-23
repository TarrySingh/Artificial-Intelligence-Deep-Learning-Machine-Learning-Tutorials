#!/usr/bin/env python3
"""Redraw the Synapsa Commons lockups in brand/. Maintainer tool; needs network once, for the font.

    python tools/brand_lockups.py

The mark's geometry, colours and type settings are copied from the Synapsa wordmark
(logo-wordmark.svg, viewBox 220 x 48): a Synapse Blue circle and a Saffron circle joined by a
synapse curve, "Synapsa" in Geist 700 at 26 px with -0.04em tracking. "Commons" follows in Geist
400 -- the brand differentiates by weight, not by a second face. Three variants, each at 2x:

  synapsa-commons-light.png   black ink, for light backgrounds (README, GitHub light theme)
  synapsa-commons-dark.png    white ink and the brand's on-dark blue and saffron
  synapsa-commons-badge.png   the light lockup on a white rounded tile, which reads on both
                              notebook themes -- used at the top of every lesson.ipynb

Text is drawn as glyph outlines, so the PNGs do not depend on Geist being installed where they
are viewed. Geist (SIL Open Font Licence) is fetched from Google Fonts into a temporary directory.
"""
import colorsys, re, tempfile, urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

OUT = Path(__file__).resolve().parent.parent / "brand"
SIZE = 26.0
TRACK = -0.04 * SIZE


def hsl(h, s, l):
    return colorsys.hls_to_rgb(h / 360, l / 100, s / 100)


def geist(tmp: Path) -> tuple:
    """Geist 400 and 700 as TTF. An old user agent makes Google Fonts serve TTF, not WOFF2."""
    req = urllib.request.Request("https://fonts.googleapis.com/css2?family=Geist:wght@400;700",
                                 headers={"User-Agent": "Mozilla/4.0"})
    urls = re.findall(r"url\((https://[^)]+\.ttf)\)", urllib.request.urlopen(req, timeout=30).read().decode())
    if len(urls) != 2:
        raise SystemExit(f"expected two Geist TTF URLs from Google Fonts, got {len(urls)}")
    paths = []
    for i, u in enumerate(urls):   # served in weight order: 400, then 700
        f = tmp / f"geist-{i}.ttf"
        f.write_bytes(urllib.request.urlopen(u, timeout=30).read())
        paths.append(FontProperties(fname=str(f)))
    return tuple(paths)


def tracked(text, prop, x0):
    """Glyph-by-glyph outlines with the wordmark's -0.04em tracking."""
    bar = TextPath((0, 0), "|", size=SIZE, prop=prop).get_extents().width
    paths, x = [], x0
    for ch in text:
        if ch != " ":
            paths.append(TextPath((x, 0), ch, size=SIZE, prop=prop))
        x += TextPath((0, 0), ch + "|", size=SIZE, prop=prop).get_extents().width - bar + TRACK
    return paths, x


def draw(name, fonts, ink, blue, saffron, stroke, bg=None):
    regular, bold = fonts
    pad = 14 if bg else 0
    word, xe = tracked("Synapsa", bold, 0)
    word2, xe2 = tracked("Commons", regular, xe + 0.28 * SIZE)
    W, H = pad + 52 + (xe2 - TRACK) + (pad if bg else 2), 48
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_alpha(0)
    if bg:
        ax.add_patch(FancyBboxPatch((0.5, 0.5), W - 1, H - 1, boxstyle="round,pad=0,rounding_size=12",
                                    fc=bg, ec=hsl(0, 0, 88), lw=0.5))
    ax.add_patch(Circle((pad + 13, 24), 9, fc=blue, ec="none"))
    ax.add_patch(Circle((pad + 32, 24), 5, fc=saffron, ec="none"))
    curve = MPath([(pad + 22, 24), (pad + 27, 16), (pad + 31, 24)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
    ax.add_patch(PathPatch(curve, fc="none", ec=stroke, lw=2.8 * 72 / 100, capstyle="round"))
    place = Affine2D().scale(1, -1).translate(pad + 52, 32)
    for p in word + word2:
        ax.add_patch(PathPatch(place.transform_path(p), fc=ink, ec="none"))
    fig.savefig(OUT / f"{name}.png", dpi=200, transparent=True)   # 2x
    plt.close(fig)
    print(f"  wrote brand/{name}.png  ({round(W) * 2} x {H * 2} px)")


def main():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fonts = geist(Path(tmp))
        draw("synapsa-commons-light", fonts, "#000000", hsl(228, 95, 54), hsl(34, 95, 56), "#000000")
        draw("synapsa-commons-dark", fonts, "#ffffff", hsl(228, 100, 64), hsl(34, 100, 60), "#ffffff")
        draw("synapsa-commons-badge", fonts, "#000000", hsl(228, 95, 54), hsl(34, 95, 56), "#000000", bg="#ffffff")


if __name__ == "__main__":
    main()
