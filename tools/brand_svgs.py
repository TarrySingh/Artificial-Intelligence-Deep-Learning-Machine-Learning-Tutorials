#!/usr/bin/env python3
"""Draw the README artwork of Synapsa Commons as small, hand-built SVGs. Maintainer tool.

    python tools/brand_svgs.py            # write every file (fetches the fonts once, then offline)
    python tools/brand_svgs.py --check    # exit 1 if any file is missing, stale or stray
    python tools/brand_svgs.py --lint     # the rules every shipped SVG obeys; needs no fonts

One idea runs through every file: the Synapsa mark is two nodes and the arc that crosses between
them. You are the blue node; the running notebook is where the signal lands. So every area
glyph has exactly one filled Synapse Blue node, the point where the learner's work happens, and
saffron appears in a few places only: the mark in the hero lockup, the node where the hero's arc
lands, and the invitation banner's mark and its "Synapsa" pillow.

GitHub shows an SVG only through <img>, where web fonts never load. So every word is drawn as
Geist (or JetBrains Mono, for eyebrows) glyph outlines, each glyph defined once and placed with
<use>, and repeated in the file's <title>, which is what the export's mention scan reads. Every
panel is opaque, so the light file, which is what a renderer that ignores <picture> shows, stays
legible on a dark page. The fonts (SIL Open Font Licence) are fetched from Google Fonts at the
pinned versions below, checked against their hashes, and cached outside the repository.

Areas (programmes, the flagship, tracks) come from internal/areas.yaml and get a tile and an
icon only once one of their lessons is released, so nothing here names unreleased work.
"""
import hashlib, math, re, sys, urllib.request
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
README_DIR = ROOT / "brand" / "readme"
AREA_DIR = ROOT / "brand" / "areas"
FONT_CACHE = Path.home() / ".cache" / "synapsa-commons" / "fonts"
GF = "https://fonts.gstatic.com/s/"
FONTS = {   # name: (pinned URL, sha256). Geist 1.800 and JetBrains Mono 2.211, as Google serves TTF.
    "g400": (GF + "geist/v5/gyBhhwUxId8gMGYQMKR3pzfaWI_RnOM4nQ.ttf",
             "ccce59465ae5c61930ada5453834f60500fbd95084d85db63e0e65fad0125455"),
    "g500": (GF + "geist/v5/gyBhhwUxId8gMGYQMKR3pzfaWI_RruM4nQ.ttf",
             "a8eefb873ba9c9020665f8a98302697ce1f3b79ab2deef6ee508c0ca6a82e876"),
    "g600": (GF + "geist/v5/gyBhhwUxId8gMGYQMKR3pzfaWI_RQuQ4nQ.ttf",
             "7171f481b121803d007f5c6c76cd9988c0d41cce4202db9a18dff9945505d36d"),
    "g700": (GF + "geist/v5/gyBhhwUxId8gMGYQMKR3pzfaWI_Re-Q4nQ.ttf",
             "1a9874b3be0fc168692a3d2327bccb4d28835653d3b91537a755296a467a83e3"),
    "mono": (GF + "jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8-qxjPQ.ttf",
             "3386a05f6ece969e4537de6be894170d20558e82f7d56c8c5d332972ef172160"),
}
MAX_BYTES = 40 * 1024

# Hex only, from the Synapsa tokens. Light files sit on an opaque white panel, dark on pure black.
THEMES = {
    "light": dict(panel="#ffffff", subtle="#f7f7f7", line="#e6e6e6", line2="#c7c7c7", head="#000000",
                  ink="#1a1a1a", muted="#616161", bar="#c7c7c7", blue="#1a47f9", saffron="#f99d24",
                  done="#239f65", fail="#e52e34"),
    "dark": dict(panel="#000000", subtle="#0a0a0a", line="#242424", line2="#383838", head="#ffffff",
                 ink="#f5f5f5", muted="#a3a3a3", bar="#383838", blue="#476cff", saffron="#ffa733",
                 done="#31d88a", fail="#f2545a"),
}
SERVICES = ("Colab", "Kaggle", "Binder", "Codespaces")
EASE = "cubic-bezier(.2,.8,.2,1)"


def num(v: float) -> str:
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------------------- fonts

class FontsUnavailable(Exception):
    pass


class Face:
    def __init__(self, path: Path):
        from fontTools.ttLib import TTFont
        self.font = TTFont(str(path))
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.hmtx = self.font["hmtx"]

    def advance(self, ch: str, size: float) -> float:
        return self.hmtx[self.cmap[ord(ch)]][0] * size / self.upm

    def outline(self, ch: str, size: float) -> str:
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.transformPen import TransformPen
        k = size / self.upm
        pen = SVGPathPen(self.glyphs, ntos=num)
        self.glyphs[self.cmap[ord(ch)]].draw(TransformPen(pen, (k, 0, 0, -k, 0, 0)))
        return pen.getCommands()

    def width(self, text: str, size: float, track: float = 0.0) -> float:
        """Advance width of a run, without the tracking after its last glyph."""
        return sum(self.advance(c, size) for c in text) + track * size * max(len(text) - 1, 0)


_FACES = {}


def faces() -> dict:
    """The five faces, fetched once into the cache at their pinned versions and hash-checked."""
    if _FACES:
        return _FACES
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FONTS.items():
        f = FONT_CACHE / f"{name}-{sha[:12]}.ttf"
        if not f.exists():
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
            except OSError as e:
                raise FontsUnavailable(f"cannot fetch {url}: {e}")
            if hashlib.sha256(data).hexdigest() != sha:
                raise FontsUnavailable(f"{url} is not the pinned file (sha256 differs)")
            f.write_bytes(data)
        _FACES[name] = Face(f)
    return _FACES


# ----------------------------------------------------------------------------------- documents

class Svg:
    """One SVG file: glyph outlines in <defs>, everything else as plain shapes, hex colours."""

    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title = w, h, title
        self.defs, self.body, self.style = {}, [], ""

    def add(self, *parts: str):
        self.body.extend(parts)

    def text(self, s: str, face: str, size: float, x: float, y: float, fill: str,
             track: float = 0.0, anchor: str = "start") -> float:
        f = faces()[face]
        w = f.width(s, size, track)
        x -= {"start": 0, "middle": w / 2, "end": w}[anchor]
        uses = []
        for ch in s:
            if ch != " ":
                key = (face, size, ch)
                if key not in self.defs:
                    self.defs[key] = (f"g{len(self.defs):x}", f.outline(ch, size))
                uses.append(f'<use href="#{self.defs[key][0]}" x="{num(x)}" y="{num(y)}"/>')
            x += f.advance(ch, size) + track * size
        self.add(f'<g fill="{fill}">' + "".join(uses) + "</g>")
        return w

    def render(self) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" role="img">', f"<title>{esc(self.title)}</title>"]
        if self.style:
            out.append(f"<style>{self.style}</style>")
        if self.defs:
            out.append("<defs>" + "".join(f'<path id="{i}" d="{d}"/>' for i, d in self.defs.values())
                       + "</defs>")
        return "\n".join(out + self.body + ["</svg>"]) + "\n"


def rect(x, y, w, h, rx, fill="none", stroke=None, sw=1.0, extra=""):
    s = f' stroke="{stroke}" stroke-width="{num(sw)}"' if stroke else ""
    return (f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" rx="{num(rx)}" '
            f'fill="{fill}"{s}{extra}/>')


def circle(cx, cy, r, fill="none", stroke=None, sw=1.0, extra=""):
    s = f' stroke="{stroke}" stroke-width="{num(sw)}"' if stroke else ""
    return f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(r)}" fill="{fill}"{s}{extra}/>'


def path(d, stroke, sw, fill="none", extra=""):
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{num(sw)}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def panel(svg: Svg, t: dict, rx: float):
    svg.add(rect(0.5, 0.5, svg.w - 1, svg.h - 1, rx, t["panel"], t["line"]))


def eyebrow(svg: Svg, s: str, x: float, y: float, fill: str, anchor="start"):
    """JetBrains Mono 500, 11 px, uppercase, +0.04em: supplementary labels only."""
    return svg.text(s.upper(), "mono", 11, x, y, fill, 0.04, anchor)


def wrap(text: str, face: str, size: float, track: float, width: float, lines: int) -> list:
    words, out = text.split(), [""]
    for w in words:
        trial = (out[-1] + " " + w).strip()
        if faces()[face].width(trial, size, track) <= width or not out[-1]:
            out[-1] = trial
        else:
            out.append(w)
    if len(out) > lines or any(faces()[face].width(l, size, track) > width for l in out):
        raise SystemExit(f"REFUSING: {text!r} does not fit {lines} line(s) of {width} px")
    return out


def glyph(items: list, x: float, y: float, size: float, sw: float, ink: str, blue: str,
          node: bool = True) -> str:
    """An area glyph from internal/areas.yaml, drawn on its 24-unit grid at `size` px; with
    node=False, its strokes alone (the small marks inside the specialist card's frame)."""
    k = size / 24
    out = [f'<g transform="translate({num(x)} {num(y)}) scale({k:.4g})" fill="none" stroke="{ink}" '
           f'stroke-width="{num(sw)}" stroke-linecap="round" stroke-linejoin="round">']
    nodes = 0
    for it in items:
        dash = f' stroke-dasharray="{it["dash"]}"' if it.get("dash") else ""
        if "d" in it:
            out.append(f'<path d="{it["d"]}"{dash}/>')
        elif "c" in it:
            cx, cy, r = it["c"]
            out.append(f'<circle cx="{num(cx)}" cy="{num(cy)}" r="{num(r)}"{dash}/>')
        elif "r" in it:
            rx_, ry_, rw, rh, rr = it["r"]
            out.append(f'<rect x="{num(rx_)}" y="{num(ry_)}" width="{num(rw)}" height="{num(rh)}" '
                       f'rx="{num(rr)}"{dash}/>')
        elif "node" in it:
            nodes += 1
            if not node:
                continue
            cx, cy = it["node"]
            out.append(f'<circle cx="{num(cx)}" cy="{num(cy)}" r="2.25" fill="{blue}" stroke="none"/>')
    if nodes != 1:
        raise SystemExit(f"REFUSING: an area glyph has {nodes} blue nodes; it must have exactly one")
    return "".join(out) + "</g>"


def play(x, cy, h, fill):
    """A play triangle: left edge at x, vertically centred on cy, h tall."""
    w = h * 0.86
    return f'<path d="M{num(x)} {num(cy - h / 2)}L{num(x + w)} {num(cy)}L{num(x)} {num(cy + h / 2)}Z" fill="{fill}"/>'


def mark_state(kind: str, cx: float, cy: float, r: float, t: dict, sw: float = 1.5) -> str:
    """The progress board's three states, told apart by shape as well as colour. "Not started" is
    the hourglass, which is what the notebook's own board prints for it."""
    if kind == "pending":
        a, b = r * 0.62, r * 0.92
        d = (f"M{num(cx - a)} {num(cy - b)}H{num(cx + a)}L{num(cx)} {num(cy)}L{num(cx + a)} {num(cy + b)}"
             f"H{num(cx - a)}L{num(cx)} {num(cy)}Z")
        return path(d, t["ink"], sw, t["panel"])
    s = r * 0.42
    if kind == "done":
        d = f"M{num(cx - s)} {num(cy + 0.05 * r)}l{num(s * 0.72)} {num(s * 0.72)} {num(s * 1.35)}-{num(s * 1.5)}"
        return circle(cx, cy, r, t["done"]) + path(d, t["panel"], max(1.4, r * 0.26))
    d = f"M{num(cx - s)} {num(cy - s)}l{num(2 * s)} {num(2 * s)}M{num(cx + s)} {num(cy - s)}l-{num(2 * s)} {num(2 * s)}"
    return circle(cx, cy, r, t["fail"]) + path(d, t["panel"], max(1.4, r * 0.26))


def bars(rows: list, x: float, fill: str, h: float = 5) -> str:
    """Rounded code bars: rows of (y, [widths]) with 6 px between bars on a row."""
    out = []
    for y, widths in rows:
        cx = x
        for w in widths:
            out.append(rect(cx, y, w, h, h / 2, fill))
            cx += w + 6
    return "".join(out)


def quad_len(p0, c, p2, n=64) -> float:
    pts = [((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * c[0] + t * t * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * c[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


# --------------------------------------------------------------------------------------- assets

def lockup(svg: Svg, x0: float, y0: float, s: float, t: dict):
    """The Commons lockup of tools/brand_lockups.py, at scale s: the wordmark's geometry, then
    "Commons" in Geist 400 after 0.28 em. Returns its width."""
    X = lambda u: x0 + s * u
    Y = lambda v: y0 + s * v
    svg.add(circle(X(13), Y(24), 9 * s, t["blue"]), circle(X(32), Y(24), 5 * s, t["saffron"]),
            path(f"M{num(X(22))} {num(Y(24))}Q{num(X(27))} {num(Y(16))} {num(X(31))} {num(Y(24))}", t["head"], 2.8 * s))
    size, track = 26 * s, -0.04
    w1 = svg.text("Synapsa", "g700", size, X(52), Y(32), t["head"], track)
    w2 = svg.text("Commons", "g400", size, X(52) + w1 + track * size + 0.28 * size, Y(32), t["head"], track)
    return 52 * s + w1 + track * size + 0.28 * size + w2


def hero(t: dict) -> str:
    svg = Svg(960, 320, "Synapsa Commons: learn AI by building it")
    # An opaque panel with no hairline: the README's h1 wraps the hero, and GitHub rules every h1
    # off underneath, so a frame of the hero's own would draw a second line 20 px above that rule.
    svg.add(rect(0, 0, svg.w, svg.h, 24, t["panel"]))
    lockup(svg, 56, 52, 1.5, t)
    size, track = 56, -0.03
    for i, line in enumerate(("Learn AI", "by building it.")):
        w = svg.text(line, "g700", size, 56, 196 + 60 * i, t["head"], track)
        if 56 + w > 510:
            raise SystemExit(f"REFUSING: hero line {line!r} is {w:.0f} px and runs into the notebook")
    # the notebook: frame, toolbar, three cells, and the progress board
    x0, y0, w, h = 600, 56, 304, 208
    svg.add(rect(x0, y0, w, h, 16, t["panel"], t["ink"], 1.5),
            path(f"M{x0 + 0.75} 88H{x0 + w - 0.75}", t["line2"], 1),
            *(circle(x0 + 18 + 11 * i, 72, 3, t["line2"]) for i in range(3)))
    rows = {116: [(110.5, [56, 88]), (118.5, [40])], 158: [(152.5, [96]), (160.5, [64, 32])],
            200: [(194.5, [72, 40]), (202.5, [104])]}
    for cy, rws in rows.items():
        svg.add(rect(652, cy - 16, 236, 32, 8, t["subtle"], t["line"]), bars(rws, 664, t["bar"]))
        if cy != 116:   # an empty prompt; cell 1's is where the signal lands
            svg.add(path(f"M624 {cy - 8}h-3v16h3M636 {cy - 8}h3v16h-3", t["muted"], 1.5))
    svg.add(path(f"M{x0 + 0.75} 228H{x0 + w - 0.75}", t["line"], 1))
    for i, kind in enumerate(("done", "done", "pending")):
        cx = x0 + 24 + 84 * i
        svg.add(mark_state(kind, cx, 246, 7, t), rect(cx + 14, 243.5, 44, 5, 2.5, t["bar"]))
    # the mark at 2.8x, crossing into the work: blue node outside, saffron node on cell 1's prompt
    bx, sx, cy = 548, 630, 116
    p0, p2 = (bx + 16.8, cy), (sx - 2.8, cy)
    c = ((p0[0] + p2[0]) / 2, cy - (p2[0] - p0[0]))
    L = math.ceil(quad_len(p0, c, p2)) + 1
    svg.add(path(f"M{num(p0[0])} {cy}Q{num(c[0])} {num(c[1])} {num(p2[0])} {cy}", t["head"], 6, extra=' class="a"'),
            circle(bx, cy, 18, t["blue"], extra=' class="n"'),
            circle(sx, cy, 10, t["saffron"], extra=' class="n n2"'))
    svg.style = ("@media (prefers-reduced-motion:no-preference){"
                 f".a{{stroke-dasharray:{L};animation:c 520ms {EASE} 200ms both}}"
                 f".n{{animation:r 320ms {EASE} both}}.n2{{animation-delay:80ms}}}}"
                 f"@keyframes c{{from{{stroke-dashoffset:{L}}}to{{stroke-dashoffset:0}}}}"
                 "@keyframes r{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}")
    return svg.render()


CARD_W = 248   # three fit one row of a GitHub README with room to spare; one per row on a phone


def card(t: dict, key: str, eyebrow_s: str, title: str, sub: tuple, band) -> str:
    svg = Svg(CARD_W, 184, f"{eyebrow_s}: {title}. {' '.join(sub)}")
    panel(svg, t, 16)
    eyebrow(svg, eyebrow_s, 20, 32, t["muted"])
    svg.text(title, "g700", 20, 20, 60, t["head"], -0.015)
    for i, line in enumerate(sub):   # two short lines, broken where a reader pauses
        if faces()["g400"].width(line, 14) > CARD_W - 40:
            raise SystemExit(f"REFUSING: card line {line!r} is wider than the card")
        svg.text(line, "g400", 14, 20, 82 + 19 * i, t["muted"])
    band(svg, t)
    return svg.render()


def arc(x1, x2, y, sw, ink, lift=None):
    """A mark arc between two points on one line: control height equal to the span, as in the mark."""
    lift = (x2 - x1) if lift is None else lift
    return path(f"M{num(x1)} {num(y)}Q{num((x1 + x2) / 2)} {num(y - lift)} {num(x2)} {num(y)}", ink, sw)


def band_new(svg, t):
    xs, y = [29, 77, 125, 173, 221], 146
    for a, b in zip(xs, xs[1:]):
        ra = 9 if a == xs[0] else 5
        svg.add(arc(a + ra + 1, b - 6, y, 2, t["ink"], lift=26))
    svg.add(circle(xs[0], y, 9, t["blue"]), *(circle(x, y, 4.25, t["panel"], t["ink"], 1.5) for x in xs[1:]))


def band_engineer(svg, t):
    y = 146
    svg.add(arc(39, 91, y, 2, t["ink"], lift=36),
            path("M102 143Q138 122 184 122", t["ink"], 2), path("M102.5 146Q143 136 184 146", t["ink"], 2),
            path("M102 149Q138 170 184 170", t["ink"], 2),
            circle(29, y, 9, t["blue"]), circle(97, y, 4.25, t["panel"], t["ink"], 1.5),
            *(circle(189, yy, 4.25, t["panel"], t["ink"], 1.5) for yy in (122, 146, 170)))


def band_field(svg, t, glyphs):
    y = 146
    svg.add(arc(39, 104, y, 2, t["ink"], lift=52), rect(106, 122, 118, 48, 12, t["subtle"], t["ink"], 1.5))
    for i, items in enumerate(glyphs):
        svg.add(glyph(items, 113 + 36 * i, 132, 28, 1.3, t["ink"], t["blue"], node=False))
    svg.add(circle(29, y, 9, t["blue"]))


TILE_W, TILE_H = 152, 130   # two share a phone's row on github.com (2 x 152 + a space < 324 px)


def tile(t: dict, a: dict) -> str:
    kind = {"programme": "Programme", "flagship": "Flagship", "track": "Track"}[a["kind"]]
    svg = Svg(TILE_W, TILE_H, f"{a['name']}, {kind.lower()}")
    panel(svg, t, 16)
    svg.add(glyph(a["glyph"], 16, 16, 40, 1.4, t["ink"], t["blue"]))
    eyebrow(svg, kind, TILE_W - 16, 28, t["muted"], "end")
    lines = wrap(a["name"], "g600", 16, -0.01, TILE_W - 32, 2)
    for i, line in enumerate(lines):
        svg.text(line, "g600", 16, 16, TILE_H - 16 - 20 * (len(lines) - 1 - i), t["head"], -0.01)
    return svg.render()


def icon(t: dict, a: dict) -> str:
    svg = Svg(48, 48, a["name"])
    svg.add(rect(0.5, 0.5, 47, 47, 12, t["subtle"], t["line"]), glyph(a["glyph"], 10, 10, 28, 1.75, t["ink"], t["blue"]))
    return svg.render()


def flow(t: dict, n: int) -> str:
    caption = ("Open", "Run all", "Fill in", "Read the board")[n - 1]
    title = (f"Step {n}: {caption}. " + (
        "The notebook opens in any of four services.",
        "Every cell runs, and unfinished exercises show as not started.",
        "A filled-in exercise is checked as soon as its cell runs, with a hint to hand.",
        "One line per exercise: passed, failed or not started.")[n - 1])
    svg = Svg(160, 176, title)
    panel(svg, t, 16)
    eyebrow(svg, f"Step {n}", 16, 28, t["muted"])
    svg.text(caption, "g600", 16, 16, 156, t["head"], -0.01)
    if n == 1:      # four services, the first chosen, above the notebook it opens
        for i in range(4):
            x = 16 + 33 * i
            svg.add(rect(x + 0.5, 42.5, 28, 14, 7, t["panel"], t["blue"] if i == 0 else t["line2"]))
            svg.add(play(x + 11, 49.5, 7, t["blue"]) if i == 0 else rect(x + 8, 48, 13, 3, 1.5, t["bar"]))
        svg.add(path("M30.5 60v4", t["blue"], 1.5),
                rect(16.75, 66.75, 126.5, 60.5, 8, t["panel"], t["ink"], 1.5),
                path("M17.5 78H142.5", t["line2"], 1), *(circle(25 + 6 * i, 72.5, 1.6, t["line2"]) for i in range(3)),
                bars([(87, [44, 22]), (97, [72]), (107, [36, 30])], 26, t["bar"], 4))
    elif n == 2:    # Run all: every cell runs; each unfinished exercise is still pending
        svg.add(rect(16.75, 42.75, 126.5, 84.5, 8, t["panel"], t["ink"], 1.5),
                path("M17.5 56H142.5", t["line2"], 1), play(24, 49.4, 7, t["blue"]))
        for cy in (70, 92, 114):
            svg.add(rect(22.5, cy - 8, 94, 16, 4, t["subtle"], t["line"]), bars([(cy - 2, [40, 18])], 30, t["bar"], 4),
                    mark_state("pending", 129, cy, 5.5, t, 1.25))
    elif n == 3:    # Fill in: your line, its check, and a hint waiting
        svg.add(rect(16.5, 42.5, 127, 50, 8, t["subtle"], t["line"]),
                bars([(53, [70]), (64, [36]), (75, [52, 20])], 27, t["bar"], 5),
                path("M70 61v11", t["blue"], 2), mark_state("done", 32, 110, 8, t),
                rect(46, 107.5, 48, 5, 2.5, t["bar"]),
                path("M126 113.5h6M127 117h4", t["ink"], 1.25),
                path("M125.6 111.3c0-1.6-2.6-3-2.6-5.8a6 6 0 0 1 12 0c0 2.8-2.6 4.2-2.6 5.8z", t["ink"], 1.25))
    else:           # Read the board: passed, passed, failed, not started
        for i, (kind, w) in enumerate((("done", 84), ("done", 64), ("fail", 76), ("pending", 52))):
            cy = 50 + 21 * i
            svg.add(mark_state(kind, 26, cy, 7, t), rect(42, cy - 2.5, w, 5, 2.5, t["bar"]))
    return svg.render()


def seal(t: dict) -> str:
    svg = Svg(96, 96, "Seal: every lesson is independently reviewed before it is marked built")
    panel(svg, t, 16)
    svg.add(circle(48, 50, 30, "none", t["ink"], 1.75),
            path("M35 50.5l8.5 8.5L61.5 41", t["blue"], 3),
            path("M58.6 15.2A36 36 0 0 1 82.8 40.7", t["ink"], 2.2))
    return svg.render()


def archive(t: dict) -> str:
    svg = Svg(96, 96, "Retired archive: the 2017-2025 tutorial collection")
    panel(svg, t, 16)
    for i, (x, y) in enumerate(((32, 22), (26, 30), (20, 38))):
        svg.add(rect(x, y, 46, 34, 4, t["panel"], t["muted"], 1.5))
    svg.add(path("M28 50h22M28 58h30M28 64.5h14", t["muted"], 1.5))
    return svg.render()


def mark(dark: bool) -> str:
    """01-brand/logo-mark.svg and logo-mark-dark.svg, verbatim but for hsl() written as hex."""
    blue, saffron, ink = ("#476cff", "#ffa733", "#ffffff") if dark else ("#1a47f9", "#f99d24", "#000000")
    svg = Svg(32, 32, "Synapsa")
    svg.add(circle(11, 16, 6.5, blue), circle(24, 16, 3.6, saffron),
            f'<path d="M17 16 Q 20.5 10, 23 16" stroke="{ink}" stroke-width="2.2" fill="none" stroke-linecap="round"/>')
    return svg.render()


def chip(t: dict, service: str) -> str:
    w = math.ceil(26 + faces()["g500"].width(service, 13) + 13)
    svg = Svg(w, 28, f"Open in {service}")
    svg.add(rect(0.5, 0.5, w - 1, 27, 13.5, t["panel"], t["line2"]), play(12, 14, 9, t["blue"]))
    svg.text(service, "g500", 13, 26, 18.6, t["ink"])
    return svg.render()


def invite(t: dict) -> str:
    """The invitation banner, in the page's own register: a card like every other piece of artwork
    here, with the page's one pillow. Three lines at 56 px, so it reads as 19 px on a phone."""
    svg = Svg(960, 320, "Worked solutions and your score come with free enrolment on Synapsa")
    panel(svg, t, 24)
    s, x0, y0 = 2.0, 38, 30
    svg.add(circle(x0 + 11 * s, y0 + 16 * s, 6.5 * s, t["blue"]), circle(x0 + 24 * s, y0 + 16 * s, 3.6 * s, t["saffron"]),
            path(f"M{num(x0 + 17 * s)} {num(y0 + 16 * s)}Q{num(x0 + 20.5 * s)} {num(y0 + 10 * s)} "
                 f"{num(x0 + 23 * s)} {num(y0 + 16 * s)}", t["head"], 2.2 * s))
    size, track = 56, -0.03
    svg.text("Worked solutions and", "g700", size, 48, 150, t["head"], track)
    svg.text("your score come with", "g700", size, 48, 214, t["head"], track)
    w = svg.text("free enrolment on", "g700", size, 48, 278, t["head"], track)
    px = 48 + w + 20
    ww = faces()["g700"].width("Synapsa", size, track)
    if px + ww + 64 > 912 or w > 860:
        raise SystemExit("REFUSING: the invitation's lines are too wide")
    svg.add(rect(px, 278 - 58, ww + 64, 80, 40, t["saffron"]))
    svg.text("Synapsa", "g700", size, px + 32, 278, "#000000", track)
    return svg.render()


def roadmap(t: dict, months: list) -> str:
    """The pathway: one node per public month down a 2 px line. Done months are green with a tick,
    the current month is Synapse Blue with a flat halo, upcoming months are outlined. Month level
    only; every word comes from calendar.yaml's public lines."""
    W, R, CX, TX = 380, 26, 50, 96
    soft = "#ebefff" if t["panel"] == "#ffffff" else "#081549"
    edge = "#949494" if t["panel"] == "#ffffff" else "#757575"     # fg-subtle: 3:1 or better on the panel
    rows, top = [], 28
    for m in months:
        full = W - TX - 24
        lines = wrap(m["theme"], "g400", 17, 0, full, 4)
        for narrower in range(int(full), 80, -4):       # balance: the narrowest measure that keeps the line count
            try:
                again = wrap(m["theme"], "g400", 17, 0, narrower, len(lines))
            except SystemExit:
                break
            lines = again
        base = top + 33 + 28
        last = base + 24 * (len(lines) - 1)
        rows.append((top, lines, base))
        top = max(top + 2 * R + 34, last + 26)
    H = top - 34 + 28
    state = {"done": "done", "now": "under way now", "next": "to come"}
    svg = Svg(W, H, "Roadmap. " + " ".join(f"{m['name']}, {state[m['state']]}: {m['theme']}." for m in months).replace("..", "."))
    panel(svg, t, 16)
    for i, (m, (tp, lines, base)) in enumerate(zip(months, rows)):
        cy = tp + R
        if i + 1 < len(months):      # the connector to the next node, green while the path is walked
            nxt = rows[i + 1][0] + R
            live = m["state"] == "done"
            svg.add(path(f"M{CX} {cy + R + 6}V{nxt - R - 6}", t["done"] if live else edge, 2))
        if m["state"] == "done":
            svg.add(circle(CX, cy, R, t["done"]),
                    path(f"M{CX - 10} {cy + 1}l7 7 13-15", t["panel"], 3.5))
        elif m["state"] == "now":
            svg.add(circle(CX, cy, R + 7, soft), circle(CX, cy, R, t["blue"]))
            svg.text(f"{m['month']:02d}", "g600", 19, CX, cy + 7, t["panel"], 0, "middle")
        else:
            svg.add(circle(CX, cy, R - 1, t["panel"], edge, 2))
            svg.text(f"{m['month']:02d}", "g600", 19, CX, cy + 7, t["muted"], 0, "middle")
        w = svg.text(m["name"], "g600", 21, TX, tp + 33, t["head"] if m["state"] == "now" else t["ink"], -0.01)
        if m["state"] == "now":
            pw = faces()["g600"].width("This month", 14, 0) + 24
            px = TX + w + 12
            svg.add(rect(px + 0.5, tp + 14.5, pw, 26, 13, t["panel"], t["blue"], 1.5))
            svg.text("This month", "g600", 14, px + 12.5, tp + 32, t["blue"])
        for j, line in enumerate(lines):
            svg.text(line, "g400", 17, TX, base + 24 * j, t["ink"] if m["state"] == "now" else t["muted"])
    return svg.render()


# ----------------------------------------------------------------------------------- the areas

def public_areas() -> list:
    """Area entries from internal/areas.yaml with at least one released lesson, in map order.
    SystemExit naming any released area that has no entry."""
    import yaml
    sys.path.insert(0, str(ROOT / "internal"))
    from release import held_back
    held = held_back()
    reg = yaml.safe_load((ROOT / "internal" / "areas.yaml").read_text())
    by_id = {a["id"]: a for a in reg["areas"]}
    live = set()
    for pat in ("lessons/*/", "flagships/*/lessons/*/", "programmes/*/lessons/*/"):
        for d in ROOT.glob(pat):
            if (d / "meta.yaml").exists() and re.match(r"[A-Z]\d\d-L\d\d", d.name) and d.name[:7] not in held:
                live.add(d.name[:3])
    missing = sorted(live - set(by_id))
    if missing:
        raise SystemExit("REFUSING: released areas with no entry in internal/areas.yaml: " + ", ".join(missing))
    return sorted((by_id[i] for i in live), key=lambda a: a["order"])


def outputs() -> dict:
    """Every file this tool owns, path -> text."""
    sys.path.insert(0, str(ROOT / "internal"))
    from release import roadmap_months
    months = roadmap_months()
    areas = public_areas()
    reg = {a["id"]: a for a in areas}
    out = {}
    for name, t in THEMES.items():
        r = lambda f: README_DIR / f"{f}-{name}.svg"
        out[r("hero")] = hero(t)
        out[r("path-new")] = card(t, "new", "New to AI", "Start from zero",
                                  ("You know some Python.", "We take it from there."), band_new)
        out[r("path-engineer")] = card(t, "engineer", "ML engineer", "Build AI systems",
                                       ("Agents, RAG, serving", "and evaluation"), band_engineer)
        field = [reg[i]["glyph"] for i in ("P07", "P03", "P02") if i in reg]
        out[r("path-field")] = card(t, "field", "Domain specialist", "AI in your field",
                                    ("Regulation, finance,", "industry and science"),
                                    lambda svg, tt: band_field(svg, tt, field))
        for n, f in enumerate(("open", "run", "fill", "board"), 1):
            out[r(f"flow-{n}-{f}")] = flow(t, n)
        out[r("seal")] = seal(t)
        out[r("archive")] = archive(t)
        out[r("mark")] = mark(name == "dark")
        for s in SERVICES:
            out[r(f"open-{s.lower()}")] = chip(t, s)
        for a in areas:
            out[AREA_DIR / f"{a['id'].lower()}-tile-{name}.svg"] = tile(t, a)
            out[AREA_DIR / f"{a['id'].lower()}-icon-{name}.svg"] = icon(t, a)
        out[r("synapsa-invite")] = invite(t)
        out[r("roadmap")] = roadmap(t, months)
    return out


def owned() -> set:
    return {p for d in (README_DIR, AREA_DIR) if d.is_dir() for p in d.glob("*.svg")}


# ----------------------------------------------------------------------------------------- lint

LESSON_ID = re.compile(r"[A-Z]\d\d-L\d\d")
BANNED = re.compile(r"<(text|image|script|foreignObject|iframe|object|embed|a)\b", re.I)
COLOUR_ATTR = re.compile(r'\b(?:fill|stroke|stop-color|color)="([^"]*)"')


def lint_one(p: Path) -> list:
    raw = p.read_bytes()
    text = raw.decode("utf-8", "replace")
    why = []
    if len(raw) > MAX_BYTES:
        why.append(f"{len(raw)} bytes, over the {MAX_BYTES} limit")
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as e:
        return [f"not well-formed XML: {e}"]
    ns = "{http://www.w3.org/2000/svg}"
    if root.tag != ns + "svg" or root.get("role") != "img":
        why.append('the root is not <svg role="img">')
    title = root.find(ns + "title")
    if title is None or not (title.text or "").strip():
        why.append("no <title>")
    if BANNED.search(text):
        why.append(f"contains <{BANNED.search(text).group(1)}>")
    for m in re.finditer(r'\bhref="([^"]*)"', text):
        if not m.group(1).startswith("#"):
            why.append(f"external href {m.group(1)!r}")
    if re.search(r"url\(|@import|data:", text):
        why.append("an external or embedded resource (url(), @import or data:)")
    for v in COLOUR_ATTR.findall(text):
        if v != "none" and not re.fullmatch(r"#[0-9a-f]{6}", v):
            why.append(f"colour {v!r} is not lower-case hex")
    if re.search(r"\b(?:hsl|rgb)a?\(", text):
        why.append("hsl() or rgb() colour")
    style = "".join(s.text or "" for s in root.iter(ns + "style"))
    inside = re.search(r"@media \(prefers-reduced-motion:no-preference\)\{(.*?\})\}", style)
    rest = style.replace(inside.group(0), "") if inside else style
    if re.search(r"\banimation\s*:", rest) or re.search(r"<(animate|animateTransform|animateMotion|set)\b", text):
        why.append("animation outside a prefers-reduced-motion: no-preference query")
    if LESSON_ID.search(text):
        why.append(f"names a lesson ({LESSON_ID.search(text).group(0)})")
    return why


def lint(quiet: bool = False) -> int:
    """Print every rule a file breaks; 1 if any does. quiet: no summary line when all is well."""
    files = sorted(owned())
    bad = 0
    for p in files:
        for why in lint_one(p):
            print(f"  LINT  {p.relative_to(ROOT)}: {why}")
            bad += 1
    if bad or not quiet:
        print(f"  {len(files)} SVGs linted, {bad} problem(s)")
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------- main

def check() -> list:
    """Problems as text: missing, stale or stray files. Needs the fonts."""
    want = outputs()
    probs = [f"missing  {p.relative_to(ROOT)}" for p in want if not p.exists()]
    probs += [f"stale    {p.relative_to(ROOT)}" for p, s in want.items() if p.exists() and p.read_text() != s]
    probs += [f"stray    {p.relative_to(ROOT)}" for p in sorted(owned() - set(want))]
    return probs


def write() -> list:
    want = outputs()
    changed = []
    for p, s in want.items():
        if not p.exists() or p.read_text() != s:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s)
            changed.append(f"written  {p.relative_to(ROOT)}")
    for p in sorted(owned() - set(want)):
        p.unlink()
        changed.append(f"removed  {p.relative_to(ROOT)}")
    return changed


def main() -> int:
    if "--lint" in sys.argv:
        return lint()
    if not (ROOT / "internal" / "areas.yaml").exists():
        print("  the artwork is drawn in the maintainers' source, which holds the area registry; "
              "nothing to do here")
        return 0
    try:
        if "--check" in sys.argv:
            probs = check()
            for p in probs:
                print(f"  {p}  (run: python tools/brand_svgs.py)")
            print(f"  {len(owned())} SVGs checked, {len(probs)} problem(s)")
            return 1 if probs else lint()
        for line in write():
            print(f"  {line}")
    except FontsUnavailable as e:
        raise SystemExit(f"REFUSING: {e}")
    return lint()


if __name__ == "__main__":
    raise SystemExit(main())
