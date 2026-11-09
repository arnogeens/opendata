#!/usr/bin/env python3
"""
Build the live global cloud map (8192x4096 greyscale, equirectangular) from
EUMETSAT's public WMS, and the latest.json manifest that points the
simulators at it.

The image recipe is a port of matteason/live-cloud-maps (index.js, CC0):
  - IR 10.8 um world cloud map, levels 72..178, gamma 1.46
  - Dust RGB: red masked by inverted blue, screened with half the inverted blue
  - screen(dust, 0.77 * IR), gamma 2.0
  - natural-colour RGB lifts grey/white (cloud-like) pixels in daylight,
    faded out near the antimeridian
  - gaps (runs of pure white = no data) filled with a horizontal gradient
  - poles: outermost 1/8 mirrored, then covered by a soft grey cap
    (his frame.png, reproduced here as a latitude profile)

Only the cloud map is built here; day, night and specular maps are static
(see build_static_earth_textures.py).

Source data: "Contains modified EUMETSAT data".
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WIDTH, HEIGHT = 8192, 4096
TILE_DEG = 90           # tiles of 90 x 90 deg; the WMS truncates larger requests
TILE_PX = 2048
WMS = "https://view.eumetsat.int/geoserver/ows"
LAYERS = {
    "ir": "mumi:worldcloudmap_ir108",
    "dust": "mumi:wideareacoverage_rgb_dust",
    "visible": "mumi:wideareacoverage_rgb_natural",
}
# The server encodes full PNGs slowly. The IR layer is grey, so 8-bit PNG is
# effectively lossless for it (0.0002% px differ, by <= 4) and much faster;
# the colour layers need full PNG (8-bit or JPEG breaks the recipe's colour tests).
FORMATS = {"ir": "image/png8", "dust": "image/png", "visible": "image/png"}
WORKERS = 12
USER_AGENT = "spacesimulator-cloud-pipeline/1.0"
STAGE_ID = f"{WIDTH}x{HEIGHT}"

# Polar cap (matteason's frame.png): grey 232, opaque down to ~1.5% of the
# height, half at ~4.6%, gone by ~7.8% (measured from the image).
CAP_GREY = 232.0
CAP_FULL_ROW = 60 / 4096
CAP_CLEAR_ROW = 320 / 4096


# --------------------------------------------------------------------- fetch

def http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        if res.status != 200:
            raise RuntimeError(f"HTTP {res.status}")
        return res.read()


def latest_time(layer: str) -> datetime:
    """Newest time step a layer offers, from its WMS capabilities."""
    name = layer.split(":", 1)[1]
    url = (f"https://view.eumetsat.int/geoserver/mumi/{name}/ows"
           "?service=WMS&request=GetCapabilities&version=1.3.0")
    xml = http_get(url, 60).decode("utf-8", "replace")
    m = re.search(r'<Dimension name="time"[^>]*>([^<]+)<', xml)
    if not m:
        raise RuntimeError(f"no time dimension for {layer}")
    last = m.group(1).strip().split(",")[-1]
    # Either a plain list of instants or start/end/period.
    stamp = last.split("/")[1] if "/" in last else last
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def common_time() -> datetime:
    """Newest time step that all three layers have, so they match."""
    return min(latest_time(layer) for layer in LAYERS.values())


def fetch_tile(layer: str, fmt: str, when: datetime, row: int, col: int) -> np.ndarray:
    lat0 = 90 - (row + 1) * TILE_DEG
    lon0 = -180 + col * TILE_DEG
    # WMS 1.3.0 with EPSG:4326 takes the bbox as lat,lon.
    bbox = f"{lat0},{lon0},{lat0 + TILE_DEG},{lon0 + TILE_DEG}"
    url = (f"{WMS}?service=WMS&request=GetMap&version=1.3.0&layers={layer}&styles="
           f"&format={fmt}&crs=EPSG:4326&bbox={bbox}&width={TILE_PX}&height={TILE_PX}"
           f"&time={when.strftime('%Y-%m-%dT%H:%M:%S.000Z')}")
    last_err: Exception | None = None
    for attempt in range(3):
        t0 = time.time()
        try:
            data = http_get(url, 300)
            with Image.open(io.BytesIO(data)) as im:
                im.load()  # fails on a truncated PNG
                if im.size != (TILE_PX, TILE_PX):
                    raise RuntimeError(f"unexpected size {im.size}")
                print(f"  {layer} r{row} c{col}: {len(data) / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
                return np.asarray(im.convert("RGB"))
        except Exception as err:  # noqa: BLE001 - retry anything
            print(f"  {layer} r{row} c{col}: attempt {attempt + 1} failed after {time.time() - t0:.0f} s: {err}", flush=True)
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"tile {layer} r{row} c{col} failed: {last_err}")


def fetch_layers(when: datetime) -> dict[str, np.ndarray]:
    """All tiles of all layers at once, slowest layer (dust) first."""
    rows, cols = 180 // TILE_DEG, 360 // TILE_DEG
    order = ["dust", "visible", "ir"]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {(k, r, c): pool.submit(fetch_tile, LAYERS[k], FORMATS[k], when, r, c)
                for k in order for r in range(rows) for c in range(cols)}
        out = {k: np.empty((HEIGHT, WIDTH, 3), np.uint8) for k in order}
        for (k, r, c), job in jobs.items():
            out[k][r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX] = job.result()
    return out


# ------------------------------------------------------------------- process

def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1 - (1 - a / 255) * (1 - b / 255)) * 255


def gamma(g: float, x: np.ndarray) -> np.ndarray:
    return 255 * np.power(np.clip(x, 0, None) / 255, 1 / g)


def ramp(a: float, b: float, lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0, 1)
    return lo + (hi - lo) * t


def fill_gaps(img: np.ndarray) -> None:
    """Fill runs of pure white (no data) with a gradient, in place.

    Same scan as matteason's: per row (poles skipped), start 3 px before the
    run, end 4 px after it.
    """
    h, w = img.shape
    buf = 3
    for y in range(h // 8 + 1, h - h // 8):
        row = img[y]
        white = row == 255
        if not white.any():
            continue
        x = 0
        while x < w:
            if not white[x]:
                nxt = np.flatnonzero(white[x:])
                if nxt.size == 0:
                    break
                x += int(nxt[0])
            start = x - buf
            start_val = float(row[x - buf]) if x - buf >= 0 else float(row[x])
            # Skip the run, then 3 more non-white pixels.
            seen = 0
            x += 1
            while x < w and (white[x] or seen < buf):
                if not white[x]:
                    seen += 1
                x += 1
            if x >= w:
                break  # gap runs off the row: leave it (as the original does)
            end_val = float(row[x])
            xs = np.arange(max(start, 0), x + 1)
            step = (end_val - start_val) / (x - start)
            row[xs] = np.clip(np.floor(start_val + (xs - start) * step), 0, 255).astype(np.uint8)
            white = row == 255
            x += 1


def build_cloud_map(ir: np.ndarray, dust: np.ndarray, vis: np.ndarray) -> np.ndarray:
    w = WIDTH
    # Work with the halves swapped (antimeridian in the middle), like the original.
    ir, dust, vis = (np.roll(a, w // 2, axis=1).astype(np.float32) for a in (ir, dust, vis))

    dust_r = dust[..., 0]
    dust_b = 255 - dust[..., 2]
    dust_v = screen(dust_r / 255 * dust_b / 255 * 255, 0.5 * dust_b)
    ir_v = gamma(1.46, ramp(72, 178, 0, 255, ir[..., 0]))
    out = gamma(2.0, screen(dust_v, ir_v * 0.77))

    vr, vg, vb = vis[..., 0], vis[..., 1], vis[..., 2]
    vmax = np.maximum(np.maximum(vr, vg), vb)
    vmin = np.minimum(np.minimum(vr, vg), vb)
    greyish = ((vmax - vmin) < 25) | ((vr < vg) & (np.abs(vg - vb) < 11) & (vg > 150))
    vis_v = gamma(1.5, vmax)
    x = np.arange(w, dtype=np.float32)[None, :]
    fade = np.abs(ramp(0.4 * w, 0.6 * w, -1, 1, x))
    in_band = (x > 0.4 * w) & (x < 0.6 * w) & (vis_v < 255)
    vis_v = np.where(in_band, vis_v * fade, vis_v)
    out = np.where(greyish, np.maximum(vis_v, out), out)

    img = np.clip(np.floor(out), 0, 255).astype(np.uint8)
    fill_gaps(img)
    img = np.roll(img, -(w // 2), axis=1)

    # Poles: mirror the band next to each cap over the cap.
    band = HEIGHT // 8
    img[:band] = img[band:2 * band][::-1]
    img[-band:] = img[-2 * band:-band][::-1]

    # Soft grey polar cap.
    rows = (np.arange(HEIGHT, dtype=np.float32) + 0.5) / HEIGHT
    from_pole = np.minimum(rows, 1 - rows)
    t = np.clip((from_pole - CAP_FULL_ROW) / (CAP_CLEAR_ROW - CAP_FULL_ROW), 0, 1)
    alpha = (1 - t * t * (3 - 2 * t))[:, None]
    return np.clip(np.floor(img * (1 - alpha) + CAP_GREY * alpha), 0, 255).astype(np.uint8)


def check_plausible(img: np.ndarray) -> None:
    """Refuse to publish an obviously broken map (blank, all white, ...)."""
    mid = img[HEIGHT // 8: -HEIGHT // 8]
    cloudy = float((mid > 128).mean())
    white = float((mid == 255).mean())
    if not 0.05 < cloudy < 0.85 or white > 0.15:
        raise SystemExit(f"implausible cloud map: cloudy={cloudy:.2f} white={white:.2f}")


# -------------------------------------------------------------------- output

def build_manifest(source_time: datetime, static_base: str) -> dict:
    static = static_base.rstrip("/")
    stamp = source_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourceTime": stamp,
        "snapshotDate": source_time.strftime("%Y-%m-%d"),
        "attribution": "Contains modified EUMETSAT data",
        "stages": [{
            "id": STAGE_ID,
            "day": f"{static}/earth.webp",
            # The file name never changes; the query skips stale CDN copies.
            "cloudAlpha": f"./clouds.webp?v={stamp.replace(':', '')}",
            "night": f"{static}/earth-night.webp",
            "specular": f"{static}/specular.webp",
        }],
    }


def previous_source_time(where: str | None) -> str | None:
    if not where:
        return None
    try:
        raw = http_get(where, 30) if where.startswith("http") else Path(where).read_bytes()
        return json.loads(raw).get("sourceTime")
    except Exception:  # noqa: BLE001 - no previous manifest is fine
        return None


def set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", required=True, help="Where clouds.webp and latest.json are written.")
    p.add_argument("--static-base", required=True, help="URL of the folder with the static day/night/specular maps.")
    p.add_argument("--previous-manifest", help="URL or path of the published latest.json; skip if its sourceTime is current.")
    p.add_argument("--time", help="Source time (ISO, UTC) instead of the newest available.")
    p.add_argument("--from-dir", help="Read ir.png, dust.png, visible.png (8192x4096) instead of downloading.")
    p.add_argument("--save-sources", action="store_true", help="Also write the downloaded layers to the output dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_dir:
        src = Path(args.from_dir)
        layers = {k: np.asarray(Image.open(src / f"{k}.png").convert("RGB")) for k in LAYERS}
        when = datetime.fromisoformat(args.time.replace("Z", "+00:00")) if args.time else datetime.now(timezone.utc)
    else:
        print("checking EUMETSAT for the newest time step", flush=True)
        when = datetime.fromisoformat(args.time.replace("Z", "+00:00")) if args.time else common_time()
        stamp = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        if previous_source_time(args.previous_manifest) == stamp:
            print(f"cloud map already at {stamp}; nothing to do")
            set_output("changed", "false")
            return
        print(f"building cloud map for {stamp}")
        t0 = time.time()
        layers = fetch_layers(when)
        print(f"downloaded {len(LAYERS) * 8} tiles in {time.time() - t0:.0f} s", flush=True)
        if args.save_sources:
            for k, a in layers.items():
                Image.fromarray(a).save(out_dir / f"{k}.png")

    img = build_cloud_map(layers["ir"], layers["dust"], layers["visible"])
    check_plausible(img)
    Image.fromarray(img).convert("RGB").save(out_dir / "clouds.webp", format="WEBP", quality=90, method=6)
    manifest = build_manifest(when, args.static_base)
    (out_dir / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'clouds.webp'} for {manifest['sourceTime']}")
    set_output("changed", "true")
    set_output("source_time", manifest["sourceTime"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
