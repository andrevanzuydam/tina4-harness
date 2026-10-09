"""Let the harness "see" the screen — snapshot a window and describe it with the vision model.

A hack, by design: the kernel runs on the Mac, so it can take a real screenshot and feed it to the
same shared vision model the chat uses. Point it at the window that is showing JupyterLab (or the
tina4 App Preview) and it answers "what's on screen right now?".

    from harness.snapshot import snapshot
    print(snapshot("What is the user doing?"))        # finds the harness window, captures, describes
    print(snapshot("Does the page render correctly?", app="Safari"))

Which part of the screen is the harness? It finds the window by title ("JupyterLab"/".ipynb"), and
once macOS Screen Recording permission is granted those titles are readable; otherwise it falls back
to the front-most browser window, or an explicit ``app=`` / ``window_id=``. Capturing a window's
CONTENTS needs Screen Recording permission (System Settings → Privacy & Security → Screen Recording)
— the first call triggers the prompt; grant it to the app that launched the kernel.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

# Apps that commonly host JupyterLab in a browser window.
_BROWSER_OWNERS = ("Safari", "Google Chrome", "Chromium", "Firefox", "Arc", "Brave Browser",
                   "Microsoft Edge", "Claude")
_TITLE_HINTS = ("jupyterlab", "jupyter", ".ipynb", "localhost:8890", "127.0.0.1:8890")


def _windows() -> list[dict]:
    """On-screen windows (id, owner, title, bounds), biggest first. Needs pyobjc-framework-Quartz."""
    import Quartz

    opts = (Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements)
    out = []
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        b = w.get("kCGWindowBounds", {}) or {}
        out.append({
            "id": int(w.get("kCGWindowNumber", 0)),
            "owner": w.get("kCGWindowOwnerName", "") or "",
            "title": w.get("kCGWindowName", "") or "",
            "w": int(b.get("Width", 0)), "h": int(b.get("Height", 0)),
            "x": int(b.get("X", 0)), "y": int(b.get("Y", 0)),
        })
    out.sort(key=lambda d: d["w"] * d["h"], reverse=True)
    return out


def find_harness_window(hint: str | None = None, app: str | None = None):
    """The window showing the harness (JupyterLab). Match order: explicit ``app`` owner → title hint
    → front-most browser window. Returns the window dict, or None."""
    wins = [w for w in _windows() if w["w"] > 300 and w["h"] > 200]
    if app:
        for w in wins:
            if app.lower() in w["owner"].lower():
                return w
    hints = [hint.lower()] if hint else list(_TITLE_HINTS)
    for w in wins:
        t = w["title"].lower()
        if t and any(h in t for h in hints):
            return w
    for w in wins:
        if any(o.lower() in w["owner"].lower() for o in _BROWSER_OWNERS):
            return w
    return wins[0] if wins else None


class PermissionError_(RuntimeError):
    """Raised when macOS Screen Recording permission blocks the capture."""


def capture(window_id: int | None = None, region: str | None = None, path: str | Path | None = None) -> Path:
    """Screenshot a window (by id) or a region ('x,y,w,h'), or the full screen. Returns the PNG path.
    Raises PermissionError_ if Screen Recording permission is not granted."""
    path = Path(path) if path else Path(tempfile.gettempdir()) / "harness-snap.png"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    cmd = ["screencapture", "-x", "-o"]            # -x: no sound, -o: no window shadow
    if window_id:
        cmd += ["-l", str(window_id)]
    elif region:
        cmd += ["-R", region]
    cmd.append(str(path))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    # A denied window capture exits non-zero ("could not create image from window"), or writes a
    # tiny/empty file. Either way, it's the Screen Recording permission.
    if p.returncode != 0 or not path.is_file() or path.stat().st_size < 2000:
        raise PermissionError_(
            "macOS blocked the screenshot — grant Screen Recording permission to the app that "
            "launched this kernel (System Settings → Privacy & Security → Screen Recording), then "
            "restart it. Detail: " + (p.stderr.strip() or "empty capture")
        )
    return path


def snapshot(prompt: str = "Describe what is on the screen.", app: str | None = None,
             hint: str | None = None, window_id: int | None = None, region: str | None = None,
             show: bool = True, max_tokens: int = 384) -> str:
    """Capture the harness window (or a given app/window/region) and describe it with the vision model.

    ``app`` picks the hosting app by name (e.g. 'Safari'); ``hint`` matches the window title;
    ``window_id``/``region`` are explicit. ``show=True`` also displays the screenshot in the notebook.
    """
    win = None
    if window_id is None and region is None:
        win = find_harness_window(hint=hint, app=app)
        if win is None:
            return ("Couldn't find a window to snapshot. Pass app=\"Safari\" (the browser showing "
                    "JupyterLab), a window_id, or a region 'x,y,w,h'.")
        window_id = win["id"]
    try:
        img = capture(window_id=window_id, region=region)
    except PermissionError_ as exc:
        return str(exc)
    if show:
        try:
            from IPython.display import Image, display

            display(Image(filename=str(img)))
        except Exception:
            pass

    from .lab import get_model

    model = get_model()
    if not hasattr(model, "describe"):
        return "The current model has no vision support (describe())."
    where = f" ({win['owner']})" if win else ""
    return model.describe(str(img), prompt, max_tokens=max_tokens) + (f"\n\n— snapshot of{where}" if win else "")
