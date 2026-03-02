#!/usr/bin/env python3
import argparse
import http.server
import os
import shutil
import socket
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OSMD Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.4/build/opensheetmusicdisplay.min.js"></script>
    <style>
      body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; }}
      header {{ padding: 12px 16px; background: #f5f5f5; border-bottom: 1px solid #ddd; }}
      #osmd-container {{ padding: 16px; }}
    </style>
  </head>
  <body>
    <header>
      <strong>OSMD Preview</strong> — {filename}
    </header>
    <div id="osmd-container"></div>
    <script>
      const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd-container", {{
        backend: "svg",
        drawingParameters: "default",
        autoBeam: false,
        autoBeamOptions: {{
          beam_rests: false,
          beam_middle_rests_only: false,
          maintain_stem_directions: false,
          groups: [[3,4],[1,4]], // example, see below
        }}
      }});
      osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = 4;
      osmd
        .load("{score_file}")
        .then(() => osmd.render())
        .catch((err) => {{
          const container = document.getElementById("osmd-container");
          container.innerHTML = "<pre style='color:#b00'>" + err + "</pre>";
        }});
      window.addEventListener("resize", () => osmd.render());
    </script>
  </body>
</html>
"""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a MusicXML file in OSMD using a local HTTP server."
    )
    parser.add_argument("musicxml", help="Path to a .musicxml/.xml/.mxl file")
    parser.add_argument("--port", type=int, default=0, help="Port for the HTTP server")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open a browser window automatically"
    )
    args = parser.parse_args()

    source = Path(args.musicxml).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"File not found: {source}")

    temp_dir = Path(tempfile.mkdtemp(prefix="osmd_preview_"))
    score_ext = source.suffix if source.suffix else ".musicxml"
    score_file = f"score{score_ext}"
    shutil.copy2(source, temp_dir / score_file)

    (temp_dir / "index.html").write_text(
        HTML_TEMPLATE.format(filename=source.name, score_file=score_file), encoding="utf-8"
    )

    port = args.port or _find_free_port()
    os.chdir(temp_dir)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port), http.server.SimpleHTTPRequestHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/index.html"
    print(f"Serving {source} at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
