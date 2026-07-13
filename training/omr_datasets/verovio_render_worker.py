import json
import sys

from homr.simple_logging import eprint


def render(xml_str: str, scale: int, font: str, mnum_interval: int) -> str | None:
    import verovio  # noqa: PLC0415 - lazy: must not be loaded in the parent training process

    try:
        tk = verovio.toolkit()
        tk.setOptions(
            {
                "breaks": "none",
                "adjustPageWidth": True,
                "adjustPageHeight": True,
                "scale": scale,
                "font": font,
                "mnumInterval": mnum_interval,
            }
        )
        if not tk.loadData(xml_str):
            return None
        return tk.renderToSVG(1)
    except Exception as e:
        eprint("Verovio rendering failed:", e)
        return None


def main() -> None:
    request = json.loads(sys.stdin.readline())
    svg = render(request["xml_str"], request["scale"], request["font"], request["mnum_interval"])
    sys.stdout.write(json.dumps({"svg": svg}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
