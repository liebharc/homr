import converter21
from music21 import converter


def mei_to_kern(mei_file: str) -> str:
    converter21.register()
    kern_file = mei_file.replace(".mei", ".krn")
    score_stream = converter.parse(mei_file, format="mei")
    score_stream.write(fmt="humdrum", fp=kern_file, makeNotation=False)
    return kern_file


def kern_to_musicxml(kern_file: str) -> str:
    converter21.register()
    music_xml_file = kern_file.replace(".krn", ".musicxml")
    score_stream = converter.parse(kern_file)
    score_stream.write("musicxml", fp=music_xml_file)
    return music_xml_file


def musicxml_to_kern(music_xml_file: str) -> str:
    converter21.register()
    kern_file = music_xml_file.replace(".musicxml", ".krn")
    score_stream = converter.parse(music_xml_file)
    score_stream.write("humdrum", fp=kern_file)
    return kern_file


def fix_utf8_encoding(file: str) -> None:
    with open(file, "rb") as f:
        raw_data = f.read()
        decoded = raw_data.decode("utf-8-sig")  # auto-handles BOM if present

    with open(file, "w", encoding="utf-8") as f:
        f.write(decoded)


if __name__ == "__main__":
    # ruff: noqa: T201
    import sys

    mei_file = sys.argv[1]
    kern_file = mei_to_kern(mei_file)
    print(kern_file)
    sys.exit(0)
    music_xml_file = kern_to_musicxml(kern_file)
    print(music_xml_file)
    print(musicxml_to_kern(music_xml_file))
