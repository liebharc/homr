import difflib

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.datasets.music_xml_parser import music_xml_file_to_tokens
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str


def validate_conversion(file: str) -> bool:
    expected = read_tokens(file)
    tmp = file.replace(".tokens", ".musicxml.tmp")

    xml = generate_xml(XmlGeneratorArguments(None, None, None), [expected], "")
    xml.write(tmp)

    actual = music_xml_file_to_tokens(tmp)
    flat_list = [x for xxs in actual for xs in xxs for x in xs]

    actual_str = token_lines_to_str(flat_list)
    expected_str = token_lines_to_str(expected)
    if actual_str != expected_str:
        actual_str = token_lines_to_str(flat_list)
        expected_str = token_lines_to_str(expected)
        eprint("====", file, "====")
        eprint(str.join("\n", difflib.ndiff(expected_str.splitlines(), actual_str.splitlines())))
        return False
    return True


if __name__ == "__main__":
    import multiprocessing
    import sys

    from homr.simple_logging import eprint

    filename = sys.argv[1]

    if filename.endswith(".tokens"):
        validate_conversion(filename)
        sys.exit(0)

    def process_file(index_entry: str) -> tuple[str, bool]:
        try:
            file = index_entry.strip().split(",")[1]
            return file, validate_conversion(file)
        except Exception as e:
            eprint(file)
            eprint(e)
            return file, False

    errors = set()
    total = 0

    index_file = open(filename)
    index_lines = index_file.readlines()
    index_file.close()

    eprint("Found", len(index_lines), "files")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for file, result in pool.imap_unordered(process_file, index_lines):
            if not result:
                errors.add(file)
            total += 1
            if total % 100 == 0 and total > 0:
                eprint(len(errors), total)

    eprint(len(errors), total)
    if len(errors) > 0:
        eprint("Files with errors")
        eprint(errors)
