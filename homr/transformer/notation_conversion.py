import sys
from homr.simple_logging import eprint
from homr.transformer.clef_conversion import to_agnostic_pitch, from_agnostic_pitch
from homr.transformer.lift_conversion import to_agnostic_lift, from_agnostic_lift
from homr.transformer.vocabulary import EncodedSymbol
from homr.circle_of_fifths import strip_naturals

def to_transformer_format(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts standard semantic notation to the transformer's internal agnostic format.
    """
    # Order matters: lift depends on absolute pitch, so we do lift first
    symbols = to_agnostic_lift(symbols)
    symbols = to_agnostic_pitch(symbols)
    return symbols

def is_too_many_ledger_lines(tokens: list[EncodedSymbol]) -> bool:
    """
    Returns True if any pitch in the token stream requires more than 5 ledger lines.
    (LL11+ or LH11+).
    """
    for symbol in tokens:
        if symbol.pitch.startswith(("LL", "LH")):
            try:
                num = int(symbol.pitch[2:])
                if num > 10:
                    return True
            except ValueError:
                pass
    return False


def from_transformer_format(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts the transformer's internal agnostic format back to standard semantic notation.
    """
    # Order matters: restore pitch first so lift conversion has the absolute pitch
    symbols = from_agnostic_pitch(symbols)
    symbols = from_agnostic_lift(symbols)
    return symbols

def verify_file(tokens_file: str) -> tuple[str, str, str]:
    """
    Verifies a single tokens file by performing a round-trip conversion.
    Returns (file_path, success, error_message).
    """
    try:
        from training.transformer.training_vocabulary import read_token_lines
        with open(tokens_file, encoding="utf-8") as f:
            lines = f.readlines()
            original_symbols = read_token_lines(lines)
        
        # Check for excessive ledger lines
        internal = to_transformer_format(original_symbols)
        if is_too_many_ledger_lines(internal):
            return (tokens_file, "SKIP", "Too many ledger lines")

        # Round trip
        restored = from_transformer_format(internal)
        
        # Normalize original for comparison (empty means natural)
        original_symbols = strip_naturals(original_symbols)
        
        # We compare strings to avoid precision issues if any
        original_str = "\n".join([str(s) for s in original_symbols])
        restored_str = "\n".join([str(s) for s in restored])
        
        if original_str == restored_str:
            return (tokens_file, "OK", "")
        else:
            import difflib
            diff = difflib.unified_diff(
                original_str.splitlines(),
                restored_str.splitlines(),
                fromfile="original",
                tofile="restored",
            )
            return (tokens_file, "FAIL", "\n".join(diff))
    except Exception as e:
        return (tokens_file, "FAIL", str(e))

if __name__ == "__main__":
    import glob
    import multiprocessing
    import os
    
    # Mock torch if not present to allow running without full environment
    try:
        import torch
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock()

    if len(sys.argv) > 1:
        tokens_file = sys.argv[1]
        file, status, msg = verify_file(tokens_file)
        if status == "OK":
            eprint(f"Validation successful: {file} is consistent.")
        elif status == "SKIP":
            eprint(f"Validation skipped: {file} - {msg}")
        else:
            eprint(f"Validation failed: {file}!")
            eprint(msg)
            sys.exit(1)
    else:
        files = glob.glob(os.path.join("datasets", "**", "*.tokens"), recursive=True)
        files = sorted(files)
        total_files = len(files)
        eprint(f"Verifying {total_files} files...")

        failed_files = 0
        skipped_files = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for i, (file, status, msg) in enumerate(pool.imap_unordered(verify_file, files), 1):
                if status == "FAIL":
                    eprint(f"\nFAILED: {file}")
                    eprint(msg)
                    failed_files += 1
                elif status == "SKIP":
                    skipped_files += 1
                
                if i % 100 == 0:
                    eprint(f"Progress: {i}/{total_files} verified. Skipped: {skipped_files}. Failed: {failed_files}...", end="\r", flush=True)

        eprint(f"\nVerification complete. {total_files} files processed.")
        eprint(f"Success: {total_files - failed_files - skipped_files}")
        eprint(f"Skipped: {skipped_files}")
        eprint(f"Failed: {failed_files}")
        
        if failed_files > 0:
            sys.exit(1)
