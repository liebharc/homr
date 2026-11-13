from homr.transformer.vocabulary import EncodedSymbol, empty


def strip_naturals(
    symbols: list[EncodedSymbol],
) -> list[EncodedSymbol]:
    results = []
    for symbol in symbols:
        if symbol.lift == "N":
            results.append(symbol.change_lift(empty))
        else:
            results.append(symbol)

    return results
