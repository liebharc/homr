from homr.model import Staff
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff: Staff, staff_image: NDArray) -> list[EncodedSymbol]:
    return predict_best(staff_image, staff=staff)


def predict_best(org_image: NDArray, staff: Staff) -> list[EncodedSymbol]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score()

    return inference.predict(org_image)
