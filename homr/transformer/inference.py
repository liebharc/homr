from .configs import default_config
from .staff2score import Staff2Score


class Inference:
    def __init__(self) -> None:
        self.config = default_config
        self.handler = Staff2Score(self.config)

    def predict(self, filepath: str) -> list[str]:
        predrhythms, predpitchs, predlifts = self.handler.predict(filepath)

        merges = []
        for i in range(len(predrhythms)):
            predlift = predlifts[i]
            predpitch = predpitchs[i]
            predrhythm = predrhythms[i]

            if len(predrhythm) == 0:
                merges.append("")
                continue

            merge = predrhythm[0] + "+"
            for j in range(1, len(predrhythm)):
                if predrhythm[j] == "|":
                    merge = merge[:-1] + predrhythm[j]
                elif "note" in predrhythm[j]:
                    lift = ""
                    if predlift[j] in (
                        "lift_##",
                        "lift_#",
                        "lift_bb",
                        "lift_b",
                        "lift_N",
                    ):
                        lift = predlift[j].split("_")[-1]
                    merge += predpitch[j] + lift + "_" + predrhythm[j].split("note-")[-1] + "+"
                else:
                    merge += predrhythm[j] + "+"
            merges.append(merge[:-1])
        return merges
