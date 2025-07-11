class Symbols:
    BACKGROUND = [0]
    LEDGERLINE = [2]
    BARLINE_BETWEEN = [3]
    BARLINE_END = [4]
    ALL_BARLINES = BARLINE_BETWEEN + BARLINE_END
    REPEAT_DOTS = [7]
    G_GLEF = [10]
    C_CLEF = [11, 12]
    F_CLEF = [13]
    ALL_CLEFS = G_GLEF + C_CLEF + F_CLEF
    NUMBERS = [19, 20]
    TIME_SIGNATURE_SUBSET = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34]
    TIME_SIGNATURE = TIME_SIGNATURE_SUBSET + [31, 32]  # Oemer hasn't used these in the past
    NOTEHEAD_FULL_ON_LINE = [35]
    UNKNOWN = [
        36,
        38,
        40,
        128,
        143,
        144,
        148,
        150,
        157,
        159,
        160,
        161,
        162,
        163,
        164,
        167,
        170,
        171,
    ]
    NOTEHEAD_FULL_BETWEEN_LINES = [37]
    NOTEHEAD_HOLLOW_ON_LINE = [39]
    NOTEHEAD_HOLLOW_BETWEEN_LINE = [41]
    WHOLE_NOTE_ON_LINE = [43]
    WHOLE_NOTE_BETWEEN_LINE = [45]
    DOUBLE_WHOLE_NOTE_ON_LINE = [47]
    DOUBLE_WHOLE_NOTE_BETWEEN_LINE = [49]
    NOTEHEADS_SOLID = NOTEHEAD_FULL_ON_LINE + NOTEHEAD_FULL_BETWEEN_LINES
    NOTEHEADS_HOLLOW = NOTEHEAD_HOLLOW_ON_LINE + NOTEHEAD_HOLLOW_BETWEEN_LINE
    NOTEHEADS_WHOLE = (
        WHOLE_NOTE_ON_LINE
        + WHOLE_NOTE_BETWEEN_LINE
        + DOUBLE_WHOLE_NOTE_ON_LINE
        + DOUBLE_WHOLE_NOTE_BETWEEN_LINE
    )
    NOTEHEADS_ALL = (
        NOTEHEAD_FULL_ON_LINE
        + NOTEHEAD_FULL_BETWEEN_LINES
        + NOTEHEAD_HOLLOW_ON_LINE
        + NOTEHEAD_HOLLOW_BETWEEN_LINE
        + WHOLE_NOTE_ON_LINE
        + WHOLE_NOTE_BETWEEN_LINE
        + DOUBLE_WHOLE_NOTE_ON_LINE
        + DOUBLE_WHOLE_NOTE_BETWEEN_LINE
    )
    DOT = [51]
    STEM = [52]
    TREMOLO = [53, 54, 55, 56]
    FLAG_DOWN = [58, 60, 61, 62, 63]
    FLAG_UP = [64, 66, 67, 68, 69]
    FLAT = [70]
    NATURAL = [72]
    SHARP = [74]
    DOUBLE_SHARP = [76]
    ALL_ACCIDENTALS = FLAT + NATURAL + SHARP + DOUBLE_SHARP
    KEY_FLAT = [78]
    KEY_NATURAL = [79]
    KEY_SHARP = [80]
    ALL_KEYS = KEY_FLAT + KEY_NATURAL + KEY_SHARP
    ACCENT_ABOVE = [81]
    ACCENT_BELOW = [82]
    STACCATO_ABOVE = [83]
    STACCATO_BELOW = [84]
    TENUTO_ABOVE = [85]
    TENUTO_BELOW = [86]
    STACCATISSIMO_ABOVE = [87]
    STACCATISSIMO_BELOW = [88]
    MARCATO_ABOVE = [89]
    MARCATO_BELOW = [90]
    FERMATA_ABOVE = [91]
    FERMATA_BELOW = [92]
    BREATH_MARK = [93]
    REST_LARGE = [95]
    REST_LONG = [96]
    REST_BREVE = [97]
    REST_FULL = [98]
    REST_QUARTER = [99]
    REST_EIGHTH = [100]
    REST_SIXTEENTH = [101]
    REST_THIRTY_SECOND = [102]
    REST_SIXTY_FOURTH = [103]
    REST_ONE_HUNDRED_TWENTY_EIGHTH = [104]
    ALL_RESTS_EXCEPT_LARGE = (
        REST_LONG
        + REST_BREVE
        + REST_FULL
        + REST_QUARTER
        + REST_EIGHTH
        + REST_SIXTEENTH
        + REST_THIRTY_SECOND
        + REST_SIXTY_FOURTH
        + REST_ONE_HUNDRED_TWENTY_EIGHTH
    )
    ALL_RESTS = ALL_RESTS_EXCEPT_LARGE
    TRILL = [127]
    GRUPPETO = [129]
    MORDENT = [130]
    DOWN_BOW = [131]
    UP_BOW = [132]
    SYMBOL = [133, 134, 135, 138, 139, 141, 142]
    TUPETS = [136, 137, 149, 151, 152, 153, 154, 155, 156]
    SLUR_AND_TIE = [145, 147]
    BEAM = [146]
    STAFF = [165]
    BRACKETS = [1]


DENSE_DATASET_DEFINITIONS = Symbols()

CLASS_CHANNEL_LIST = [
    DENSE_DATASET_DEFINITIONS.STEM
    + DENSE_DATASET_DEFINITIONS.BARLINE_BETWEEN
    + DENSE_DATASET_DEFINITIONS.BARLINE_END,
    DENSE_DATASET_DEFINITIONS.NOTEHEADS_ALL,
    DENSE_DATASET_DEFINITIONS.ALL_CLEFS
    + DENSE_DATASET_DEFINITIONS.ALL_KEYS
    + DENSE_DATASET_DEFINITIONS.ALL_ACCIDENTALS,
    DENSE_DATASET_DEFINITIONS.STAFF,
    DENSE_DATASET_DEFINITIONS.BRACKETS,
]

CLASS_CHANNEL_MAP = {
    color: idx + 1 for idx, colors in enumerate(CLASS_CHANNEL_LIST) for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 1  # Plus 'background' channel.
