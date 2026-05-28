Learnings:
- Corpus contains slurs which are not used. For example: homr/datasets/Corpus/000051662-1_1_2
- Ties have their own rhythm-symbol
- The tests/test_music_xml_parser.py:453 currently fails. The current implementation puts the "staccato" into the first note instead of the last one:
      is:           - note_12 G5 _ _ _ upper&note_12 D5 # staccato _ upper
E           ?               --                    ---------
expected:           + note_12 G5 _ staccato _ upper&note_12 D5 # _ _ upper
