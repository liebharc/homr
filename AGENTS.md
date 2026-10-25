# AGENTS.md

Guidance for AI agents, and for the people who work with them, on `homr`.

## Respect the reader's time

`homr` is a volunteer project. AI is welcome, but it must not shift the work to us.
Generating a plausible issue or analysis takes minutes. Reading, checking and
understanding it takes us hours. Do the checking and condensing yourself before
you post.

- Write briefly and plainly, without slang or filler. Every word takes a human's time.
- Support your claims. Add support only if it earns its place: check that it really
  supports the claim and that you verified it.
- The best support is real images, real cases and benchmark results.

## What counts as progress

- Use real images: scans or photos of actual sheet music. Synthetic or AI generated
  images are no evidence for a bug or an improvement.
- It's easy to optimize an OMR system for a handful of examples. Such changes don't
  bring the project forward.
- The benchmarks are a check, not a target. Our goal is to work well on diverse sheet
  music, scanned or photographed with a smartphone, not to be the best at a benchmark.
  Make a change for real images, then use the benchmarks to show that it holds up on
  music it wasn't made for.
- Polish scores is one benchmark of rather clean scans. A good result there is
  required, but it doesn't prove that a change helps real world use.

## Required for every PR that changes `homr/` or the models

1. Run the Polish scores benchmark on `main` and on your change. Put both results
   and the list of changed pages into the PR:
   ```
   poetry run python -m validation.polish-scores --tool homr --output polish-main.db  # on main
   poetry run python -m validation.polish-scores --tool homr --output polish-pr.db    # on your change
   poetry run python -m validation.compare_benchmarks polish-pr.db --base polish-main.db
   ```
2. Add the real images your change targets, with the results before and after.
3. Hard rules:
   - Original scans and real photos must not get worse. A better average doesn't
     excuse a worse page.
   - Never tune or train on evaluation data: Polish scores, SMB and real photos.
     Develop on other data, for example the OpenScore Lieder corpus.

Maintainers also check SMB and a private set of real photos.

## Issues and PRs

Issues are for problems and need a real image. Solutions go into PRs, with or without
an issue. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Working in the repo

- Set up with `make init`. Run `make ci` before every commit.
- `homr --init` downloads the model weights, they aren't in git.
- Pipeline: segmentation (`homr/segmentation`) → staff detection → one transformer
  run per staff (`homr/transformer`) → MusicXML (`homr/music_xml_generator.py`).
  The entry point is `homr/main.py`.
- Benchmark details: [Benchmark.md](Benchmark.md).
