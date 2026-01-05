# Goals

This document defines the project scope. It clarifies what the project targets and explicitly states what is out of scope to maintain focus.

## Western Sheet Music for Pitched Instruments

The project focuses on Western sheet music for pitched instruments. In practice, the training data primarily consists of piano music written in treble and bass clef, so performance is expected to be strongest in this domain.

**Out of scope:**

- Unpitched instruments (e.g., drums).
- Non-Western or alternative music notation systems. Contributions extending support are welcome.
- Lyrics. Supporting multilingual lyrics is considered out of scope for this project. However, integrations with external OMR software that handle lyrics may be considered via contributions.

## Processing Camera-Captured Sheet Music

The project aims to convert photographs of sheet music taken with smartphone cameras into machine-readable MusicXML.

**Also supported:**

- Scanned sheet music images in PNG or JPG format.

**Out of scope:**

- Multi-page images. The project [relieur](https://github.com/papoteur-mga/relieur) addresses sheet music spanning multiple pages.
- PDF files.

## MusicXML as the Output Format

MusicXML is the only supported output format. A wide ecosystem of tools exists to convert MusicXML into other formats, so this project restricts its scope to MusicXML generation.

**Out of scope:**

- Any other output formats, unless independently contributed and maintained.

## Desktop-Centric Focus

The primary target platforms are desktop PCs and laptops. The implementation assumes access to sufficient CPU resources to prioritize output quality.

The following environments are considered secondary and are supported only when they do not conflict with the main project goals:

- Mobile devices (see [Andromr](https://github.com/aicelen/Andromr))
- Cloud GPU environments such as Google Colab
- Serverless environments without GPUs, where constraints on image size (model size, size of dependencies ...) apply
