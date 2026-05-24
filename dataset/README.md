PDMX dataset
============

This dataset is provided by the PDMX project: https://github.com/pnlong/PDMX/

Required files to download:
- `mxl.tar.gz` — contains the `mxl/` directory with MusicXML files. Extract this into the project's `dataset/` folder so the files end up under `dataset/mxl/`.
- `PDMX.csv` — place this file into the project's `dataset/` folder as `dataset/PDMX.csv`.

Installation steps:
1. Visit https://github.com/pnlong/PDMX/ and download `mxl.tar.gz` and `PDMX.csv`.
2. Place `PDMX.csv` in this project's `dataset/` folder.
3. Extract `mxl.tar.gz` into this project's `dataset/` folder. Example commands:

   - Linux / macOS:

     ```bash
     tar -xzf mxl.tar.gz -C dataset
     ```

   - Windows (PowerShell):

     ```powershell
     tar -xzf mxl.tar.gz -C dataset
     ```

Optional:
- Rendered images (if generated) belong in `dataset/images/`.
- Attack artifacts belong in `attacks/artifacts/`.

Notes:
- This repository already ignores `dataset/mxl/`, `dataset/images/`, `dataset/PDMX.csv`, and `attacks/artifacts/` via `.gitignore`.
- If any of these files were previously committed, stop tracking them with:

```
git rm --cached dataset/PDMX.csv
git rm -r --cached dataset/mxl dataset/images
git commit -m "Stop tracking PDMX dataset files"
```
