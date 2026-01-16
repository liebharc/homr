# Run e.g. with docker build -t homr . && docker run --rm -p 8080:8000 homr
# And then send images: curl -X POST -F "file=@tabi.jpg" http://localhost:8080/process --output tabi.musicxml

FROM python:3.11

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN git clone https://github.com/liebharc/homr .

RUN /root/.local/bin/poetry install --without dev

RUN /root/.local/bin/poetry run pip install --no-cache-dir fastapi uvicorn python-multipart

# Pre-download models
RUN /root/.local/bin/poetry run python homr/main.py --init

# Generate FastAPI app inline
RUN cat <<'EOF' > api.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import subprocess
import tempfile
import os
import shutil

app = FastAPI()

@app.post("/process")
def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    subprocess.check_call([
        "python", "homr/main.py", input_path
    ])

    base, _ = os.path.splitext(input_path)
    output_path = base + ".musicxml"

    background_tasks.add_task(shutil.rmtree, tmpdir)

    return FileResponse(
        output_path,
        media_type="application/xml",
        filename=os.path.basename(output_path),
    )
EOF

EXPOSE 8000

CMD ["/root/.local/bin/poetry", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

