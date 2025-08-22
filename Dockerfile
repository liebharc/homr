FROM python:3.12

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN git clone https://github.com/liebharc/homr homr

RUN cd homr && /root/.local/bin/poetry install --without dev
