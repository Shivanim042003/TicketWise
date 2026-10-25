FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/ticketwise/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

RUN useradd --create-home --uid 10001 ticketwise \
    && mkdir -p /home/ticketwise/.cache/huggingface \
    && chown -R ticketwise:ticketwise /home/ticketwise

COPY --chown=ticketwise:ticketwise backend ./backend
COPY --chown=ticketwise:ticketwise data/tickets.json ./data/tickets.json

USER ticketwise

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
