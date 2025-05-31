# ---- Base Image ----
FROM python:3.9-slim

# ---- Environment ----
# Don’t write .pyc files and flush stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Create Non-Root User ----
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ---- Set Working Directory ----
WORKDIR /app

# ---- System Dependencies ----
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# ---- Python Dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Application Code ----
COPY . .

# ---- Switch to Non-Root ----
USER appuser

# ---- Entrypoint & Default Command ----
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
