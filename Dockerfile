#Old Docerk file comfig, won't work because of some installation dependencies missing
# FROM python:3.10-slim-buster
# WORKDIR /app
# COPY . /app

# RUN apt-get update -y && apt-get install awscli -y

# RUN apt-get update && pip install -r requirements.txt
# CMD ["python3", "app.py"]


FROM python:3.10-slim-bookworm

WORKDIR /app
COPY . /app

# Install dependencies needed for apt to work properly
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip \
        gnupg && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]