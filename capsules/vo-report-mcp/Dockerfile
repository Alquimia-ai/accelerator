# 1) Base
FROM python:3.13-slim

# 2) System deps for WeasyPrint (Debian trixie names)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libcairo2 \
      libpango-1.0-0 \
      libpangoft2-1.0-0 \
      libharfbuzz0b \
      libfontconfig1 \
      libgdk-pixbuf-2.0-0 \
      libxml2 \
      libxslt1.1 \
      gcc \
      python3-dev \
      fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# 3) Workdir
WORKDIR /app

# 4) Copy project *then* install (so pip can see your source)
COPY . .

# 5) Install your project (reads pyproject.toml)
RUN pip install --no-cache-dir .

# 6) Expose and run
EXPOSE 8000
CMD ["python", "src/main.py"]
