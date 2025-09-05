# 1. Use Python 3.13-slim
FROM python:3.13-slim

# 2. Install system dependencies required for WeasyPrint
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libfontconfig1 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libxml2 \
    libxslt1.1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 3. Create /app and set it as working directory
WORKDIR /app

# 4. Copy only pyproject.toml (so Docker caches the install layer)
COPY pyproject.toml .

# 5. Install your project (which includes all dependencies via pyproject.toml)
RUN pip install --no-cache-dir .

# 6. Copy your entire application (main.py, any modules, README.md, etc.)
COPY . .

# 7. Expose port 8000 (matches the mcp.run port)
EXPOSE 8000

# 8. At runtime, start your FastMCP server:
CMD ["python", "src/main.py"]
