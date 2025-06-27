# 1. Use Python 3.13-slim
FROM python:3.13-slim

# 2. Create /app and set it as working directory
WORKDIR /app

# 3. Copy only pyproject.toml (so Docker caches the install layer)
COPY pyproject.toml .

# 4. Install your project (which includes mcp[cli]>=2.3.0 via pyproject.toml)
RUN pip install --no-cache-dir .

# 5. Copy your entire application (app.py, any modules, README.md, etc.)
COPY . .

# 6. Expose port 8000 (matches the mcp.run port)
EXPOSE 8000

# 7. At runtime, start your FastMCP server:
#    (Assumes your main script is named app.py)
CMD ["python", "main.py"]
