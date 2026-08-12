FROM python:3.13-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Install Playwright browsers at build time so they're baked into the image
RUN uv run playwright install chromium --with-deps

COPY . .

CMD ["sh", "-c", "uv run python scripts/build_pinyin_reference.py && exec uv run python bot.py"]
