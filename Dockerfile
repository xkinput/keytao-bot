FROM python:3.13-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY . .

CMD ["uv", "run", "python", "bot.py"]
