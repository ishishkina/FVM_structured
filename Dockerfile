FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

# Default: run the test suite. Run the solver with:
#   docker run --rm fvm-structured python A_main.py
CMD ["pytest", "-v"]
