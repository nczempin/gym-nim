FROM python:3.12-slim

# Set environment variable to suppress pip root user warnings
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app
COPY . /app

# Install dependencies with pinned versions for reproducible builds
RUN pip install --no-cache-dir \
    pytest==7.4.4 \
    pytest-cov==4.1.0 \
    numpy==1.26.4 \
    && pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "tests.py"]
