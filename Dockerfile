FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --root-user-action=ignore --no-cache-dir pytest pytest-cov numpy \
    && pip install --root-user-action=ignore --no-cache-dir -e .
CMD ["python", "tests.py"]
