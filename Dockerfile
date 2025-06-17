FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pytest pytest-cov \ 
    && pip install --no-cache-dir -e .
CMD ["python", "tests.py"]
