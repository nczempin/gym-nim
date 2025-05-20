FROM python:3.6-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir nose \ 
    && pip install --no-cache-dir -e .
CMD ["python", "tests.py"]
