FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# use this for deployment
CMD ["sh", "-c", "uvicorn AIRecommenderModel:app --host 0.0.0.0 --port $PORT"]

# Use this for local testing 
# CMD ["sh", "-c", "uvicorn AIRecommenderModel:app --host 0.0.0.0 --port $PORT 8000"]