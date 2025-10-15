FROM python:3.12-slim

# Install dependencies including Postgres client
RUN apt-get update && apt-get install -y postgresql-client gcc libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Make wait-for-db.sh executable
RUN chmod +x wait-for-db.sh

CMD ["./wait-for-db.sh", "db", "python", "manage.py", "runserver", "0.0.0.0:8000"]
