FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Download the model weights
RUN python setup_model.py

# Expose the port Hugging Face uses
EXPOSE 7860

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
