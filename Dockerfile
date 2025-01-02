# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the model file into the container
COPY notebook/mlruns/models/insurance_price_predictor.pkl /app/models/insurance_price_predictor.pkl

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install streamlit explicitly in case it's not in the requirements.txt
RUN pip install streamlit

# Expose the port the app runs on
EXPOSE 8501

# Define environment variable
ENV PYTHONUNBUFFERED 1

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]
