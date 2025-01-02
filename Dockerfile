FROM python:3.9-slim

WORKDIR /app

COPY . /app


COPY notebook/mlruns/models/insurance_price_predictor.pkl /app/models/insurance_price_predictor.pkl

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install streamlit

EXPOSE 8501

ENV PYTHONUNBUFFERED 1


CMD ["streamlit", "run", "app.py"]
