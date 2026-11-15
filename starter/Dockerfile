FROM python:3.13-slim
WORKDIR /bot
COPY main.py ./
COPY harness ./harness
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "main.py"]
