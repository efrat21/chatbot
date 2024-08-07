FROM python:3.11
WORKDIR /Tensorbot
RUN pip install poetry
COPY . .
RUN poetry install --no-dev
EXPOSE 8000
CMD ["poetry", "run", "python", "server.py"]


