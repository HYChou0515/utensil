FROM python:3.9-slim-buster
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# System deps:
WORKDIR /app
COPY docker/requirements.txt /
COPY docker/utensil-*.whl /dist/

# Project initialization:
RUN pip install -r /requirements.txt
RUN pip install /dist/utensil-*.whl

# Creating folders, and files for a project:
COPY . /app

# wait for some dependency
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.2.1/wait /wait
RUN chmod +x /wait

CMD /wait && python main.py
