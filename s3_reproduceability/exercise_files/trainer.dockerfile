# Base image
FROM python:3.7-slim
# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/
# copy essentials
COPY model.py model.py
COPY requirements.txt requirements.txt
COPY vae_mnist.py vae_mnist.py


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "vae_mnist.py"]