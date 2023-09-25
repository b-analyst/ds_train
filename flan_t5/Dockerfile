  FROM nvidia/cuda

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
# RUN apt-get install -y nvidia-cuda-toolkit
# RUN apt install -y nvidia-driver-510

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /tmp/requirements.txt


WORKDIR /app 

COPY . .