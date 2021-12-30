FROM ubuntu:20.04
RUN pwd
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
MAINTAINER DSE_DEV
WORKDIR /dse_sagar/
COPY requirements.txt .
COPY requirements2.txt .
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements2.txt
RUN pip3 install --upgrade awscli
COPY lib/PySpark .
COPY lib/PBS .
COPY bin/ .
COPY config.ini .
COPY data .
RUN pwd
RUN ls -l
COPY bin/PySparkMainPipeline.py PySparkMainPipeline.py
COPY bin/DSEMainPipeline.py DSEMainPipeline.py