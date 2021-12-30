FROM ubuntu:20.04
RUN pwd
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pwd
MAINTAINER Dse_sagar
RUN mkdir /dse
WORKDIR /dse
COPY requirements.txt /dse/
COPY requirements2.txt /dse/
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements2.txt
RUN pip3 install tqdm
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
#RUN python3.7 -m pip install pip
RUN pip3 install djangorestframework
RUN pip3 install django-tinymce
COPY . /dse/
RUN ls -la /
RUN pwd

#FROM gcr.io/datamechanics/spark-py-connectors:3.0.0-dm4