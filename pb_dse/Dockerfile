FROM gcr.io/datamechanics/spark-py-connectors:3.0.0-dm4
RUN pwd
MAINTAINER Dse_sagar
RUN mkdir /dse
WORKDIR /dse
COPY requirements.txt /dse/
COPY requirements2.txt /dse/
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements2.txt
RUN pip3 install tqdm
RUN python3.7 -m pip install pip 
RUN pip3 install djangorestframework
RUN pip install django-tinymce
RUN pip install skflow
COPY . /dse/
RUN ls -la /
RUN pwd