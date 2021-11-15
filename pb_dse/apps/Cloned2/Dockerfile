FROM gcr.io/datamechanics/spark-py-connectors:3.0.0-dm4
RUN pwd
MAINTAINER PYSpark
WORKDIR /dse_sagar/
COPY requirements.txt .
COPY requiremant2.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install -r requiremant2.txt
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
ENV PYSPARK_MAJOR_PYTHON_VERSION=3