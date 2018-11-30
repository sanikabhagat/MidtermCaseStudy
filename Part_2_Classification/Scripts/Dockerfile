# Use an official Python runtime as a parent image
FROM python:3.6-slim

USER root

# Set the working directory to /app
# WORKDIR /app

# Copy the current directory contents into the container at /app
# ADD . /app


RUN pip3 install ipython
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install BeautifulSoup4
RUN pip3 install scipy
RUN pip3 install lxml


RUN apt-get update \
  && apt-get install -y curl

EXPOSE 8888

WORKDIR /src/

RUN mkdir /src/assign_3
RUN mkdir /src/assign_3/out


ADD wrapper_script.sh /src/assign_3
ADD main_script_classfn.py /src/assign_3
ADD wrapper_script.py /src/assign_3
ADD wrapper_script_short.py /src/assign_3

RUN chmod +x /src/assign_3/wrapper_script.sh
RUN chmod +x /src/assign_3/wrapper_script.py
