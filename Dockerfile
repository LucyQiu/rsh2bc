FROM centos:7

MAINTAINER ruhld@bc.edu 

RUN yum -y update && yum -y install \
epel-release \
python-devel \
gcc \
pygtk2 \
tkinter \
&& yum -y install python-pip \
&& yum clean all \ 
&& rm -rf /var/cache/yum

RUN pip --no-cache-dir install networkx==2.2 numpy==1.13.3 scipy==0.19.1 multiprocessing==2.6.2.1 matplotlib==2.1.1 Pillow

ENV HOME /apps
WORKDIR /apps
RUN mkdir -p /apps/.local/share
COPY Img2net /apps/
CMD python img2net_run.py
