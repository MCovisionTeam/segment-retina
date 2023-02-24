FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /workspace

ADD . /workspace

RUN pip install -r requirements.txt

