FROM python:3.7

ADD . /gluonts

RUN pip install -r /gluonts/requirements/requirements-mxnet.txt
RUN pip install -r /gluonts/requirements/requirements-extras-shell.txt
RUN pip install -r /gluonts/requirements/requirements.txt
RUN pip install /gluonts[shell]
RUN pip install pandas==1.0.5
RUN pip install ujson
ENTRYPOINT ["python", "-m", "gluonts.shell"]