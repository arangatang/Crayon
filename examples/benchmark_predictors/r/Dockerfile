FROM gluonts:cpu_official
#RUN cd .. && git clone https://github.com/facebook/prophet.git && cd prophet && cd python && pip install . -r requirements.txt && cd ../../gluonts
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install r-base -y
RUN pip install 'rpy2>=2.9.*,<3.*'
RUN R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'
