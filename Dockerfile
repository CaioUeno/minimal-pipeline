FROM python:3.8

# set a directory
WORKDIR /evaluation

# copy data, code, luigi config, and requirements
COPY data/ data/
COPY src/ src/
COPY luigi.cfg luigi.cfg
COPY requirements-docker.txt requirements-docker.txt

# use current dir as pythonpath
ENV PYTHONPATH="."

# install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# run the command
CMD ["luigi", "--module", "src.tasks.DeployTask", "DeployTask", "--in-file", "data/iris.csv"]
