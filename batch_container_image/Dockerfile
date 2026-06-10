FROM python:3.10

ARG SCRIPT_PATH=batch_container_image
ARG WORKDIR=/usr/local/workspace
RUN mkdir -p ${WORKDIR}

WORKDIR ${WORKDIR}
COPY ${SCRIPT_PATH}/requirements.txt .
RUN pip install -r requirements.txt

COPY ${SCRIPT_PATH}/../error_reduction ${WORKDIR}/error_reduction
COPY ${SCRIPT_PATH}/job_script.py .
COPY ${SCRIPT_PATH}/container_script.py .

ENTRYPOINT ["python", "container_script.py"]
