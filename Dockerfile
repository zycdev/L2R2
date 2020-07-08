FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /workspace/anli
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install dependencies.
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining code.
COPY . .
RUN chmod +x *.sh && \
    mkdir /results

# Run code.
#CMD ["python", "-u"]
CMD ["/bin/bash"]