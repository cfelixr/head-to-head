FROM python:3.13-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip git gcc libc6-dev make ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

RUN pip install --no-cache-dir pipx && pipx ensurepath
ENV PATH=/root/.local/bin:$PATH
RUN pipx install aws-sam-cli

# Install testing dependencies in the image
COPY tests/requirements.txt /tmp/test-requirements.txt
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt && rm /tmp/test-requirements.txt

WORKDIR /work
ENTRYPOINT ["/bin/bash","-lc"]
