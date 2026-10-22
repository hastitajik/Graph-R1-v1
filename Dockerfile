FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git build-essential curl python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda clean -a -y

RUN conda create -n graphr1 python=3.11 -y && conda clean -a -y
SHELL ["conda", "run", "-n", "graphr1", "/bin/bash", "-c"]

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install psutil ninja packaging
RUN pip install flash_attn --no-build-isolation

RUN pip install -e .

RUN pip install -r requirements.txt --no-build-isolation

RUN python3 -c "\
import re; \
p = '/opt/conda/envs/graphr1/lib/python3.11/site-packages/FlagEmbedding/inference/embedder/encoder_only/base.py'; \
c = open(p).read(); \
c = re.sub(r'(?<!torch_)\bdtype=self\.get_model_torch_dtype\(\),', 'torch_dtype=self.get_model_torch_dtype(),', c); \
open(p, 'w').write(c)"

EXPOSE 8001
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "graphr1"]
CMD ["bash"]
