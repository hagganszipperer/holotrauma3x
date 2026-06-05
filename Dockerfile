FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /opt/holotrauma3x

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["holotrauma3x"]
CMD ["--help"]
