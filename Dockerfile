# The TAKK API (step 18 of ROADMAP-takk.md). It serves one bundle (see takk/bundle.py) and holds no
# dataset, no prepared store and no training run. Build from the repo root, with a bundle built first:
#     uv run scripts/build_serving.py
#     docker build --build-arg BUNDLE=outputs/serving/iv14_h384_e20-sts_lexikon-234f4575 -t takk .
#     docker run --rm -p 8002:8002 takk
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# ffmpeg decodes the recording's audio (takk/speech.py); nothing else here needs a system library.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv HF_HOME=/opt/huggingface
WORKDIR /app

# The dependencies are a layer of their own, so a new bundle or a code change does not reinstall
# torch. The versions are the lockfile's, but torch comes from PyTorch's CPU index instead of PyPI:
# the wheels there carry no CUDA runtime, which is 12 GB of an image that scores on the CPU. The
# nvidia and triton packages go with them. --no-default-groups leaves out the extraction and dataset
# packages, which the app never imports (see pyproject.toml and tests/test_layout.py).
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-default-groups --no-emit-project --no-hashes -o requirements.txt \
    && sed -i -E '/^(nvidia-|triton)/d' requirements.txt \
    && uv venv \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
       -r requirements.txt

# The Swedish speech model that times the spoken words, about 1.2 GB, downloaded at build time
# rather than on the first attempt of the first learner.
RUN /app/.venv/bin/python -c "\
from transformers import AutoModelForCTC, AutoProcessor; \
AutoModelForCTC.from_pretrained('KBLab/wav2vec2-large-voxrex-swedish'); \
AutoProcessor.from_pretrained('KBLab/wav2vec2-large-voxrex-swedish')"

COPY src ./src
RUN uv pip install --no-deps .

ARG BUNDLE
COPY ${BUNDLE} ./bundle

# Cloud Run sets $PORT; locally it is the same 8002 the page's dev proxy expects.
ENV PORT=8002
EXPOSE 8002
CMD /app/.venv/bin/takk --bundle /app/bundle --host 0.0.0.0 --port $PORT
