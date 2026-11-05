ARG SLIME_IMAGE=mimicagent-slime:base
FROM ${SLIME_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap patch && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall -y flash-attn-3 flash-attn-4 && \
    python -m pip install --no-build-isolation flash-attn==2.8.3
