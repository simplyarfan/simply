# Match 2 (OCR).
#
# Same base image, Python packages and model weights as Match 3, so this is
# built on top of the Match 3 image (match3/Dockerfile) and only the app code
# changes. That keeps the 16 GB of weights from being fetched and pushed twice.
# The mandated rocm/pytorch layers are still the bottom of the image.
#
#   docker build --build-arg BASE_IMAGE=<your Match 3 image> -t <name> match2
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --link: a layer of its own, so the base layers never have to be pulled
COPY --link --chmod=755 app/ /app/

# Healthy once the model is loaded and the server is listening, for anything
# that waits on the container before sending the first image.
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 \
    CMD test -S /tmp/simply.sock || exit 1

CMD ["/app/start.sh"]
