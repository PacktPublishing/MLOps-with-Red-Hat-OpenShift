FROM registry.access.redhat.com/ubi9/python-39:latest

LABEL name="custom-notebook-ubi9-python-3.9" \
    summary="Minimal Jupyter notebook image for ODH notebooks" \
    description="Minimal Jupyter notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    io.k9s.display-name="Minimal Jupyter notebook image for ODH notebooks" \
    io.k9s.description="Minimal Jupyter notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    authoritative-source-url="https://github.com/PacktPublishing/MLOps-with-Red-Hat-OpenShift"

WORKDIR /opt/app-root/bin

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install -U "micropipenv[toml]"

# Install the oc client
RUN curl -L https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/stable/openshift-client-linux.tar.gz \
        -o /tmp/openshift-client-linux.tar.gz && \
    tar -xzvf /tmp/openshift-client-linux.tar.gz oc && \
    rm -f /tmp/openshift-client-linux.tar.gz

# Fix permissions to support pip in Openshift environments
RUN chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
      fix-permissions /opt/app-root -P

COPY utils utils/

COPY Pipfile.lock start-notebook.sh ./
USER root
RUN chmod 755 start-notebook.sh
USER 1001

# Install Python dependencies from Pipfile.lock file
RUN echo "Installing softwares and packages" && micropipenv install && rm -f ./Pipfile.lock

WORKDIR /opt/app-root/src

# Replace Notebook's launcher, "(ipykernel)" with Python's version 3.x.y
RUN sed -i -e "s/Python.*/$(python --version)\",/" /opt/app-root/share/jupyter/kernels/python3/kernel.json

ENTRYPOINT ["/opt/app-root/bin/start-notebook.sh"]