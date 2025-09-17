FROM bitnami/spark:3.5

# Switch to root user to install things system-wide if needed, or just copy files
USER root

# Copy your requirements file into the container
COPY requirements.txt  /tmp/requirements.txt

# Switch back to the default user to install packages
USER ${NB_UID}

# Install the packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt