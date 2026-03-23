FROM apache/spark:3.5.6-java17-python3

# Switch to root user to install things system-wide if needed, or just copy files
USER root

# Copy your requirements file into the container
COPY requirements.txt  /tmp/requirements.txt

# Install the packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt