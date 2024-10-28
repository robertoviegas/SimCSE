# Use a base image, for example, TensorFlow with GPU support
FROM tensorflow/tensorflow:2.13.0-gpu

# Install system dependencies, including curl and build-essential for Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Optionally update to ensure the latest stable version of Rust and Cargo
RUN rustup update

# # Set up Python and install dependencies
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the application files
COPY . .