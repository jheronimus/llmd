# Multi-stage Dockerfile for llmd daemon
# Stage 1: Build
FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    libclang-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -O"
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/llmd /app/llmd

ENV LLMD_HOST="0.0.0.0"
ENV LLMD_PORT="8080"
ENV HOME="/root"

VOLUME ["/root/.cache/llmd/models"]
EXPOSE 8080

ENTRYPOINT ["/app/llmd", "--host", "0.0.0.0", "--port", "8080"]
