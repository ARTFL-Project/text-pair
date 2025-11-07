#!/usr/bin/env python3
"""
Standalone llama-server manager script.
Starts and manages a llama-server process with automatic cleanup.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


class LlamaServerManager:
    def __init__(self, model_path, port=8080, context_window=8192, concurrency_limit=8, max_retries=30, retry_delay=1):
        self.model_path = model_path
        self.port = port
        self.context_window = context_window
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.process = None
        self.base_url = f"http://127.0.0.1:{port}"
        self.llama_server_cmd = []

    def find_llama_server(self):
        """Find llama-server binary in common locations"""
        possible_locations = [
            "llama-server",  # In PATH
            "llama.cpp/llama-server",  # Local build
            "~/llama.cpp/llama-server",  # Home directory
            "/usr/local/bin/llama-server",  # Homebrew on macOS
            "/opt/homebrew/bin/llama-server",  # M1 Mac homebrew
        ]

        for location in possible_locations:
            expanded_path = Path(location).expanduser()
            if expanded_path.exists() and expanded_path.is_file():
                return str(expanded_path)

        # Try to find in PATH
        import shutil
        if shutil.which("llama-server"):
            return "llama-server"

        raise FileNotFoundError(
            "llama-server binary not found. Please ensure llama.cpp is installed and "
            "llama-server is in your PATH or in one of the common locations."
        )

    def start(self):
        """Start the llama-server process"""
        llama_server = self.find_llama_server()

        # When using --parallel, the context window is divided among all slots
        # So we need to multiply the desired context window by the parallel value
        total_context_size = self.context_window * self.concurrency_limit

        cmd = [
            llama_server,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(total_context_size),
            "--n-gpu-layers", "99",
            "--parallel", str(self.concurrency_limit),  # Handle multiple concurrent requests
            "--log-disable",    # Reduce log noise
            "--threads", "4",   # CPU threads
            "--mlock",           # Enable memory locking
            "--cont-batching",   # Enable continuous batching
            "-fa", "on"        # Enable fast attention
        ]

        # If file exists locally, use --model, otherwise assume HF repo
        if os.path.exists(self.model_path):
            cmd.extend(["--model", self.model_path])
            print(f"Starting llama-server with local model: {self.model_path}")
        else:
            cmd.extend(["-hf", self.model_path])
            print(f"Starting llama-server with HF repo: {self.model_path}")

        print(f"Context window per slot: {self.context_window}, Parallel slots: {self.concurrency_limit}, Total context: {total_context_size}")

        self.llama_server_cmd = cmd
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except FileNotFoundError:
            raise RuntimeError(f"Failed to start llama-server. Binary not found: {llama_server}")

        # Register cleanup function
        atexit.register(self.stop)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Wait for server to be ready
        self.wait_for_ready()

    def wait_for_ready(self):
        """Wait for server to be ready to accept requests"""
        for attempt in range(self.max_retries):
            if self.process and self.process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.process.communicate()
                raise RuntimeError(f"llama-server process terminated early:\nstdout: {stdout}\nstderr: {stderr}")

            try:
                # Try to make a simple health check request
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return  # Server is ready
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
                pass  # Server not ready yet

            time.sleep(self.retry_delay)


        raise RuntimeError(
            f"Server failed to become ready after {self.max_retries} attempts.\n"
            f"This may indicate:\n"
            f"  - Model download from Hugging Face is taking too long\n"
            f"  - Insufficient system resources (RAM/GPU)\n"
            f"  - Invalid model path or format\n\n"
            f"Try running the server manually to diagnose the issue:\n"
            f"  {' '.join(self.llama_server_cmd)}"
        )

    def stop(self):
        """Stop the llama-server process"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.stop()
        sys.exit(0)

    def is_running(self):
        """Check if the server process is still running"""
        return self.process is not None and self.process.poll() is None


def main():
    if len(sys.argv) < 2:
        print("Usage: python llama_server_manager.py <model_path> [port] [concurrency_limit]")
        print("Examples:")
        print("  Local model: python llama_server_manager.py /path/to/model.gguf 8080")
        print("  HF model:    python llama_server_manager.py microsoft/DialoGPT-medium 8080")
        print("  HF model:    python llama_server_manager.py unsloth/gemma-2-2b-it-bnb-4bit 8080")
        sys.exit(1)

    model_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    context_window = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    concurrency_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    server = LlamaServerManager(model_path, port, context_window, concurrency_limit)

    try:
        server.start()

        # Keep the server alive
        if server.process:
            server.process.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        server.stop()


if __name__ == "__main__":
    main()