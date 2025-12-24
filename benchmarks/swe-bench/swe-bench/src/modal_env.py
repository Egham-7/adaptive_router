"""Modal deployment wrapper using swe-rex."""

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import BashAction, Command, CreateBashSessionRequest

T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: float = 0.1,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay cap in seconds
        jitter: Random jitter factor (0.0-1.0)

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)

            print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    raise last_exception


@dataclass
class ModalConfig:
    """Configuration for Modal deployment."""

    image: str = "python:3.11"
    startup_timeout: float = 180.0
    runtime_timeout: float = 300.0
    deployment_timeout: float = 3600.0
    sandbox_idle_timeout: float = 90.0  # Reduced from 600s (6.6x cost savings)

    # Resource configuration
    cpu: float = 0.5  # Half a CPU core (sufficient for shell commands)
    memory: int = 512  # 512 MiB (adequate for most SWE-bench tasks)

    # Retry configuration
    max_retries: int = 3
    retry_base_delay: float = 1.0  # Exponential backoff: 1s, 2s, 4s


class ModalEnvironment:
    """Wrapper for swe-rex Modal deployment."""

    def __init__(self, config: Optional[ModalConfig] = None):
        self.config = config or ModalConfig()
        self.deployment: Optional[ModalDeployment] = None
        self.runtime = None

    async def start(self):
        """Start the Modal deployment."""
        self.deployment = ModalDeployment(
            image=self.config.image,
            startup_timeout=self.config.startup_timeout,
            runtime_timeout=self.config.runtime_timeout,
            deployment_timeout=self.config.deployment_timeout,
            modal_sandbox_kwargs={
                "idle_timeout": int(self.config.sandbox_idle_timeout),
                "cpu": self.config.cpu,
                "memory": self.config.memory,
            },
        )
        await self.deployment.start()
        self.runtime = self.deployment.runtime

        # Create default bash session
        await self.runtime.create_session(CreateBashSessionRequest(session="default"))

        return self

    async def stop(self):
        """Stop the Modal deployment."""
        if self.deployment:
            await self.deployment.stop()
            self.deployment = None
            self.runtime = None

    async def run_command(self, command: str, timeout: float = 60.0) -> tuple[int, str]:
        """Run a command in the Modal environment.

        Returns:
            Tuple of (return_code, output)
        """
        if not self.runtime:
            raise RuntimeError("Environment not started")

        result = await self.runtime.run_in_session(
            BashAction(command=command, timeout=timeout, session="default", check="silent")
        )
        return result.exit_code, result.output

    async def execute(self, command: list[str]) -> tuple[int, str]:
        """Execute a one-off command.

        Returns:
            Tuple of (exit_code, output)
        """
        if not self.runtime:
            raise RuntimeError("Environment not started")

        result = await self.runtime.execute(Command(command=command))
        output = result.stdout + result.stderr if result.stderr else result.stdout
        return result.exit_code, output

    async def setup_repo(self, repo_url: str, base_commit: str) -> bool:
        """Clone and setup a repository at a specific commit.

        Returns:
            True if successful
        """
        commands = [
            f"git clone {repo_url} /testbed",
            f"cd /testbed && git checkout {base_commit}",
            "cd /testbed && pip install -e . 2>/dev/null || true",
        ]

        for cmd in commands:
            returncode, output = await self.run_command(cmd, timeout=120.0)
            if returncode != 0 and "git checkout" in cmd:
                print(f"Failed to setup repo: {output}")
                return False

        return True

    async def get_patch(self) -> str:
        """Get the git diff patch from the repo."""
        returncode, output = await self.run_command(
            "cd /testbed && git add -A && git diff --cached"
        )
        return output if returncode == 0 else ""

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def get_docker_image(instance_id: str) -> str:
    """Get the SWE-bench Docker image name for an instance."""
    # Convert instance_id to Docker-compatible format
    # e.g., "astropy__astropy-12907" -> "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    id_docker = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
