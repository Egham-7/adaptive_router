"""SWE-bench agent using LiteLLM for LLM calls."""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import litellm

# Configure LiteLLM for Nordlys routing
litellm.drop_params = True


SYSTEM_PROMPT = """You are a helpful assistant that can interact with a shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.

Format your response as shown:

THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```

Failure to follow these rules will cause your response to be rejected."""

INSTANCE_PROMPT = """<pr_description>
{problem_statement}
</pr_description>

<instructions>
You're a software engineer working on fixing the issue described above.
Your task is to make changes to the source code in /testbed to fix the issue.

IMPORTANT: This is an interactive process. Issue ONE command at a time.

## Workflow
1. Explore the codebase to understand the issue
2. Create a script to reproduce the issue
3. Edit the source code to fix the issue
4. Verify your fix works
5. Test edge cases

## Rules
- Each response should include a THOUGHT section and ONE bash command
- Use `cat <<'EOF' > file` for creating/editing files
- Use `sed -i` for inline edits
- Do NOT modify tests or config files

## When Done
When you've completed your fix, issue:
```bash
echo COMPLETE_TASK_AND_SUBMIT && git add -A && git diff --cached
```
</instructions>"""


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    model: str = "anthropic/nordlys/nordlys-code"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: int = 50
    cost_limit: float = 3.0
    temperature: float = 0.0


@dataclass
class AgentState:
    """State of the agent during execution."""

    messages: list = field(default_factory=list)
    steps: int = 0
    total_cost: float = 0.0
    done: bool = False
    patch: str = ""


class SWEAgent:
    """Agent for solving SWE-bench instances."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # Configure API (use NORDLYS_ env vars)
        if self.config.api_base:
            litellm.api_base = self.config.api_base
        elif os.environ.get("NORDLYS_API_BASE"):
            litellm.api_base = os.environ["NORDLYS_API_BASE"]

    def _create_initial_messages(self, problem_statement: str) -> list:
        """Create initial messages for the conversation."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INSTANCE_PROMPT.format(problem_statement=problem_statement),
            },
        ]

    def _extract_command(self, response: str) -> Optional[str]:
        """Extract bash command from LLM response."""
        # Match ```bash ... ``` blocks
        pattern = r"```bash\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        if len(matches) > 1:
            print(f"Warning: Found {len(matches)} bash blocks, using first one")

        return matches[0].strip()

    def _is_done(self, command: str, output: str) -> bool:
        """Check if the agent has completed the task."""
        return "COMPLETE_TASK_AND_SUBMIT" in command

    def _clean_patch(self, output: str) -> str:
        """Extract and clean the git diff from command output."""
        # Remove terminal escape sequences
        import re
        output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        output = re.sub(r'\x1b[=>]', '', output)
        output = output.replace('\r', '')

        # Find the diff section
        lines = output.split('\n')
        diff_lines = []
        in_diff = False

        for line in lines:
            if line.startswith('diff --git'):
                in_diff = True
            if in_diff:
                diff_lines.append(line)

        return '\n'.join(diff_lines).strip()

    async def get_next_action(self, state: AgentState) -> Optional[str]:
        """Get the next action from the LLM.

        Returns:
            The bash command to execute, or None if parsing failed.
        """
        try:
            response = await litellm.acompletion(
                model=self.config.model,
                messages=state.messages,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
            )

            content = response.choices[0].message.content

            # Track cost
            if hasattr(response, "usage"):
                # Approximate cost calculation
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                # Using approximate pricing
                state.total_cost += (input_tokens * 0.003 + output_tokens * 0.015) / 1000

            # Add assistant response to messages
            state.messages.append({"role": "assistant", "content": content})

            # Extract command
            command = self._extract_command(content)
            if not command:
                # Add error message and try again
                state.messages.append(
                    {
                        "role": "user",
                        "content": "Please provide exactly ONE bash code block with your command.",
                    }
                )
                return None

            return command

        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def add_observation(self, state: AgentState, returncode: int, output: str):
        """Add command output to the conversation."""
        # Truncate long output
        if len(output) > 10000:
            output = (
                output[:5000]
                + f"\n\n... ({len(output) - 10000} characters truncated) ...\n\n"
                + output[-5000:]
            )

        observation = f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
        state.messages.append({"role": "user", "content": observation})

    async def solve(
        self, problem_statement: str, run_command, get_patch
    ) -> tuple[str, float]:
        """Solve a SWE-bench instance.

        Args:
            problem_statement: The problem description
            run_command: Async function to run commands (returns returncode, output)
            get_patch: Async function to get the git diff patch

        Returns:
            Tuple of (patch, total_cost)
        """
        state = AgentState(messages=self._create_initial_messages(problem_statement))

        while state.steps < self.config.max_steps and not state.done:
            state.steps += 1

            # Check cost limit
            if state.total_cost >= self.config.cost_limit:
                print(f"Cost limit reached: ${state.total_cost:.2f}")
                break

            # Get next action
            command = await self.get_next_action(state)
            if not command:
                continue

            # Execute command
            returncode, output = await run_command(command)

            # Check if done
            if self._is_done(command, output):
                state.done = True
                state.patch = self._clean_patch(output)
                break

            # Add observation
            self.add_observation(state, returncode, output)

        # Get final patch if not already captured
        if not state.patch:
            raw_patch = await get_patch()
            state.patch = self._clean_patch(raw_patch)

        return state.patch, state.total_cost
