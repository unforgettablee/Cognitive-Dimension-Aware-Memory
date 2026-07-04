"""MTL (Memory Transfer Learning) Agent.

A thin subclass of DefaultAgent that accepts pre-computed memory context
and injects it into the task's Jinja2 template via extra_template_vars.

Usage::

    agent = MTLAgent(model, env, **agent_config)
    result = agent.run(task_text, memory_context="## Relevant Past Experiences...")

The agent expects the ``instance_template`` to include ``{{memory_context}}``.
If memory_context is empty, it renders as an empty string (safe no-op).
"""

from minisweagent.agents.default import DefaultAgent


class MTLAgent(DefaultAgent):
    """Agent that injects retrieved memories into the task context.

    Memory is passed via ``extra_template_vars["memory_context"]``,
    which Jinja2 templates can render with ``{{memory_context}}``.

    The agent itself has zero dependency on the MTL package — the caller
    (typically ``swebench_mtl.py``) is responsible for computing the
    ``memory_context`` string ahead of time.
    """

    def run(self, task: str = "", **kwargs) -> dict:
        """Run the agent on a task, optionally with memory context.

        Args:
            task: The task description / problem statement.
            **kwargs: Passed through to DefaultAgent.run().  If
                ``memory_context`` is present, it is captured into
                ``extra_template_vars`` so templates can render it.
        """
        if "memory_context" in kwargs:
            self.extra_template_vars["memory_context"] = kwargs.pop("memory_context")
        return super().run(task, **kwargs)
