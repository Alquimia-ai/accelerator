from typing import Any, Dict, Tuple
import contextlib
import traceback
import ast
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import os
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.resources import FileResource


class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.
    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.
    NOTE: not safe for production use! Use with caution.
    """

    def __init__(
        self,
        locals: Dict[str, Any],
        globals: Dict[str, Any],
        suppress_warnings: bool = False,
    ):
        """
        Initialize the code executor.
        Args:
            locals: Local variables to use in the execution context
            globals: Global variables to use in the execution context
            suppress_warnings: Whether to suppress warnings during code execution
        """
        self.globals = globals
        self.locals = locals
        self.suppress_warnings = suppress_warnings

    def execute(self, code: str) -> Tuple[bool, str, Any]:
        """
        Execute Python code and capture output and return values.
        Args:
            code: Python code to execute
        Returns:
            Dict with keys `success`, `output`, and `return_value`
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        output = ""
        return_value = None

        try:
            context_managers = [
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ]

            if self.suppress_warnings:
                context_managers.append(warnings.catch_warnings())

            with contextlib.ExitStack() as stack:
                for cm in context_managers:
                    stack.enter_context(cm)

                if self.suppress_warnings:
                    warnings.simplefilter("ignore")

                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None
                    if isinstance(last_node, ast.Expr):
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = (
                            code[: -len(last_line)] + "\n__result__ = " + last_line
                        )
                        exec(exec_code, self.globals, self.locals)
                        return_value = self.locals.get("__result__")
                    else:
                        exec(code, self.globals, self.locals)
                except:
                    exec(code, self.globals, self.locals)

            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()

        except Exception as e:
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        if return_value is not None:
            output += "\n\n" + str(return_value)

        return output


mcp = FastMCP("dynamics-kyc")

# Add CSV resource
csv_path = Path("./leasing_data.csv").resolve()
if csv_path.exists():
    csv_resource = FileResource(
        uri=f"file://{csv_path.as_posix()}",
        path=csv_path,
        name="Leasing Data CSV",
        description="CSV file containing leasing data for analysis",
        mime_type="text/csv",
    )
    mcp.add_resource(csv_resource)

# Initialize code executor with persistent state
code_executor = SimpleCodeExecutor(
    suppress_warnings=True,
    locals={},
    globals={
        "__builtins__": __builtins__,
        "datetime": datetime,
        "plt": plt,
        "pd": pd,
        "np": np,
        "re": re,
    },
)


@mcp.tool()
def execute_python_code(code: str) -> str:
    """Execute Python code for leasing data analysis and consultation. Use this tool to:
    - Load and analyze the leasing dataset (available as CSV resource)
    - Perform data exploration, filtering, and aggregation on leasing data
    - Generate insights, statistics, and visualizations from leasing records
    - Answer questions about leasing trends, patterns, and metrics

    State persists between executions, so variables and loaded data remain available.
    Common libraries available: pandas (pd), numpy (np), matplotlib (plt), datetime, re.
    """
    return code_executor.execute(code)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info",
    )
