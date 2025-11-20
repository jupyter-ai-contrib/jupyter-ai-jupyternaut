import os
from typing import Optional
from jupyterlab_commands_toolkit.tools import execute_command

from .utils import get_serverapp


async def open_file(file_path: str):
    """
    Opens a file in JupyterLab main area

    Args:
        file_path: Path to the file relative to jupyter root

    Returns:
        dict: A dictionary containing the response from JupyterLab
        The structure includes:
        - success (bool): Whether the command executed successfully
        - result (any): The return value after opening the file
        - error (str, optional): Error message if the command failed

    Raises:
        asyncio.TimeoutError: If the frontend doesn't respond within the timeout period

    """
    # If file_path is absolute, convert it to a path relative to the jupyter root
    if os.path.isabs(file_path):
        try:
            serverapp = get_serverapp()
            root_dir = serverapp.root_dir
        except Exception:
            # Fallback to current working directory if server app is not available
            root_dir = os.getcwd()

        try:
            # Convert absolute path to relative path from jupyter root
            file_path = os.path.relpath(file_path, root_dir)
        except ValueError:
            # If paths are on different drives (Windows), use the absolute path as is
            # JupyterLab may still be able to handle it
            pass

    return await execute_command("docmanager:open", {"path": file_path})


async def run_all_cells():
    """
    Runs all cells in the currently active Jupyter notebook
    """
    return await execute_command("notebook:run-all-cells")


async def run_cell(cell_id: str, username: Optional[str] = None) -> dict:
    """
    Runs a specific cell in the active notebook by selecting it and executing it.

    Args:
        cell_id: The UUID of the cell to run, or a numeric index as string
        username: Optional username to get the active cell for that specific user

    Returns:
        dict: A dictionary containing the response from the run-cell command

    Raises:
        ValueError: If the cell_id is not found in the notebook
        RuntimeError: If there is no active notebook or notebook is not currently open
    """
    from .notebook import select_cell

    # First, select the target cell
    await select_cell(cell_id, username)

    # Then run the currently selected cell
    return await execute_command("notebook:run-cell")


async def select_cell_below():
    """
    Moves the cursor down to select the cell below the currently selected cell
    """
    return await execute_command("notebook:move-cursor-down")


async def select_cell_above():
    """
    Moves the cursor up to select the cell above the currently selected cell
    """
    return await execute_command("notebook:move-cursor-up")


async def restart_kernel():
    """
    Restarts the notebook kernel, useful when new packages are installed
    """
    return await execute_command("notebook:restart-kernel")


toolkit = [open_file, run_cell, run_all_cells, restart_kernel]
