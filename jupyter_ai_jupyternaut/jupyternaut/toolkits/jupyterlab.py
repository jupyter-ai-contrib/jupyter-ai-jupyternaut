from jupyterlab_commands_toolkit.tools import execute_command


async def open_file(file_path: str):
    """
    Opens a file in JupyterLab main area
    """
    await execute_command("docmanager:open", {"path": file_path})

async def run_all_cells():
    """
    Runs all cells in the currently active Jupyter notebook
    """
    return await execute_command("notebook:run-all-cells")

toolkit = [open_file, run_all_cells]