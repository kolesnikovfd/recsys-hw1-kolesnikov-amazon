"""Execute with the current Python interpreter, no global kernel registration."""
import os
import sys
import tempfile
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
os.chdir(root)
(root/'data').mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(root/'data'/'mpl-cache'))
os.environ.setdefault('IPYTHONDIR', str(root/'data'/'ipython'))
with tempfile.TemporaryDirectory() as temp:
    spec = Path(temp)/'kernels'/'recsys-local'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Recsys local', 'language': 'python'}))
    os.environ['JUPYTER_PATH'] = temp + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    nb = nbformat.read(root/'experiments.ipynb', as_version=4)
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:90]}', flush=True)
    client = NotebookClient(nb, timeout=3600, startup_timeout=600, kernel_name='recsys-local',
                            resources={'metadata': {'path': str(root)}}, on_cell_start=progress)
    try:
        client.execute()
    finally:
        nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        nbformat.write(nb, root/'experiments.ipynb')
    nbformat.validate(nb)
    print('Notebook executed and saved successfully.', flush=True)
