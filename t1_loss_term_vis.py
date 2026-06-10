import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
tmp_results_dir = root_dir / '.tmp_results'
tmp_results_dir.mkdir(parents=True, exist_ok=True)