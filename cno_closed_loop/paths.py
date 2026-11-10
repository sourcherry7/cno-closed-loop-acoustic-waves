"""Project paths and the trained runs used in the report.

Every location can be overridden with an environment variable, so the code runs
from any checkout (laptop, cluster) without editing source files:

    CNO_CL_DATA_DIR        dataset folder (new_wave_gauss_data.h5, new_normalization_stats.json)
    CNO_CL_CHECKPOINT_DIR  trained models, in forward/<run>/ and inverse/<run>/
    CNO_CL_RESULTS_DIR     figures, metrics and logs written by the evaluation scripts

Importing this module (done by ``cno_closed_loop/__init__.py``) also makes the
third-party code importable: the time-dependent CNO from the ``external/CNO``
submodule and the vendored FDTD solver in ``external/seismic_cpml``.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("CNO_CL_DATA_DIR", REPO_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("CNO_CL_CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("CNO_CL_RESULTS_DIR", REPO_ROOT / "results"))

FORWARD_CHECKPOINT_DIR = CHECKPOINT_DIR / "forward"
INVERSE_CHECKPOINT_DIR = CHECKPOINT_DIR / "inverse"

# Trained runs evaluated in the report (override to evaluate other runs).
FORWARD_RUN = os.environ.get("CNO_CL_FORWARD_RUN", "model_0903_newgauss")
FORWARD_CKPT = os.environ.get("CNO_CL_FORWARD_CKPT", "epoch=20-step=182784.ckpt")
INVERSE_RUN = os.environ.get("CNO_CL_INVERSE_RUN", "model_0103_new_data")
INVERSE_CKPT = os.environ.get("CNO_CL_INVERSE_CKPT", "epoch=3-step=30720.ckpt")

CNO_DIR = REPO_ROOT / "external" / "CNO" / "CNO2d_time_dependent_&_foundation_model"
SEISMIC_CPML_DIR = REPO_ROOT / "external" / "seismic_cpml"

# The CNO dataset classes read their data folder from CNO_DATA_PATH.
os.environ.setdefault("CNO_DATA_PATH", str(DATA_DIR))

# The CNO code base is imported as top-level modules (CNO_timeModule_CIN, DataLoaders, ...).
for _p in (CNO_DIR, SEISMIC_CPML_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
