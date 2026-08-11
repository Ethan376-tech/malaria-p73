"""Central path config for the project.

Data, checkpoints and outputs live under a single data root. Set it with the
MALARIA_DATA_ROOT environment variable, e.g.

    export MALARIA_DATA_ROOT=/path/to/A_Dissertation

It defaults to the original project location. NONE of these paths are shipped in the
public repository: the datasets are patient microscopy (not redistributable) and the
checkpoints are trained on them — see the README ("Data availability").

Import in any script:  from common.paths import RAW, CHECKPOINTS, OUTPUTS, ...
"""
import os
from pathlib import Path

DATA_PROJ   = Path(os.environ.get("MALARIA_DATA_ROOT", "/root/autodl-tmp/A_Dissertation"))
DATA_ROOT   = DATA_PROJ / "data"
RAW         = DATA_ROOT / "raw"
PROCESSED   = DATA_ROOT / "processed"
SPLITS      = DATA_ROOT / "splits"
REFERENCES  = DATA_PROJ / "references"
CHECKPOINTS = DATA_PROJ / "checkpoints"
OUTPUTS     = DATA_PROJ / "outputs"

# dataset shortcuts
CHITTAGONG1  = RAW / "chittagong-1"
CHITTAGONG2  = RAW / "chittagong-2"
TANZANIA     = RAW / "tanzania"
IBADAN_PART1 = RAW / "ibadan" / "samples_part1"
IBADAN_PART2 = RAW / "ibadan" / "samples_part2"
LABELS_CSV   = REFERENCES / "MILCA" / "data" / "tbf_samples_parasite_count_nih_ucl.csv"
