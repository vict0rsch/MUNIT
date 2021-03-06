import subprocess
from pathlib import Path
import sys

sh_file = "train_process.sh"
prefix = f"sbatch {sh_file} python train.py"

params = [{"disen_ab": 0}, {"disen_ab": 5}]

output_path = Path("/home/vsch/scratch/munit/outputs/disen_v1_shift_single")

print("Launching processes with outputs in", str(output_path))
print("Did you update slurm-output location in", sh_file, "?")
if "n" in input("[y/n] (default: y)  "):
    sys.exit()

for i, param in enumerate(params):
    path = output_path / str(i)
    if not path.exists():
        print("making dir", str(path))
        path.mkdir(parents=True)

    fix = f"--config=/home/vsch/MUNIT/scripts/floods_v1_double.yaml"
    fix += f" --output_path={str(path)}"
    fix += f" --trainer=DoubleMUNIT"

    var = " ".join(f"--{k}={v}" for k, v in param.items())
    cmd = f"{prefix} {fix} {var}"
    with (path / "command.txt").open("w") as f:
        f.write(cmd)
    print("Running ", cmd)
    subprocess.check_output(cmd, shell=True)
