"""Save and reload a SimulatedCommissioning object."""

from pathlib import Path

from pySC import SimulatedCommissioning, generate_SC

Path("output").mkdir(exist_ok=True)

sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)
sc.to_json("output/sc.json")
reloaded_sc = SimulatedCommissioning.from_json("output/sc.json")

print(len(sc.lattice.ring), len(reloaded_sc.lattice.ring))
