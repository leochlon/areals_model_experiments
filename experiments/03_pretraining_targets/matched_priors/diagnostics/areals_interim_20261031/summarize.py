from pathlib import Path
import json, math
ROOT=Path("/workspace/runs/matched_priors_windsor_20261030")
OUT=ROOT/"diagnostics/areals_interim_20261031"
manifest=json.loads((OUT/"manifest.json").read_text())
trials=[]
for i in range(3):
    run=OUT/f"pressure/lr{i}"
    done=json.loads((run/"complete.json").read_text())
    val=json.loads((run/"best_validation.json").read_text())
    assert done["step"]==8000
    assert len(val["cars"])==35 and math.isfinite(val["mean_nrmse"])
    trials.append(dict(lr=done["config"]["lr"],validation_nrmse=val["mean_nrmse"],best_step=val["step"],checkpoint=str(run/"best.pt")))
selected=min(trials,key=lambda r:(r["validation_nrmse"],r["lr"]))
result=dict(scope=manifest["scope"],geometry_step=manifest["geometry_step"],windsor_training_cars=18,validation_cars=35,seed=0,field_updates=8000,selected=selected,trials=trials)
(OUT/"validation_summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
