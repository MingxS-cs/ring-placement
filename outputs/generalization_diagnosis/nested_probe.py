"""Train-source-only nested selection diagnostic; does not revise frozen probe."""
import json
import numpy as np
import representation_probe as probe

def nested(groups, configs):
    folds=[]
    for source in probe.TRAIN_IDS:
        outer_ids=[i for i in probe.TRAIN_IDS if i!=source]
        ranked=[]
        for cfg in configs:
            losses=[]
            for inner in outer_ids:
                predict,_=probe.train(cfg,[groups[i] for i in outer_ids if i!=inner])
                losses.append(probe.evaluate(predict(groups[inner]),groups[inner])["mean_mm"])
            ranked.append((float(np.mean(losses)),cfg))
        ranked.sort(key=lambda r:r[0])
        selected=ranked[0][1]
        predict,_=probe.train(selected,[groups[i] for i in outer_ids])
        result=probe.evaluate(predict(groups[source]),groups[source])
        result.update(selected=selected,inner_lopo_mm=ranked[0][0],straight_mean_mm=float(np.sum(np.linalg.norm(groups[source]["residual"],axis=1)*groups[source]["w"])))
        folds.append(result)
    return dict(candidate_count=len(configs),outer_mean_mm=float(np.mean([r["mean_mm"] for r in folds])),straight_mean_mm=float(np.mean([r["straight_mean_mm"] for r in folds])),folds=folds)

def main():
    groups,_=probe.load()
    configs=probe.candidates()
    restricted=[c for c in configs if c.get("family")=="ridge" and c.get("frame")=="canonical" and c.get("geometry")=="direction" and not c.get("normalized")]
    report=dict(protocol="Five outer training-source folds; each chooses model/hyperparameters using four inner training-source folds. Existing E05/test labels never affect either choice; no frozen-model revision. Restricted family is an explanatory sensitivity diagnostic motivated by same-length/different-direction collision, not a replacement independent benchmark.",all_candidates=nested(groups,configs),restricted_direction_ridge=nested(groups,restricted))
    (probe.OUT/"nested_probe.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
