"""Exploratory diagnosis only; never imports or selects on old test labels.

Candidate ranking uses source-leave-one-out over five existing training paths.
E05 is evaluated after ranking is frozen. Other accepted records supply only
endpoint length/direction metadata for a descriptive coverage table.
"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRAIN_IDS = ["E01", "E02", "E04", "E09", "E14"]
VAL_IDS = ["E05"]
CAP_MM = 25.

def load():
    raw = json.loads((ROOT / "data/real_motion.json").read_text(encoding="utf8"))
    episodes = {r["episode_id"]: r for r in raw["episodes"]}
    groups = {}
    for source in TRAIN_IDS + VAL_IDS:
        r = episodes[source]
        assert r["status"] == "accepted"
        p = r["path"]
        t = np.asarray(p["progress"])
        xy = np.asarray(p["relative_xy_m"]) * 1000
        d = xy[-1] - xy[0]
        length = np.linalg.norm(d)
        u = d / length
        basis = np.array([u, [-u[1], u[0]]])
        residual = xy - ((1-t[:,None])*xy[0] + t[:,None]*xy[-1])
        w = np.r_[(t[1]-t[0])/2, (t[2:]-t[:-2])/2, (t[-1]-t[-2])/2]
        kinds = p["sample_kind"]
        w *= np.array([1. if k == "observed" else .25 for k in kinds])
        w /= w.sum()
        groups[source] = dict(source=source, t=t, xy=xy, d=d, length=length, u=u,
            basis=basis, residual=residual, w=w, kinds=kinds,
            frames=np.asarray(p["sample_source_frames"]), record=r)
    metadata = []
    for r in raw["episodes"]:
        if r.get("status") != "accepted":
            continue
        p = r["path"]
        start, end = np.asarray(p["relative_xy_m"])[[0,-1]]
        d = end-start
        length = float(np.linalg.norm(d))
        metadata.append(dict(source=r["episode_id"],scene_group=r["scene_group"],
            endpoint_distance_mm=1000*length,direction_deg=float(np.degrees(np.arctan2(d[1],d[0]))),
            split="train" if r["episode_id"] in TRAIN_IDS else "val" if r["episode_id"] in VAL_IDS else "previously_seen_test_metadata_only"))
    return groups, metadata

def basis_t(t, degree):
    z=2*t-1
    b=[np.ones_like(t),z,(3*z*z-1)/2]
    return 4*t[:,None]*(1-t[:,None])*np.column_stack(b[:degree+1])

def geometry(g, name):
    if name == "none": return np.empty(0)
    if name == "length": return np.array([g["length"]/50.])
    if name == "direction": return g["u"]
    if name == "displacement": return g["d"]/50.
    if name == "direction_length": return np.r_[g["u"],g["length"]/50.]
    raise ValueError(name)

def target(g, frame, normalized):
    y=g["residual"] if frame == "world" else g["residual"] @ g["basis"].T
    return y / g["length"] if normalized else y

def convert_prediction(pred, g, frame, normalized):
    if normalized: pred=pred*g["length"]
    if frame == "canonical": pred=pred @ g["basis"]
    norms=np.linalg.norm(pred,axis=1,keepdims=True)
    return pred * np.minimum(1,CAP_MM/np.maximum(norms,1e-12))

def fit_ridge(train, cfg):
    geom=np.array([geometry(g,cfg["geometry"]) for g in train])
    mean=geom.mean(0)
    std=np.maximum(geom.std(0),.1)
    def design(g):
        geo=np.r_[1.,(geometry(g,cfg["geometry"])-mean)/std]
        return (basis_t(g["t"],cfg["degree"])[:,:,None]*geo[None,None,:]).reshape(len(g["t"]),-1)
    x=np.concatenate([design(g) for g in train])
    y=np.concatenate([target(g,cfg["frame"],cfg["normalized"]) for g in train])
    w=np.concatenate([g["w"]/len(train) for g in train])
    lhs=x.T@(w[:,None]*x)+cfg["alpha"]*np.eye(x.shape[1])
    coef=np.linalg.solve(lhs,x.T@(w[:,None]*y))
    return lambda g: convert_prediction(design(g)@coef,g,cfg["frame"],cfg["normalized"]), dict(coef=coef.tolist(),mean=mean.tolist(),std=std.tolist(),parameters=int(coef.size))

def fit_kernel(train, cfg):
    # Independent low-degree shapes per real source; interpolation happens only
    # in endpoint features, with ridge shrinkage toward zero displacement.
    geometries=np.array([geometry(g,cfg["geometry"]) for g in train])
    mean=geometries.mean(0)
    std=np.maximum(geometries.std(0),.1)
    coords=(geometries-mean)/std
    coefs=[]
    for g in train:
        x=basis_t(g["t"],cfg["degree"])
        w=g["w"]
        c=np.linalg.solve(x.T@(w[:,None]*x)+.02*np.eye(x.shape[1]),x.T@(w[:,None]*target(g,cfg["frame"],cfg["normalized"])))
        coefs.append(c)
    coefs=np.array(coefs)
    def predict(g):
        coord=(geometry(g,cfg["geometry"])-mean)/std
        distances=np.sum((coords-coord)**2,axis=1)
        k=np.exp(-distances/(2*cfg["bandwidth"]**2))
        c=np.tensordot(k,coefs,axes=(0,0))/(k.sum()+cfg["shrinkage"])
        return convert_prediction(basis_t(g["t"],cfg["degree"])@c,g,cfg["frame"],cfg["normalized"])
    return predict,dict(source_coefficients=coefs.tolist(),training_geometry=geometries.tolist(),mean=mean.tolist(),std=std.tolist(),parameters=int(coefs.size))

def evaluate(pred,g):
    error=np.linalg.norm(pred-g["residual"],axis=1)
    return dict(source=g["source"],mean_mm=float(np.sum(error*g["w"])),max_mm=float(error.max()))

def train(cfg,groups):
    if cfg["family"] == "straight": return lambda g: np.zeros_like(g["residual"]),dict(parameters=0)
    return fit_kernel(groups,cfg) if cfg["family"] == "kernel" else fit_ridge(groups,cfg)

def candidates():
    result=[dict(name="straight",family="straight")]
    # Deliberately bounded grid: 36 small models, all selected using train paths.
    for frame,normalized,geoname in [("canonical",True,"length"),("canonical",False,"length"),("canonical",False,"direction"),("world",False,"none"),("world",False,"direction"),("world",False,"displacement")]:
        for degree in [0,1]:
            for alpha in [.02,.2,2.]:
                result.append(dict(name=f"ridge_{frame}_{'normalized' if normalized else 'mm'}_{geoname}_d{degree}_a{alpha}",family="ridge",frame=frame,normalized=normalized,geometry=geoname,degree=degree,alpha=alpha))
    for geoname in ["direction", "displacement"]:
        for bandwidth in [1.,2.]:
            for shrinkage in [.1,1.]:
                result.append(dict(name=f"kernel_world_mm_{geoname}_d1_b{bandwidth}_s{shrinkage}",family="kernel",frame="world",normalized=False,geometry=geoname,degree=1,bandwidth=bandwidth,shrinkage=shrinkage))
    return result

def main():
    groups,metadata=load()
    train_groups=[groups[i] for i in TRAIN_IDS]
    rows=[]
    for cfg in candidates():
        folds=[]
        for source in TRAIN_IDS:
            pred,_=train(cfg,[groups[i] for i in TRAIN_IDS if i!=source])
            folds.append(evaluate(pred(groups[source]),groups[source]))
        rows.append(dict(config=cfg,source_lopo_mean_mm=float(np.mean([r["mean_mm"] for r in folds])),folds=folds))
    rows.sort(key=lambda r:r["source_lopo_mean_mm"])
    # Freeze winner before evaluating E05 for the diagnostic report.
    winner=rows[0]["config"]
    prediction,state=train(winner,train_groups)
    frozen=dict(config=winner,state=state,selection_ids=TRAIN_IDS,selection_rule="lowest source-uniform mean LOO displacement in approximate image-plane mm",maximum_residual_mm=CAP_MM)
    (OUT/"probe_frozen_model.json").write_text(json.dumps(frozen,indent=2),encoding="utf8")
    frozen_sha=hashlib.sha256((OUT/"probe_frozen_model.json").read_bytes()).hexdigest()
    validation=[evaluate(prediction(groups[i]),groups[i]) for i in VAL_IDS]
    training=[evaluate(prediction(groups[i]),groups[i]) for i in TRAIN_IDS]
    summary=[]
    for source in TRAIN_IDS+VAL_IDS:
        g=groups[source]
        p=g["t"]
        local=g["residual"]@g["basis"].T
        frame_time=(g["frames"]-g["frames"][0])/(g["frames"][-1]-g["frames"][0])
        summary.append(dict(source=source,points=len(p),observed=g["kinds"].count("observed"),length_mm=float(g["length"]),direction_deg=float(np.degrees(np.arctan2(g["u"][1],g["u"][0]))),
            arc_to_chord=float(g["record"]["path"]["arc_length_m"]*1000/g["length"]),
            straight_mean_mm=float(np.sum(np.linalg.norm(g["residual"],axis=1)*g["w"])),
            local_mean_residual_mm=(g["w"]@local).tolist(),world_mean_residual_mm=(g["w"]@g["residual"]).tolist(),
            maximum_arc_vs_time_progress=float(np.max(abs(p-frame_time))),
            movement_duration_s=g["record"]["movement_duration_s"],
            max_gap_frames=g["record"]["quality"]["maximum_sample_gap_frames"],
            max_observed_bracket_gap_frames=g["record"]["quality"]["maximum_observed_bracket_gap_frames"]))
    pairs=[]
    for j,a in enumerate(train_groups):
        for b in train_groups[j+1:]:
            pairs.append(dict(a=a["source"],b=b["source"],length_difference_mm=abs(a["length"]-b["length"]),direction_difference_deg=float(np.degrees(np.arccos(np.clip(a["u"]@b["u"],-1,1))))))
    pairs.sort(key=lambda r:r["length_difference_mm"])
    result=dict(protocol="Exploratory only; existing old-test endpoints describe feature coverage; no old-test residual labels are consumed or evaluated. Candidates selected by five-training-source LOO; E05 diagnostic reported after frozen model write, no revision based on E05.",
        source_motion_sha256=hashlib.sha256((ROOT/"data/real_motion.json").read_bytes()).hexdigest(),
        train_ids=TRAIN_IDS,val_ids=VAL_IDS,train_points=sum(len(g["t"]) for g in train_groups),train_observed=sum(g["kinds"].count("observed") for g in train_groups),
        feature_coverage_metadata=metadata,allowed_path_summaries=summary,closest_length_pairs=pairs[:3],
        ranked_train_lopo=rows,frozen_model_sha256=frozen_sha,selected_model_training=training,selected_model_E05_validation=validation)
    (OUT/"representation_probe.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(dict(summary=summary,closest_length_pairs=pairs[:3],top_train_lopo=rows[:8],frozen_validation=validation,training=training,metadata=metadata),indent=2))

if __name__ == "__main__":
    main()
