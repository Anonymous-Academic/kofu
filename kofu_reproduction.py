from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, os, pickle, random, shutil, sys, time, traceback, urllib.request, warnings, zipfile, multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import OrderedDict, Counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False


MAIN11 = [
    'safe_drive','texting_right','phonecall_right','texting_left','phonecall_left',
    'radio','drinking','reach_side','hair_and_makeup','talking_to_passenger','reach_backseat'
]
NONCANONICAL3 = ['unclassified','change_gear','standstill_or_waiting']
NATIVE14 = MAIN11 + NONCANONICAL3
SAFE_LABEL = 'safe_drive'
C_PROTOCOLS = {
    'C1_phone_unknown': {
        'known': ['safe_drive','radio','drinking','reach_side','hair_and_makeup','talking_to_passenger','reach_backseat'],
        'unknown': ['texting_right','texting_left','phonecall_right','phonecall_left'],
    },
    'C2_object_control_unknown': {
        'known': ['safe_drive','texting_right','phonecall_right','texting_left','phonecall_left','hair_and_makeup','talking_to_passenger','reach_backseat'],
        'unknown': ['radio','drinking','reach_side'],
    },
    'C3_cabin_social_body_unknown': {
        'known': ['safe_drive','texting_right','phonecall_right','texting_left','phonecall_left','radio','drinking','reach_side'],
        'unknown': ['hair_and_makeup','talking_to_passenger','reach_backseat'],
    },
}
VIEWS = ('hands','face','body')
VIEW_TO_ID = {v:i for i,v in enumerate(VIEWS)}
CROPS = {
    'hands': (0,0,640,360),
    'face': (0,360,640,720),
    'body': (640,360,1280,720),
}
EXPECTED_W, EXPECTED_H = 1280, 720
VIEW_MASKS = np.asarray([
    [1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]
], dtype=np.float32)
VIEW_MASK_NAMES = ['hands_only','face_only','body_only','hands_face','hands_body','face_body','full']
D_CONDITION_NAMES = ['missing_face_body','missing_hands_body','missing_hands_face','missing_body','missing_face','missing_hands','full']
SEM_DIM = 10
SEM_GAZE = slice(0,2)
SEM_HANDS = slice(2,6)

DEFAULT_SEED = 20260821
SCRIPT_DIR = Path(__file__).resolve().parent
OFFICIAL_VIDEOMAE_V2S_URL = 'https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth'

REPRODUCIBILITY_SEED_OFFSETS = {
    'A/fold0/Falsification': 501687637,
    'A/fold1/Falsification': 1602600516,
    'A/fold2/Falsification': 2552539823,
    'A/fold3/Falsification': 4023594436,
    'A/fold4/Falsification': 3397447259,
    'C1_phone_unknown/fold0/Falsification': 2183962966,
    'C1_phone_unknown/fold1/Falsification': 1607418121,
    'C1_phone_unknown/fold2/Falsification': 2381027638,
    'C1_phone_unknown/fold3/Falsification': 1175014305,
    'C1_phone_unknown/fold4/Falsification': 1262556024,
    'C2_object_control_unknown/fold0/Falsification': 3009613787,
    'C2_object_control_unknown/fold1/Falsification': 1902650719,
    'C2_object_control_unknown/fold2/Falsification': 2431164502,
    'C2_object_control_unknown/fold3/Falsification': 2303626194,
    'C2_object_control_unknown/fold4/Falsification': 3105705273,
    'C3_cabin_social_body_unknown/fold0/Falsification': 3884674920,
    'C3_cabin_social_body_unknown/fold1/Falsification': 212473740,
    'C3_cabin_social_body_unknown/fold2/Falsification': 2588658765,
    'C3_cabin_social_body_unknown/fold3/Falsification': 450096087,
    'C3_cabin_social_body_unknown/fold4/Falsification': 3729782517,
    'EvaluationCache/fold0/A-test-cache': 596611960,
    'EvaluationCache/fold1/A-test-cache': 259360950,
    'EvaluationCache/fold2/A-test-cache': 1910296008,
    'EvaluationCache/fold3/A-test-cache': 2182796518,
    'EvaluationCache/fold4/A-test-cache': 3531121393,
    'FoundationReuse/protocol_a/fold0/tr': 3312663161,
    'FoundationReuse/protocol_a/fold0/va': 1751013039,
    'FoundationReuse/protocol_a/fold1/tr': 1024910086,
    'FoundationReuse/protocol_a/fold1/va': 2207253364,
    'FoundationReuse/protocol_a/fold2/tr': 1415841057,
    'FoundationReuse/protocol_a/fold2/va': 4087300150,
    'FoundationReuse/protocol_a/fold3/tr': 3624012600,
    'FoundationReuse/protocol_a/fold3/va': 953193514,
    'FoundationReuse/protocol_a/fold4/tr': 1048412093,
    'FoundationReuse/protocol_a/fold4/va': 1279881759,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold0/tr': 858566023,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold0/va': 2093691049,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold1/tr': 3468689993,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold1/va': 853575393,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold2/tr': 3579216407,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold2/va': 2314269727,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold3/tr': 214402384,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold3/va': 3135365459,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold4/tr': 1509223042,
    'FoundationReuse/protocol_c/C1_phone_unknown/fold4/va': 1642192104,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold0/tr': 141276321,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold0/va': 4233850720,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold1/tr': 1258158674,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold1/va': 3833616843,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold2/tr': 2701815490,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold2/va': 1735566755,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold3/tr': 847440507,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold3/va': 275655013,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold4/tr': 4244523500,
    'FoundationReuse/protocol_c/C2_object_control_unknown/fold4/va': 459881118,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold0/tr': 2845197800,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold0/va': 2555468827,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold1/tr': 2809528015,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold1/va': 4042609049,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold2/tr': 595693259,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold2/va': 938068629,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold3/tr': 4172171958,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold3/va': 918238426,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold4/tr': 618979849,
    'FoundationReuse/protocol_c/C3_cabin_social_body_unknown/fold4/va': 1046898775,
    'ProtocolB/fold0': 1541072646,
    'ProtocolB/fold1': 3839180148,
    'ProtocolB/fold2': 2277011522,
    'ProtocolB/fold3': 2493339331,
    'ProtocolB/fold4': 1403714051,
    'ProtocolC/C1_phone_unknown/fold0': 3375486568,
    'ProtocolC/C1_phone_unknown/fold1': 3620719650,
    'ProtocolC/C1_phone_unknown/fold2': 376524732,
    'ProtocolC/C1_phone_unknown/fold3': 3643604937,
    'ProtocolC/C1_phone_unknown/fold4': 883532819,
    'ProtocolC/C2_object_control_unknown/fold0': 1985281036,
    'ProtocolC/C2_object_control_unknown/fold1': 389307117,
    'ProtocolC/C2_object_control_unknown/fold2': 693025668,
    'ProtocolC/C2_object_control_unknown/fold3': 2198261193,
    'ProtocolC/C2_object_control_unknown/fold4': 3682033317,
    'ProtocolC/C3_cabin_social_body_unknown/fold0': 2162795497,
    'ProtocolC/C3_cabin_social_body_unknown/fold1': 3322174475,
    'ProtocolC/C3_cabin_social_body_unknown/fold2': 2755242726,
    'ProtocolC/C3_cabin_social_body_unknown/fold3': 571203145,
    'ProtocolC/C3_cabin_social_body_unknown/fold4': 211890316,
    'ProtocolE/fold0/KOFU_LearnedUtility': 2489165321,
    'ProtocolE/fold1/KOFU_LearnedUtility': 855371358,
    'ProtocolE/fold2/KOFU_LearnedUtility': 1336633962,
    'ProtocolE/fold3/KOFU_LearnedUtility': 2525677345,
    'ProtocolE/fold4/KOFU_LearnedUtility': 1442859157,
    'SelectiveAcquisition/fold0/train': 1494294631,
    'SelectiveAcquisition/fold1/train': 380610764,
    'SelectiveAcquisition/fold2/train': 1826395878,
    'SelectiveAcquisition/fold3/train': 177942802,
    'SelectiveAcquisition/fold4/train': 247438298,
    'protocol_a/fold0/Falsification': 2805522367,
    'protocol_a/fold1/Falsification': 4176629878,
    'protocol_a/fold2/Falsification': 3122073092,
    'protocol_a/fold3/Falsification': 4265783697,
    'protocol_a/fold4/Falsification': 2984406956,
    'protocol_c/C1_phone_unknown/fold0/Falsification': 4186867593,
    'protocol_c/C1_phone_unknown/fold1/Falsification': 1308959384,
    'protocol_c/C1_phone_unknown/fold2/Falsification': 3552165173,
    'protocol_c/C1_phone_unknown/fold3/Falsification': 4112749496,
    'protocol_c/C1_phone_unknown/fold4/Falsification': 2323299251,
    'protocol_c/C2_object_control_unknown/fold0/Falsification': 3350192815,
    'protocol_c/C2_object_control_unknown/fold1/Falsification': 2866868116,
    'protocol_c/C2_object_control_unknown/fold2/Falsification': 1779981501,
    'protocol_c/C2_object_control_unknown/fold3/Falsification': 200298707,
    'protocol_c/C2_object_control_unknown/fold4/Falsification': 1765485942,
    'protocol_c/C3_cabin_social_body_unknown/fold0/Falsification': 2223131202,
    'protocol_c/C3_cabin_social_body_unknown/fold1/Falsification': 1151725411,
    'protocol_c/C3_cabin_social_body_unknown/fold2/Falsification': 138519489,
    'protocol_c/C3_cabin_social_body_unknown/fold3/Falsification': 1424140407,
    'protocol_c/C3_cabin_social_body_unknown/fold4/Falsification': 158547604,
}

def reproduction_seed(key: str, base: int = DEFAULT_SEED) -> int:
    if key in REPRODUCIBILITY_SEED_OFFSETS:
        return int((int(base) + REPRODUCIBILITY_SEED_OFFSETS[key]) % (2**31-1))
    return stable_seed(key, base=base)


def now() -> str: return time.strftime('%Y-%m-%d %H:%M:%S')
def log(msg: str) -> None: print(f'[{now()}] {msg}', flush=True)
def ensure_dir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def stable_seed(*parts: Any, base: int = DEFAULT_SEED) -> int:
    s='|'.join(map(str,parts)).encode('utf-8'); h=hashlib.sha256(s).digest()
    return int((base + int.from_bytes(h[:4],'little')) % (2**31-1))
def seed_everything(seed: int) -> None:
    seed=int(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass

def format_seconds(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0: return '?'
    sec=int(sec); h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f'{h}h{m:02d}m' if h else (f'{m}m{s:02d}s' if m else f'{s}s')

class GlobalClock:
    def __init__(self,total:int): self.total=max(1,int(total)); self.done=0; self.durations=[]
    def complete(self,d:float): self.done+=1; self.durations.append(float(d))
    def all_eta(self,current_eta:float=0.0,current_full:float=0.0)->float:
        avg=float(np.mean(self.durations)) if self.durations else max(1.0,float(current_full))
        return max(0.0,float(current_eta)) + max(0,self.total-self.done-1)*avg

class ExperimentBar:

    def __init__(self, clock:GlobalClock, idx:int, name:str, epochs:int, eval_units:int=1):
        self.clock=clock; self.idx=idx; self.name=name; self.epochs=int(epochs); self.eval_units=int(eval_units)
        self.total=max(1,self.epochs+self.eval_units); self.start=time.time(); self.epoch_times=[]; self.eval_times=[]
        self.bar=tqdm(total=self.total, desc=f'[EXP {idx}/{clock.total}] {name}', dynamic_ncols=True, leave=True)
    def epoch(self, ep:int, sec:float, loss:float, val_f1:float, best:float):
        self.epoch_times.append(float(sec)); ae=float(np.mean(self.epoch_times)); av=float(np.mean(self.eval_times)) if self.eval_times else max(1.0,ae*.35)
        rem=max(0,self.epochs-ep)*ae + self.eval_units*av; full=self.epochs*ae+self.eval_units*av
        self.bar.set_postfix_str(f'ep={ep}/{self.epochs} loss={loss:.3f} valF1={val_f1:.3f} best={best:.3f} exp_eta={format_seconds(rem)} all_eta={format_seconds(self.clock.all_eta(rem,full))}', refresh=False)
        self.bar.update(1)
    def eval(self,phase:str,sec:float,remaining:int=0):
        self.eval_times.append(float(sec)); av=float(np.mean(self.eval_times)); rem=max(0,int(remaining))*av; ae=float(np.mean(self.epoch_times)) if self.epoch_times else av
        self.bar.set_postfix_str(f'phase={phase} exp_eta={format_seconds(rem)} all_eta={format_seconds(self.clock.all_eta(rem,self.epochs*ae+self.eval_units*av))}', refresh=False)
        self.bar.update(1)
    def close(self):

        self.bar.close(); self.clock.complete(time.time()-self.start)

def write_json(path:Path,obj:Any): ensure_dir(path.parent); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def atomic_torch_save(obj:Any,path:Path):
    ensure_dir(path.parent); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def fpr_at_tpr95(y_unknown:np.ndarray, score:np.ndarray)->float:
    y=np.asarray(y_unknown,dtype=int); s=np.asarray(score,dtype=float)
    if len(np.unique(y))<2: return float('nan')
    fpr,tpr,_=roc_curve(y,s); ids=np.where(tpr>=.95)[0]
    return float(fpr[ids[0]]) if len(ids) else 1.0

def ood_metrics(known_scores:np.ndarray, unknown_scores:np.ndarray)->Dict[str,float]:
    ks=np.asarray(known_scores,dtype=float); us=np.asarray(unknown_scores,dtype=float)
    y=np.r_[np.zeros(len(ks),dtype=int),np.ones(len(us),dtype=int)]; s=np.r_[ks,us]
    return {'auroc':float(roc_auc_score(y,s)),'aupr':float(average_precision_score(y,s)),'fpr95':fpr_at_tpr95(y,s),'n_known':int(len(ks)),'n_unknown':int(len(us))}

def energy_np(logits:np.ndarray)->np.ndarray:
    x=np.asarray(logits,dtype=np.float64); m=np.max(x,axis=1,keepdims=True); return -(m[:,0]+np.log(np.exp(x-m).sum(1)))
def msp_ood_np(logits:np.ndarray)->np.ndarray:
    x=np.asarray(logits,dtype=np.float64); x=x-x.max(1,keepdims=True); p=np.exp(x); p/=p.sum(1,keepdims=True); return -p.max(1)
def zfit(x:np.ndarray)->Tuple[float,float]: return float(np.mean(x)),float(np.std(x)+1e-8)
def zapply(x:np.ndarray,mu:float,sd:float)->np.ndarray: return (np.asarray(x)-mu)/sd

def expected_calibration_error(prob:np.ndarray,y:np.ndarray,n_bins:int=15)->float:
    prob=np.asarray(prob); y=np.asarray(y,dtype=int); conf=prob.max(1); pred=prob.argmax(1); acc=(pred==y).astype(float)
    e=0.0
    for lo,hi in zip(np.linspace(0,1,n_bins,endpoint=False),np.linspace(0,1,n_bins+1)[1:]):
        m=(conf>lo)&(conf<=hi) if lo>0 else (conf>=lo)&(conf<=hi)
        if m.any(): e += float(m.mean())*abs(float(acc[m].mean())-float(conf[m].mean()))
    return float(e)

def classification_metrics(y:np.ndarray,pred:np.ndarray,prob:np.ndarray,labels:Sequence[str])->Dict[str,float]:




    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); prob=np.asarray(prob); C=len(labels)
    if y.ndim!=1 or pred.ndim!=1 or len(y)!=len(pred):raise RuntimeError(f'Invalid metric shapes y={y.shape}, pred={pred.shape}')
    if prob.ndim!=2 or prob.shape!=(len(y),C):raise RuntimeError(f'Invalid probability shape {prob.shape}; expected {(len(y),C)}')
    cm=np.zeros((C,C),dtype=np.int64)
    valid=(y>=0)&(y<C)&(pred>=0)&(pred<C)
    if not bool(valid.all()):raise RuntimeError('Classification metric received label index outside the declared label set')
    np.add.at(cm,(y,pred),1)
    tp=np.diag(cm).astype(np.float64); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp
    fden=2*tp+fp+fn; f1=np.divide(2*tp,fden,out=np.zeros(C,dtype=np.float64),where=fden>0)
    support=cm.sum(1).astype(np.float64); rec=np.divide(tp,support,out=np.zeros(C,dtype=np.float64),where=support>0); present=support>0
    acc=float(tp.sum()/max(1,len(y))); bacc=float(rec[present].mean()) if present.any() else float('nan')
    safe=labels.index(SAFE_LABEL) if SAFE_LABEL in labels else None; fs=float('nan')
    if safe is not None:
        non=y!=safe; fs=float(((pred==safe)&non).sum()/max(1,int(non.sum())))
    return {'acc':acc,'macro_f1':float(f1.mean()),'balanced_acc':bacc,'false_safe':fs,'ece':expected_calibration_error(prob,y)}

def per_class_f1(y:np.ndarray,pred:np.ndarray,labels:Sequence[str])->Dict[str,float]:
    vals=f1_score(y,pred,labels=list(range(len(labels))),average=None,zero_division=0)
    return {f'f1_{lab}':float(v) for lab,v in zip(labels,vals)}


def find_column(df:pd.DataFrame,candidates:Sequence[str],required:bool=True)->Optional[str]:
    lower={str(c).lower():c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    if required: raise RuntimeError(f'Missing required column {list(candidates)}; existing={list(df.columns)}')
    return None

def resolve_path(root:Path,p:Any)->Path:

    root=Path(root).expanduser().resolve();pp=Path(str(p)).expanduser()
    if pp.exists():return pp.resolve()


    parts=list(pp.parts)
    for marker in (root.name,'DMD_distraction_extracted'):
        if marker in parts:
            i=parts.index(marker);cand=root/Path(*parts[i+1:])
            if cand.exists():return cand.resolve()
    if 'dmd' in parts:
        i=parts.index('dmd');cand=root/Path(*parts[i:])
        if cand.exists():return cand.resolve()
    if not pp.is_absolute():
        cand=root/pp
        if cand.exists():return cand.resolve()
        for i in range(len(parts)):
            cand=root/Path(*parts[i:])
            if cand.exists():return cand.resolve()
        return cand
    return pp

def load_interval_csv(split_dir:Path,root:Path)->pd.DataFrame:
    p=split_dir/'dmd_evidms_interval_assignments_native14_v1.csv'
    if not p.exists(): raise FileNotFoundError(f'Missing fixed interval assignment CSV: {p}')
    df=pd.read_csv(p)
    cf=find_column(df,['fold']); cs=find_column(df,['split']); cl=find_column(df,['label','action_label','class'])
    cu=find_column(df,['subject','subject_id']); cr=find_column(df,['recording_id','recording','video_id'])
    cv=find_column(df,['mosaic_path','video_path','path','avi_path','rgb_path']); cst=find_column(df,['start_frame','frame_start','fs','start','begin_frame','frame_begin','first_frame']); cen=find_column(df,['end_frame','frame_end','fe','end','stop_frame','last_frame'])
    cj=find_column(df,['json_path','annotation_path','ann_path'],required=False)
    out=df.copy(); out['_fold']=out[cf].astype(str); out['_split']=out[cs].astype(str); out['_label']=out[cl].astype(str); out['_subject']=out[cu].astype(str); out['_recording_id']=out[cr].astype(str)
    out['_video_path']=out[cv].map(lambda x:str(resolve_path(root,x))); out['_start']=out[cst].astype(int); out['_end']=out[cen].astype(int)
    bad=out['_end']<out['_start']
    if bad.any():
        a=out.loc[bad,'_start'].copy(); out.loc[bad,'_start']=out.loc[bad,'_end'].values; out.loc[bad,'_end']=a.values
    if cj is not None: out['_json_path']=out[cj].map(lambda x:str(resolve_path(root,x)))
    else: out['_json_path']=out['_video_path'].map(lambda v:str(Path(v).with_name(Path(v).name.replace('_rgb_mosaic.avi','_rgb_ann_distraction.json'))))
    missing_v=sorted({p for p in out['_video_path'].unique() if not Path(p).exists()})
    missing_j=sorted({p for p in out['_json_path'].unique() if not Path(p).exists()})
    if missing_v: raise FileNotFoundError(f'{len(missing_v)} videos unresolved; examples={missing_v[:5]}')
    if missing_j: raise FileNotFoundError(f'{len(missing_j)} annotation JSONs unresolved; examples={missing_j[:5]}')
    return out

def validate_split(df:pd.DataFrame)->None:
    folds=sorted(df['_fold'].unique().tolist())
    if folds!=[f'fold{i}' for i in range(5)]:
        raise RuntimeError(f'Expected fold0..fold4, got {folds}')
    unexpected=set(df['_label'].unique())-set(NATIVE14)
    if unexpected:
        raise RuntimeError(f'Unexpected native labels: {sorted(unexpected)}')





    id_cols=['_recording_id','_start','_end','_label']
    reference_ids=None
    for fold in folds:
        dff=df[df['_fold']==fold].copy()
        if dff.duplicated(id_cols).any():
            raise RuntimeError(f'{fold}: duplicate interval identity inside fold')
        ids=set(map(tuple,dff[id_cols].astype(object).to_numpy().tolist()))
        if len(ids)!=2275:
            raise RuntimeError(f'{fold}: expected 2275 unique native14 intervals, got {len(ids)}')
        n_main=int(dff['_label'].isin(MAIN11).sum())
        n_non=int(dff['_label'].isin(NONCANONICAL3).sum())
        if n_main!=1987 or n_non!=288:
            raise RuntimeError(f'{fold}: expected 1987 MAIN11 + 288 noncanonical intervals; got MAIN11={n_main}, noncanonical={n_non}')
        if dff['_recording_id'].astype(str).nunique()!=49:
            raise RuntimeError(f'{fold}: expected all 49 recordings, got {dff["_recording_id"].astype(str).nunique()}')
        if reference_ids is None:
            reference_ids=ids
        elif ids!=reference_ids:
            raise RuntimeError(f'{fold}: interval identity set differs from fold0; frozen split must assign the same 2275 intervals in every fold')

        parts={}
        for sp in ('train','val','test'):
            d=dff[dff['_split']==sp]
            if len(d)==0:
                raise RuntimeError(f'{fold}/{sp}: empty partition')
            parts[sp]=set(d['_subject'].astype(str))
            miss=set(MAIN11)-set(d[d['_label'].isin(MAIN11)]['_label'])
            if miss:
                raise RuntimeError(f'{fold}/{sp} missing MAIN11 classes {sorted(miss)}')
        if set(dff['_split'].unique())!={'train','val','test'}:
            raise RuntimeError(f'{fold}: unexpected split labels {sorted(dff["_split"].unique())}')
        if parts['train']&parts['val'] or parts['train']&parts['test'] or parts['val']&parts['test']:
            raise RuntimeError(f'Subject leakage in {fold}')

    if len(df)!=5*2275:
        raise RuntimeError(f'Frozen assignment table should contain exactly one row per interval per fold (11375 rows); got {len(df)}')

def validate_split_manifest(split_dir:Path,df:pd.DataFrame)->None:
    jp=Path(split_dir)/'dmd_evidms_subject_split_v1.json'
    if not jp.exists(): raise FileNotFoundError(f'Missing frozen split manifest: {jp}')
    obj=json.loads(jp.read_text(encoding='utf-8'))
    if int(obj.get('n_subjects',-1))!=14 or int(obj.get('n_recordings',-1))!=49:
        raise RuntimeError(f'Frozen split manifest count mismatch: n_subjects={obj.get("n_subjects")} n_recordings={obj.get("n_recordings")}')
    if int(obj.get('n_native14_intervals',-1))!=2275 or int(obj.get('n_main11_intervals',-1))!=1987 or int(obj.get('n_noncanonical3_intervals',-1))!=288:
        raise RuntimeError('Frozen split manifest interval counts differ from the audited counts (2275/1987/288).')
    labs=obj.get('labels',{}) or {}
    if list(labs.get('protocol_a_main11',[]))!=MAIN11: raise RuntimeError('Frozen split MAIN11 differs from embedded protocol; refusing to continue.')
    if list(labs.get('protocol_b_noncanonical3',[]))!=NONCANONICAL3: raise RuntimeError('Frozen split NONCANONICAL3 differs from embedded protocol; refusing to continue.')
    if list(labs.get('native14',[]))!=NATIVE14: raise RuntimeError('Frozen split native14 label order differs from embedded protocol; refusing to continue.')
    if set(df['_subject'].astype(str).unique())!=set(obj.get('all_subjects',[])): raise RuntimeError('CSV subject set differs from frozen split JSON.')
    splits=obj.get('splits',{}) or {}
    for fold in [f'fold{i}' for i in range(5)]:
        if fold not in splits: raise RuntimeError(f'Frozen split JSON missing {fold}')
        for sp in ('train','val','test'):
            a=set(df[(df['_fold']==fold)&(df['_split']==sp)]['_subject'].astype(str).unique()); b=set(map(str,splits[fold].get(sp,[])))
            if a!=b: raise RuntimeError(f'{fold}/{sp}: CSV subject assignment differs from frozen split JSON: csv={sorted(a)} json={sorted(b)}')
    test_counts=Counter(s for fold in [f'fold{i}' for i in range(5)] for s in map(str,splits[fold].get('test',[])))
    if set(test_counts)!=set(obj.get('all_subjects',[])) or any(v!=1 for v in test_counts.values()): raise RuntimeError(f'Frozen split test-subject rotation invalid: {dict(test_counts)}')

def filter_df(df:pd.DataFrame,fold:str,split:str,labels:Sequence[str])->pd.DataFrame:
    return df[(df['_fold']==fold)&(df['_split']==split)&(df['_label'].isin(list(labels)))].copy().sort_index().reset_index(drop=True)

def balanced_limit_df(df:pd.DataFrame,labels:Sequence[str],per_class:int,seed:int)->pd.DataFrame:
    rng=np.random.RandomState(int(seed)); pieces=[]
    for lab in labels:
        d=df[df['_label']==lab]
        if len(d):
            ids=np.arange(len(d)); rng.shuffle(ids); pieces.append(d.iloc[ids[:min(len(ids),max(1,int(per_class)))]] )
    return pd.concat(pieces,ignore_index=True) if pieces else df.iloc[:0].copy()


def parse_bool(v:Any)->bool:
    if isinstance(v,bool): return v
    if isinstance(v,(int,float,np.integer,np.floating)): return bool(v)
    s=str(v).strip().lower()
    if s in {'true','1','yes','y','t'}: return True
    if s in {'false','0','no','n','f','','none','null'}: return False
    return bool(v)
def _parse_interval(iv:Any)->Optional[Tuple[int,int]]:
    if isinstance(iv,dict): s=iv.get('frame_start',iv.get('start',iv.get('frameStart'))); e=iv.get('frame_end',iv.get('end',iv.get('frameEnd')))
    elif isinstance(iv,(list,tuple)) and len(iv)>=2: s,e=iv[0],iv[1]
    else: return None
    if s is None or e is None: return None
    return int(s),int(e)
def _norm_type(x:Any)->str: return str(x or '').strip().lower().replace('-','_').replace(' ','_')
@dataclass
class EvidenceTimeline:
    gaze:np.ndarray; hands:np.ndarray; talking:np.ndarray; objects:np.ndarray; occlusion:np.ndarray
    @property
    def nframes(self)->int: return int(len(self.gaze))

def parse_evidence_json(json_path:Path)->EvidenceTimeline:
    with open(json_path,'r',encoding='utf-8') as f: data=json.load(f)
    ol=data.get('openlabel',{}) or {}; actions=ol.get('actions',{}) or {}; objects=ol.get('objects',{}) or {}; frames=ol.get('frames',{}) or {}
    max_frame=0
    for obj in list(actions.values())+list(objects.values()):
        for iv in obj.get('frame_intervals',[]) or []:
            se=_parse_interval(iv)
            if se is not None: max_frame=max(max_frame,int(se[1]))
    for k in frames:
        try:max_frame=max(max_frame,int(k))
        except Exception:pass
    n=max_frame+1; gaze=np.full(n,-1,dtype=np.int8); hands=np.full(n,-1,dtype=np.int8); talking=np.zeros(n,dtype=np.uint8); obj_arr=np.zeros((n,3),dtype=np.uint8); occ=np.zeros((n,3),dtype=np.uint8)
    def fill(arr,ivs,val):
        for iv in ivs:
            se=_parse_interval(iv)
            if se is None: continue
            s,e=max(0,se[0]),min(len(arr)-1,se[1])
            if e>=s: arr[s:e+1]=val
    for a in actions.values():
        typ=_norm_type(a.get('type')); ivs=a.get('frame_intervals',[]) or []
        if typ.endswith('gaze_on_road/looking_road'): fill(gaze,ivs,0)
        elif typ.endswith('gaze_on_road/not_looking_road'): fill(gaze,ivs,1)
        elif typ.endswith('hands_using_wheel/both'): fill(hands,ivs,0)
        elif typ.endswith('hands_using_wheel/only_left'): fill(hands,ivs,1)
        elif typ.endswith('hands_using_wheel/only_right'): fill(hands,ivs,2)
        elif typ.endswith('hands_using_wheel/none'): fill(hands,ivs,3)
        elif typ.endswith('talking/talking'): fill(talking,ivs,1)
    obj_map={'cellphone':0,'bottle':1,'hair_comb':2,'haircomb':2}
    for o in objects.values():
        typ=_norm_type(o.get('type')).split('/')[-1]
        if typ not in obj_map: continue
        j=obj_map[typ]
        for iv in o.get('frame_intervals',[]) or []:
            se=_parse_interval(iv)
            if se is None: continue
            s,e=max(0,se[0]),min(n-1,se[1])
            if e>=s: obj_arr[s:e+1,j]=1
    stream_to_j={'hands_camera':0,'face_camera':1,'body_camera':2}
    for k,fr in frames.items():
        try:fi=int(k)
        except Exception:continue
        if not (0<=fi<n):continue
        streams=((fr.get('frame_properties') or {}).get('streams') or {})
        for sn,j in stream_to_j.items():
            sp=((streams.get(sn) or {}).get('stream_properties') or {}); oc=sp.get('occlusion',False); val=oc.get('val',False) if isinstance(oc,dict) else oc
            if parse_bool(val): occ[fi,j]=1
    return EvidenceTimeline(gaze,hands,talking,obj_arr,occ)

def build_annotation_cache(df:pd.DataFrame,cache_path:Path,overwrite:bool=False)->Dict[str,EvidenceTimeline]:
    if cache_path.exists() and not overwrite:
        try:
            with open(cache_path,'rb') as f: obj=pickle.load(f)
            expected={str(Path(v).resolve()) for v in df['_video_path'].unique()}
            if isinstance(obj,dict) and expected and expected.issubset(set(obj.keys())): return obj
        except Exception: pass
    cache={}
    for r in df[['_video_path','_json_path']].drop_duplicates().to_dict('records'):
        vp=str(Path(r['_video_path']).resolve()); cache[vp]=parse_evidence_json(Path(r['_json_path']))
    ensure_dir(cache_path.parent)
    with open(cache_path,'wb') as f: pickle.dump(cache,f,pickle.HIGHEST_PROTOCOL)
    return cache

def evidence_target_from_indices(tl:EvidenceTimeline,indices:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    idx=np.clip(np.asarray(indices,dtype=np.int64),0,tl.nframes-1); sem=np.zeros(SEM_DIM,dtype=np.float32); valid=np.zeros(2,dtype=np.float32)
    g=tl.gaze[idx]; gv=g[g>=0]
    if len(gv): valid[0]=1; sem[SEM_GAZE]=np.bincount(gv.astype(np.int64),minlength=2).astype(np.float32)/len(gv)
    h=tl.hands[idx]; hv=h[h>=0]
    if len(hv): valid[1]=1; sem[SEM_HANDS]=np.bincount(hv.astype(np.int64),minlength=4).astype(np.float32)/len(hv)
    sem[6]=float(tl.talking[idx].mean()); sem[7:10]=tl.objects[idx].mean(0).astype(np.float32); occ=tl.occlusion[idx].mean(0).astype(np.float32)
    return sem,occ,valid


def sample_indices(start:int,end:int,nframes:int,train:bool,seed:int,index:int,epoch:int=0)->np.ndarray:
    start,end=int(start),int(end)
    if end<=start:return np.full(nframes,start,dtype=np.int64)
    base=np.linspace(start,end,nframes)
    if train and end-start+1>nframes:
        rng=np.random.RandomState(int(seed)+int(epoch)*1000003+int(index)*9176); stride=max(1.0,(end-start+1)/nframes); base+=rng.uniform(-.25*stride,.25*stride,size=nframes)
    return np.sort(np.clip(np.round(base),start,end).astype(np.int64))

def resize_pad_rgb(frame:np.ndarray,size:int)->np.ndarray:
    h,w=frame.shape[:2]; scale=min(size/float(w),size/float(h)); nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
    interp=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR; x=cv2.resize(frame,(nw,nh),interpolation=interp); out=np.zeros((size,size,3),dtype=np.uint8); x0=(size-nw)//2; y0=(size-nh)//2; out[y0:y0+nh,x0:x0+nw]=x; return out

class _VideoBackend:
    def __init__(self): self.readers=OrderedDict(); self.caps=OrderedDict(); self.max_cache=4
    def read(self,path:str,idx:np.ndarray)->np.ndarray:
        if HAS_DECORD:
            try:
                vr=self.readers.pop(path,None)
                if vr is None:vr=VideoReader(path,ctx=cpu(0))
                self.readers[path]=vr
                while len(self.readers)>self.max_cache:self.readers.popitem(last=False)
                ii=np.clip(idx,0,len(vr)-1).astype(np.int64); arr=vr.get_batch(ii).asnumpy()
                return arr
            except Exception: pass
        cap=self.caps.pop(path,None)
        if cap is None:
            cap=cv2.VideoCapture(path)
            if not cap.isOpened(): raise RuntimeError(f'Cannot open video {path}')
        self.caps[path]=cap
        while len(self.caps)>self.max_cache:
            _,old=self.caps.popitem(last=False); old.release()
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); arr=[]
        for j in idx:
            jj=int(np.clip(j,0,max(0,n-1))); cap.set(cv2.CAP_PROP_POS_FRAMES,jj); ok,bgr=cap.read()
            if not ok or bgr is None: raise RuntimeError(f'OpenCV decode failed {path} frame={jj}')
            arr.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        return np.stack(arr)
    def close(self):
        for c in self.caps.values():
            try:c.release()
            except Exception:pass
        self.caps.clear(); self.readers.clear()

class DMDMultiViewDataset(Dataset):
    def __init__(self,df:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],frames:int=16,size:int=224,train:bool=False,seed:int=DEFAULT_SEED,return_evidence:bool=True,limit_per_class:int=0):
        self.df=df.reset_index(drop=True).copy(); self.labels=list(labels); self.li={x:i for i,x in enumerate(self.labels)}; self.timelines=timelines; self.frames=int(frames); self.size=int(size); self.train=bool(train); self.seed=int(seed); self.return_evidence=return_evidence; self._epoch_shared=mp.Value('i',0,lock=False); self.backend=None
        if limit_per_class>0:self.df=balanced_limit_df(self.df,self.labels,limit_per_class,self.seed)
    def set_epoch(self,e:int):self._epoch_shared.value=int(e)
    def __len__(self):return len(self.df)
    def _backend(self):
        if self.backend is None:self.backend=_VideoBackend()
        return self.backend
    def __getitem__(self,i:int)->Dict[str,Any]:
        r=self.df.iloc[int(i)]; vp=str(Path(r['_video_path']).resolve()); idx=sample_indices(r['_start'],r['_end'],self.frames,self.train,self.seed,i,int(self._epoch_shared.value)); raw=self._backend().read(vp,idx)
        if raw.ndim!=4 or raw.shape[1]!=EXPECTED_H or raw.shape[2]!=EXPECTED_W: raise RuntimeError(f'Unexpected DMD mosaic shape {raw.shape} for {vp}; expected T,720,1280,3')
        views=[]
        for v in VIEWS:
            x1,y1,x2,y2=CROPS[v]; crop=raw[:,y1:y2,x1:x2,:]; rr=np.stack([resize_pad_rgb(fr,self.size) for fr in crop]); views.append(torch.from_numpy(np.ascontiguousarray(rr.transpose(3,0,1,2))).to(torch.uint8))
        sem=np.zeros(SEM_DIM,dtype=np.float32); occ=np.zeros(3,dtype=np.float32); valid=np.zeros(2,dtype=np.float32)
        if self.return_evidence:
            sem,occ,valid=evidence_target_from_indices(self.timelines[vp],idx)
        lab=str(r['_label']); y=self.li.get(lab,-1)
        return {'x':torch.stack(views,0),'y':torch.tensor(y,dtype=torch.long),'label':lab,'semantic':torch.from_numpy(sem),'occlusion':torch.from_numpy(occ),'valid':torch.from_numpy(valid),'native_id':torch.tensor(NATIVE14.index(lab) if lab in NATIVE14 else -1),'row_index':torch.tensor(i)}
    def __del__(self):
        if self.backend is not None:self.backend.close()

class DMDMosaicBaselineDataset(Dataset):

    def __init__(self,df:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],frames:int=16,train:bool=False,seed:int=DEFAULT_SEED,return_evidence:bool=True,limit_per_class:int=0):
        self.df=df.reset_index(drop=True).copy();self.labels=list(labels);self.li={x:i for i,x in enumerate(self.labels)};self.timelines=timelines;self.frames=int(frames);self.train=bool(train);self.seed=int(seed);self.return_evidence=bool(return_evidence);self._epoch_shared=mp.Value('i',0,lock=False);self.backend=None
        if limit_per_class>0:self.df=balanced_limit_df(self.df,self.labels,limit_per_class,self.seed)
    def set_epoch(self,e:int):self._epoch_shared.value=int(e)
    def __len__(self):return len(self.df)
    def _backend(self):
        if self.backend is None:self.backend=_VideoBackend()
        return self.backend
    def __getitem__(self,i:int)->Dict[str,Any]:
        r=self.df.iloc[int(i)];vp=str(Path(r['_video_path']).resolve());idx=sample_indices(r['_start'],r['_end'],self.frames,self.train,self.seed,i,int(self._epoch_shared.value));raw=self._backend().read(vp,idx)
        if raw.ndim!=4 or raw.shape[1]!=EXPECTED_H or raw.shape[2]!=EXPECTED_W:raise RuntimeError(f'Unexpected DMD mosaic shape {raw.shape} for {vp}')
        comps=[]
        for fr in raw:
            crops=[]
            for v in VIEWS:
                x1,y1,x2,y2=CROPS[v];crops.append(fr[y1:y2,x1:x2,:])
            comp=np.concatenate(crops,axis=1)
            if comp.shape[:2]!=(360,1920):raise RuntimeError(f'Mosaic baseline composite shape error: {comp.shape}')
            comps.append(cv2.resize(comp,(112,112),interpolation=cv2.INTER_AREA))
        arr=np.stack(comps);x=torch.from_numpy(np.ascontiguousarray(arr.transpose(3,0,1,2))).to(torch.uint8)
        sem=np.zeros(SEM_DIM,dtype=np.float32);occ=np.zeros(3,dtype=np.float32);valid=np.zeros(2,dtype=np.float32)
        if self.return_evidence:sem,occ,valid=evidence_target_from_indices(self.timelines[vp],idx)
        lab=str(r['_label']);y=self.li.get(lab,-1)
        return {'x':x,'y':torch.tensor(y,dtype=torch.long),'label':lab,'semantic':torch.from_numpy(sem),'occlusion':torch.from_numpy(occ),'valid':torch.from_numpy(valid),'native_id':torch.tensor(NATIVE14.index(lab) if lab in NATIVE14 else -1),'row_index':torch.tensor(i)}
    def __del__(self):
        if self.backend is not None:self.backend.close()

def worker_init_fn(_:int):
    try:cv2.setNumThreads(0)
    except Exception:pass
    try:torch.set_num_threads(1)
    except Exception:pass

def make_loader(ds:Dataset,batch:int,workers:int,shuffle:bool,seed:int)->DataLoader:
    g=torch.Generator(); g.manual_seed(int(seed)); kw=dict(dataset=ds,batch_size=int(batch),shuffle=bool(shuffle),num_workers=max(0,int(workers)),pin_memory=False,drop_last=False,generator=g,worker_init_fn=worker_init_fn)
    if workers>0:kw.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(**kw)


def drop_path(x:torch.Tensor,drop_prob:float=0.,training:bool=False)->torch.Tensor:
    if drop_prob==0. or not training:return x
    keep=1-drop_prob; shape=(x.shape[0],)+(1,)*(x.ndim-1); rnd=keep+torch.rand(shape,dtype=x.dtype,device=x.device); rnd.floor_(); return x.div(keep)*rnd
class DropPath(nn.Module):
    def __init__(self,p:float=0.):super().__init__();self.p=float(p)
    def forward(self,x):return drop_path(x,self.p,self.training)
class MLPBlock(nn.Module):
    def __init__(self,d:int,ratio:float=4.,drop:float=0.):super().__init__();h=int(d*ratio);self.fc1=nn.Linear(d,h);self.act=nn.GELU();self.fc2=nn.Linear(h,d);self.drop=nn.Dropout(drop)
    def forward(self,x):return self.drop(self.fc2(self.act(self.fc1(x))))
class Attention(nn.Module):
    def __init__(self,d:int,heads:int,qkv_bias:bool=True,attn_drop:float=0.,proj_drop:float=0.):
        super().__init__();self.num_heads=heads;hd=d//heads;self.scale=hd**-0.5;self.qkv=nn.Linear(d,d*3,bias=False)
        self.q_bias=nn.Parameter(torch.zeros(d)) if qkv_bias else None;self.v_bias=nn.Parameter(torch.zeros(d)) if qkv_bias else None;self.attn_drop=nn.Dropout(attn_drop);self.proj=nn.Linear(d,d);self.proj_drop=nn.Dropout(proj_drop)
    def forward(self,x):
        B,N,C=x.shape; bias=None
        if self.q_bias is not None:bias=torch.cat((self.q_bias,torch.zeros_like(self.v_bias,requires_grad=False),self.v_bias))
        qkv=F.linear(x,self.qkv.weight,bias).reshape(B,N,3,self.num_heads,-1).permute(2,0,3,1,4);q,k,v=qkv[0],qkv[1],qkv[2];q=q*self.scale;attn=(q@k.transpose(-2,-1)).softmax(-1);attn=self.attn_drop(attn);x=(attn@v).transpose(1,2).reshape(B,N,C);return self.proj_drop(self.proj(x))
class ViTBlock(nn.Module):
    def __init__(self,d:int,heads:int,ratio:float=4.,drop:float=0.,attn_drop:float=0.,dp:float=0.):
        super().__init__();self.norm1=nn.LayerNorm(d,eps=1e-6);self.attn=Attention(d,heads,True,attn_drop,drop);self.drop_path=DropPath(dp) if dp>0 else nn.Identity();self.norm2=nn.LayerNorm(d,eps=1e-6);self.mlp=MLPBlock(d,ratio,drop)
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)));x=x+self.drop_path(self.mlp(self.norm2(x)));return x
class PatchEmbed3D(nn.Module):
    def __init__(self,img=224,patch=16,d=384,frames=16,tube=2):
        super().__init__();self.img_size=(img,img);self.tubelet_size=tube;self.patch_size=(patch,patch);self.num_patches=(img//patch)*(img//patch)*(frames//tube);self.proj=nn.Conv3d(3,d,kernel_size=(tube,patch,patch),stride=(tube,patch,patch))
    def forward(self,x):
        B,C,T,H,W=x.shape
        if (H,W)!=self.img_size:raise RuntimeError(f'VideoMAE expects 224x224, got {H}x{W}')
        return self.proj(x).flatten(2).transpose(1,2)
def sinusoid_table(n:int,d:int)->torch.Tensor:
    pos=np.arange(n)[:,None]; i=np.arange(d)[None,:]; angle=pos/np.power(10000,2*(i//2)/d); angle[:,0::2]=np.sin(angle[:,0::2]);angle[:,1::2]=np.cos(angle[:,1::2]);return torch.tensor(angle,dtype=torch.float32).unsqueeze(0)
class VideoMAEv2Small(nn.Module):

    def __init__(self,frames:int=16,drop_path_rate:float=.1):
        super().__init__();d=384;depth=12;heads=6;self.embed_dim=d;self.patch_embed=PatchEmbed3D(224,16,d,frames,2);self.register_buffer('pos_embed',sinusoid_table(self.patch_embed.num_patches,d),persistent=False);self.pos_drop=nn.Dropout(0.);dpr=torch.linspace(0,drop_path_rate,depth).tolist();self.blocks=nn.ModuleList([ViTBlock(d,heads,4.,0.,0.,dpr[i]) for i in range(depth)]);self.norm=nn.Identity();self.fc_norm=nn.LayerNorm(d,eps=1e-6);self.head_dropout=nn.Dropout(0.);self.head=nn.Identity();self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):nn.init.trunc_normal_(m.weight,std=.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.LayerNorm):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def forward_tokens(self,x:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        x=self.patch_embed(x);x=x+self.pos_embed.expand(x.size(0),-1,-1).type_as(x);x=self.pos_drop(x)
        for blk in self.blocks:x=blk(x)
        pooled=self.fc_norm(x.mean(1));return x,pooled
    def forward(self,x):return self.forward_tokens(x)[1]

def ensure_videomae_checkpoint(path:Path,auto_download:bool=True)->Path:
    path=Path(path).expanduser();
    if path.exists() and path.stat().st_size>10_000_000:return path
    if not auto_download:raise FileNotFoundError(f'VideoMAE V2-S checkpoint missing: {path}')
    ensure_dir(path.parent); log(f'Downloading official VideoMAE V2-S checkpoint to {path}')
    tmp=path.with_suffix(path.suffix+'.part')
    err=None
    try:
        from huggingface_hub import hf_hub_download
        src=Path(hf_hub_download(repo_id='OpenGVLab/VideoMAE2',filename='distill/vit_s_k710_dl_from_giant.pth'));shutil.copy2(src,tmp);os.replace(tmp,path);return path
    except Exception as e: err=e
    try:
        urllib.request.urlretrieve(OFFICIAL_VIDEOMAE_V2S_URL,tmp); os.replace(tmp,path);return path
    except Exception as e:
        if tmp.exists():tmp.unlink()
        raise RuntimeError(f'Official checkpoint download failed via huggingface_hub ({err!r}) and direct URL ({e!r}). Manually put vit_s_k710_dl_from_giant.pth at {path}.')

def load_videomae_pretrained(model:VideoMAEv2Small,path:Path)->Dict[str,Any]:
    obj=torch.load(path,map_location='cpu',weights_only=False); sd=obj.get('model',obj.get('module',obj)) if isinstance(obj,dict) else obj

    clean={}
    for k,v in sd.items():
        kk=k
        for pref in ('_orig_mod.','module.','backbone.'):
            if kk.startswith(pref):kk=kk[len(pref):]
        if kk.startswith('head.') or kk.startswith('head_dropout.'):continue
        clean[kk]=v
    cur=model.state_dict(); keep={k:v for k,v in clean.items() if k in cur and tuple(v.shape)==tuple(cur[k].shape)}
    missing,unexpected=model.load_state_dict(keep,strict=False); loaded=sum(v.numel() for v in keep.values()); total=sum(v.numel() for k,v in cur.items() if not k.startswith('head.'))
    ratio=loaded/max(1,total)
    if ratio<.97:raise RuntimeError(f'VideoMAE checkpoint compatibility too low: {ratio:.4f}; missing first={missing[:12]}, unexpected first={unexpected[:12]}')
    return {'loaded_parameter_ratio':ratio,'missing':list(missing),'unexpected':list(unexpected),'checkpoint_sha256':sha256_file(path)}

def normalize_videomae(x_uint8:torch.Tensor)->torch.Tensor:
    return x_uint8.float().div_(255.).sub_(.5).div_(.5)


class CheapViewEncoder(nn.Module):
    def __init__(self,d:int=128):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(3,32,(3,5,5),stride=(1,2,2),padding=(1,2,2)),nn.BatchNorm3d(32),nn.GELU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(32,64,3,stride=(2,2,2),padding=1),nn.BatchNorm3d(64),nn.GELU(),nn.Conv3d(64,d,3,stride=(2,2,2),padding=1),nn.BatchNorm3d(d),nn.GELU(),nn.AdaptiveAvgPool3d(1))
    def forward(self,x):return self.net(x).flatten(1)
class CheapOverview(nn.Module):
    def __init__(self,ncls:int,d:int=128):
        super().__init__();self.enc=CheapViewEncoder(d);self.view_embed=nn.Parameter(torch.randn(3,d)*.02);self.view_head=nn.Linear(d,ncls);self.occ_head=nn.Linear(d,1);self.fuse=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.LayerNorm(d));self.head=nn.Linear(d,ncls);self.d=d
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):


        B,V,C,T,H,W=x_uint8.shape; ids=torch.linspace(0,T-1,min(8,T),device=x_uint8.device).long(); feats=[torch.zeros(B,self.d,device=x_uint8.device) for _ in range(3)]; vlog=[torch.zeros(B,self.view_head.out_features,device=x_uint8.device) for _ in range(3)]; olog=[torch.zeros(B,device=x_uint8.device) for _ in range(3)]
        for v in active:
            xv=x_uint8[:,v].index_select(2,ids).float()/255.; xv=F.interpolate(xv,size=(xv.shape[2],112,112),mode='trilinear',align_corners=False); xv=(xv-.5)/.5; f=self.enc(xv)+self.view_embed[v];feats[v]=f;vlog[v]=self.view_head(f);olog[v]=self.occ_head(f).squeeze(-1)
        vf=torch.stack(feats,1); vl=torch.stack(vlog,1); ol=torch.stack(olog,1); g=self.fuse(vf.flatten(1));return {'global':g,'view_features':vf,'view_logits':vl,'occ_logits':ol,'logits':self.head(g)}


class EvidenceFoundation(nn.Module):

    def __init__(self,ncls:int,checkpoint:Path,frames:int=16,qnum:int=4,overview_dim:int=128):
        super().__init__();self.ncls=ncls;self.d=384;self.qnum=qnum;self.input_frames=int(frames);self.input_size=224;self.encoder=VideoMAEv2Small(frames);self.pretrain_info=load_videomae_pretrained(self.encoder,checkpoint)
        self.view_embed=nn.Parameter(torch.randn(3,self.d)*.02);self.q=nn.Parameter(torch.randn(qnum,self.d)*.02);self.ev_attn=nn.MultiheadAttention(self.d,6,batch_first=True)
        self.unc_head=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,1));self.occ_head=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,1))
        self.ev_view_score=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,1));self.fuse=nn.Sequential(nn.Linear(self.d*2,self.d),nn.GELU(),nn.LayerNorm(self.d))
        self.gaze_head=nn.Linear(self.d,2);self.hands_head=nn.Linear(self.d,4);self.talk_head=nn.Linear(self.d,1);self.object_head=nn.Linear(self.d,3)
        self.overview=CheapOverview(ncls,overview_dim);self.overview_to_d=nn.Linear(overview_dim,self.d);self.final_fuse=nn.Sequential(nn.Linear(self.d*2,self.d),nn.GELU(),nn.LayerNorm(self.d));self.final_head=nn.Linear(self.d,ncls)
    def encode_view(self,x_uint8:torch.Tensor,view_id:int)->Dict[str,torch.Tensor]:
        x=normalize_videomae(x_uint8);tokens,pool=self.encoder.forward_tokens(x);ve=self.view_embed[int(view_id)];tokens=tokens+ve;pool=pool+ve;q=self.q[None].expand(x.size(0),-1,-1)+ve;ev,_=self.ev_attn(q,tokens,tokens,need_weights=False)
        return {'pool':pool,'evidence':ev,'unc':F.softplus(self.unc_head(pool).squeeze(-1))+1e-4,'occ_logit':self.occ_head(pool).squeeze(-1)}
    def encode_all(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2))->Dict[str,Any]:
        d={};
        for v in active:d[int(v)]=self.encode_view(x_uint8[:,int(v)],int(v))
        return d
    def aggregate_evidence_tensor(self,ev_by_view:torch.Tensor,mask:torch.Tensor)->torch.Tensor:




        ev32=ev_by_view.float(); mask32=mask.float()
        score=self.ev_view_score(ev32).squeeze(-1).permute(0,2,1).float()
        score=score.masked_fill(mask32[:,None,:]<.5,-1e9)
        ew=torch.softmax(score,2)
        zero=(mask32.sum(1)==0)
        ev=(ev32.permute(0,2,1,3)*ew[:,:,:,None]).sum(2)
        ev[zero]=0
        return ev
    def semantic_prob_from_evidence(self,ev:torch.Tensor)->torch.Tensor:
        return torch.cat([F.softmax(self.gaze_head(ev[:,0]),1),F.softmax(self.hands_head(ev[:,1]),1),torch.sigmoid(self.talk_head(ev[:,3])),torch.sigmoid(self.object_head(ev[:,2]))],1)
    def aggregate(self,encoded:Dict[int,Dict[str,torch.Tensor]],overview_out:Dict[str,torch.Tensor])->Dict[str,torch.Tensor]:
        active=sorted(encoded.keys());B=overview_out['global'].shape[0];device=overview_out['global'].device
        if not active:
            z=overview_out['logits'];return {'logits':z,'base_logits':z,'fused':self.overview_to_d(overview_out['global']),'evidence':torch.zeros(B,self.qnum,self.d,device=device),'weights':torch.zeros(B,3,device=device),'unc':torch.zeros(B,3,device=device),'occ_logits':torch.zeros(B,3,device=device),'semantic_logits':None}
        pools=torch.stack([encoded[v]['pool'] for v in active],1);unc=torch.stack([encoded[v]['unc'] for v in active],1);occ=torch.stack([encoded[v]['occ_logit'] for v in active],1)
        rel_logits=-unc-F.softplus(occ);w=torch.softmax(rel_logits,1);fused_view=(pools*w[:,:,None]).sum(1)
        ev_by_view=torch.stack([encoded[v]['evidence'] for v in active],1)
        score=self.ev_view_score(ev_by_view).squeeze(-1).permute(0,2,1);ew=torch.softmax(score,2);ev=(ev_by_view.permute(0,2,1,3)*ew[:,:,:,None]).sum(2)
        basefeat=self.fuse(torch.cat([fused_view,ev.mean(1)],1));ov=self.overview_to_d(overview_out['global']);ff=self.final_fuse(torch.cat([basefeat,ov],1));z=self.final_head(ff)
        weights_full=torch.zeros(B,3,device=device);unc_full=torch.zeros(B,3,device=device);occ_full=torch.zeros(B,3,device=device)
        for j,v in enumerate(active):weights_full[:,v]=w[:,j];unc_full[:,v]=unc[:,j];occ_full[:,v]=occ[:,j]
        sem={'gaze':self.gaze_head(ev[:,0]),'hands':self.hands_head(ev[:,1]),'talk':self.talk_head(ev[:,3]).squeeze(-1),'objects':self.object_head(ev[:,2])}
        return {'logits':z,'base_logits':z,'fused':ff,'evidence':ev,'weights':weights_full,'unc':unc_full,'occ_logits':occ_full,'semantic_logits':sem}
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2))->Dict[str,torch.Tensor]:
        ov=self.overview(x_uint8,active=active);enc=self.encode_all(x_uint8,active);out=self.aggregate(enc,ov);out['overview']=ov;out['encoded']=enc;return out

class PooledFusionModel(nn.Module):




    def __init__(self,ncls:int,checkpoint:Path,mode:str,frames:int=16):
        super().__init__();self.mode=mode;self.ncls=ncls;self.d=384;self.input_frames=int(frames);self.input_size=224
        self.encoder=VideoMAEv2Small(frames);self.pretrain_info=load_videomae_pretrained(self.encoder,checkpoint)
        self.view_embed=nn.Parameter(torch.randn(3,self.d)*.02);self.head=nn.Linear(self.d,ncls)
        self.score=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,1)) if mode=='weighted_late' else None
        self.class_score=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,ncls)) if mode=='classwise' else None
        self.unc=nn.Sequential(nn.LayerNorm(self.d),nn.Linear(self.d,1)) if mode=='uncertainty' else None
        if mode not in {'logit_mean','weighted_late','classwise','uncertainty'}:raise ValueError(mode)
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):
        if len(active)==0:raise RuntimeError('PooledFusionModel requires at least one active view')
        ps=[]
        for v in active:
            _,p=self.encoder.forward_tokens(normalize_videomae(x_uint8[:,v]));ps.append(p+self.view_embed[v])
        p=torch.stack(ps,1);vl=self.head(p)
        if self.mode=='logit_mean':z=vl.mean(1);w=torch.ones(p.shape[:2],device=p.device,dtype=p.dtype)/p.shape[1]
        elif self.mode=='weighted_late':w=torch.softmax(self.score(p).squeeze(-1),1);z=(vl*w[:,:,None]).sum(1)
        elif self.mode=='classwise':cw=torch.softmax(self.class_score(p),1);z=(vl*cw).sum(1);w=cw.mean(2)
        else:
            u=F.softplus(self.unc(p).squeeze(-1))+1e-4;w=torch.softmax(-u,1);z=(vl*w[:,:,None]).sum(1)
        return {'logits':z,'view_logits':vl,'weights':w}

class FullTokenCrossViewModel(nn.Module):





    def __init__(self,ncls:int,checkpoint:Path,frames:int=16,nq:int=8):
        super().__init__();self.ncls=ncls;self.d=384;self.input_frames=int(frames);self.input_size=224
        self.encoder=VideoMAEv2Small(frames);self.pretrain_info=load_videomae_pretrained(self.encoder,checkpoint)
        self.view_embed=nn.Parameter(torch.randn(3,self.d)*.02);self.fusion_q=nn.Parameter(torch.randn(nq,self.d)*.02)
        self.cross_attn=nn.MultiheadAttention(self.d,6,batch_first=True);self.norm=nn.LayerNorm(self.d);self.cross_head=nn.Linear(self.d,ncls);self.view_head=nn.Linear(self.d,ncls)
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):
        if len(active)==0:raise RuntimeError('FullTokenCrossViewModel requires at least one active view')
        toks=[];pools=[]
        for v in active:
            t,p=self.encoder.forward_tokens(normalize_videomae(x_uint8[:,v]));ve=self.view_embed[v];toks.append(t+ve);pools.append(p+ve)
        bank=torch.cat(toks,1);q=self.fusion_q[None].expand(bank.size(0),-1,-1);q,_=self.cross_attn(q,bank,bank,need_weights=False);q=self.norm(q)
        z=self.cross_head(q.mean(1));pp=torch.stack(pools,1);vl=self.view_head(pp);return {'logits':z,'view_logits':vl,'features':pp}

class FixedViewVideoMAE(nn.Module):

    def __init__(self,ncls:int,checkpoint:Path,view_id:int,frames:int=16):
        super().__init__();self.view_id=int(view_id);self.input_frames=int(frames);self.input_size=224;self.d=384
        self.encoder=VideoMAEv2Small(frames);self.pretrain_info=load_videomae_pretrained(self.encoder,checkpoint);self.view_embed=nn.Parameter(torch.randn(self.d)*.02);self.head=nn.Linear(self.d,ncls)
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):
        _,p=self.encoder.forward_tokens(normalize_videomae(x_uint8[:,self.view_id]));p=p+self.view_embed;return {'logits':self.head(p),'features':p}


def replace_last_linear(root:nn.Module)->int:
    name=None;mod=None
    for n,m in root.named_modules():
        if isinstance(m,nn.Linear):name=n;mod=m
    if name is None:raise RuntimeError('No Linear classifier found')
    parent=root;parts=name.split('.')
    for p in parts[:-1]:parent=parent._modules[p]
    parent._modules[parts[-1]]=nn.Identity();return int(mod.in_features)
class TorchvisionThreeView(nn.Module):
    def __init__(self,name:str,ncls:int):
        super().__init__();self.name=name
        if name=='r3d18':
            from torchvision.models.video import r3d_18,R3D_18_Weights;self.weights=R3D_18_Weights.KINETICS400_V1;self.base=r3d_18(weights=self.weights);self.input_frames=16;self.input_size=224
        elif name=='r2plus1d18':
            from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights;self.weights=R2Plus1D_18_Weights.KINETICS400_V1;self.base=r2plus1d_18(weights=self.weights);self.input_frames=16;self.input_size=224
        elif name=='swin3d_t':
            from torchvision.models.video import swin3d_t,Swin3D_T_Weights;self.weights=Swin3D_T_Weights.KINETICS400_V1;self.base=swin3d_t(weights=self.weights);self.input_frames=16;self.input_size=224
        elif name=='mvit_v2_s':
            from torchvision.models.video import mvit_v2_s,MViT_V2_S_Weights;self.weights=MViT_V2_S_Weights.KINETICS400_V1;self.base=mvit_v2_s(weights=self.weights);self.input_frames=16;self.input_size=224
        else:raise ValueError(name)
        d=replace_last_linear(self.base);self.head=nn.Linear(d,ncls);self.d=d;tr=self.weights.transforms();self.register_buffer('mean',torch.tensor(getattr(tr,'mean',[.45,.45,.45])).view(1,3,1,1,1));self.register_buffer('std',torch.tensor(getattr(tr,'std',[.225,.225,.225])).view(1,3,1,1,1))
    def encode(self,xu):
        x=xu.float()/255.;x=(x-self.mean)/self.std;f=self.base(x);return f.flatten(1) if f.ndim>2 else f
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):
        feats=[self.encode(x_uint8[:,v]) for v in active];f=torch.stack(feats,1);vl=self.head(f);return {'logits':vl.mean(1),'view_logits':vl,'features':f}

class PyTorchVideoThreeView(nn.Module):
    def __init__(self,name:str,ncls:int):
        super().__init__();self.name=name
        try:
            if name=='slowfast_r50':
                from pytorchvideo.models.hub import slowfast_r50;self.base=slowfast_r50(pretrained=True);self.kind='slowfast';self.input_frames=32;self.input_size=224
            elif name=='x3d_s':
                from pytorchvideo.models.hub import x3d_s;self.base=x3d_s(pretrained=True);self.kind='x3d';self.input_frames=16;self.input_size=224
            elif name=='x3d_m':
                from pytorchvideo.models.hub import x3d_m;self.base=x3d_m(pretrained=True);self.kind='x3d';self.input_frames=16;self.input_size=224
            else:raise ValueError(name)
        except Exception as e:raise RuntimeError(f'pytorchvideo model {name} unavailable: {e!r}')
        d=replace_last_linear(self.base);self.head=nn.Linear(d,ncls);self.register_buffer('mean',torch.tensor([.45,.45,.45]).view(1,3,1,1,1));self.register_buffer('std',torch.tensor([.225,.225,.225]).view(1,3,1,1,1))
    def encode(self,xu):
        x=xu.float()/255.;x=(x-self.mean)/self.std
        if self.kind=='slowfast':
            t=x.shape[2];idx=torch.linspace(0,t-1,max(1,t//4),device=x.device).long();f=self.base([x.index_select(2,idx),x])
        else:f=self.base(x)
        if isinstance(f,(tuple,list)):f=f[0]
        return f.flatten(1) if f.ndim>2 else f
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):
        feats=[self.encode(x_uint8[:,v]) for v in active];f=torch.stack(feats,1);vl=self.head(f);return {'logits':vl.mean(1),'view_logits':vl,'features':f}

class MosaicR3D18Baseline(nn.Module):
    def __init__(self,ncls:int):
        super().__init__();from torchvision.models.video import r3d_18,R3D_18_Weights;self.weights=R3D_18_Weights.KINETICS400_V1;self.base=r3d_18(weights=self.weights);self.input_frames=16;self.input_size=112;self.mosaic_baseline=True;d=replace_last_linear(self.base);self.head=nn.Linear(d,ncls);tr=self.weights.transforms();self.register_buffer('mean',torch.tensor(getattr(tr,'mean',[.45,.45,.45])).view(1,3,1,1,1));self.register_buffer('std',torch.tensor(getattr(tr,'std',[.225,.225,.225])).view(1,3,1,1,1))
    def forward(self,x_uint8:torch.Tensor,active:Sequence[int]=(0,1,2)):

        if x_uint8.ndim!=5:raise RuntimeError(f'Mosaic baseline expects B,C,T,H,W; got {tuple(x_uint8.shape)}')
        x=x_uint8.float()/255.;x=(x-self.mean)/self.std;f=self.base(x);f=f.flatten(1) if f.ndim>2 else f;return {'logits':self.head(f),'features':f}

def model_data_spec(model:nn.Module,args:argparse.Namespace)->Tuple[int,int,bool]:
    return int(getattr(model,'input_frames',args.frames)),int(getattr(model,'input_size',args.size)),bool(getattr(model,'mosaic_baseline',False))

def make_model_dataset(model:nn.Module,df:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],train:bool,seed:int,args:argparse.Namespace,limit:int=0)->Dataset:
    frames,size,mosaic_baseline=model_data_spec(model,args)
    if mosaic_baseline:return DMDMosaicBaselineDataset(df,labels,timelines,frames,train,seed,True,limit)
    return DMDMultiViewDataset(df,labels,timelines,frames,size,train,seed,True,limit)


def action_class_weights(df:pd.DataFrame,labels:Sequence[str])->torch.Tensor:
    c=np.array([(df['_label']==l).sum() for l in labels],dtype=np.float64);c=np.maximum(c,1);w=c.sum()/(len(labels)*c);w/=w.mean();return torch.tensor(w,dtype=torch.float32)
def evidence_pos_weights(df:pd.DataFrame,timelines:Dict[str,EvidenceTimeline],frames:int)->Tuple[torch.Tensor,torch.Tensor]:
    ss=[];oo=[]
    for _,r in df.iterrows():
        idx=np.linspace(int(r['_start']),int(r['_end']),frames).round().astype(np.int64);s,o,_=evidence_target_from_indices(timelines[str(Path(r['_video_path']).resolve())],idx);ss.append(s);oo.append(o)
    sa=np.stack(ss);oa=np.stack(oo);b=sa[:,6:10];pos=b.sum(0);neg=len(b)-pos;pw=np.clip(neg/np.maximum(pos,1),1,15);po=oa.sum(0);no=len(oa)-po;pwo=np.clip(no/np.maximum(po,1),1,20);return torch.tensor(pw,dtype=torch.float32),torch.tensor(pwo,dtype=torch.float32)
def soft_ce(logits:torch.Tensor,target:torch.Tensor,valid:torch.Tensor)->torch.Tensor:
    loss=-(target*F.log_softmax(logits,1)).sum(1);den=valid.sum().clamp_min(1.);return (loss*valid).sum()/den
def foundation_loss(out:Dict[str,Any],batch:Dict[str,Any],class_w:Optional[torch.Tensor],pos_sem:torch.Tensor,pos_occ:torch.Tensor,lambda_ev:float=.25,lambda_occ:float=.05,lambda_overview:float=.15)->Tuple[torch.Tensor,Dict[str,float]]:
    y=batch['y'];la=F.cross_entropy(out['logits'],y,weight=class_w);sem=batch['semantic'];valid=batch['valid'];occ=batch['occlusion'];sl=out['semantic_logits']
    if sl is None:lev=torch.tensor(0.,device=y.device)
    else:
        lg=soft_ce(sl['gaze'],sem[:,0:2],valid[:,0]);lh=soft_ce(sl['hands'],sem[:,2:6],valid[:,1]);bl=torch.cat([sl['talk'][:,None],sl['objects']],1);lb=F.binary_cross_entropy_with_logits(bl,sem[:,6:10],pos_weight=pos_sem);lev=lg+lh+lb
    lo_detail=F.binary_cross_entropy_with_logits(out['occ_logits'],occ,pos_weight=pos_occ);lo_overview=F.binary_cross_entropy_with_logits(out['overview']['occ_logits'],occ,pos_weight=pos_occ);lo=.5*(lo_detail+lo_overview)
    lov_global=F.cross_entropy(out['overview']['logits'],y,weight=class_w);ov_v=out['overview']['view_logits'];lov_views=sum(F.cross_entropy(ov_v[:,v],y,weight=class_w) for v in range(3))/3.;lov=lov_global+.5*lov_views
    loss=la+lambda_ev*lev+lambda_occ*lo+lambda_overview*lov
    return loss,{'action':float(la.detach()),'evidence':float(lev.detach()),'occlusion':float(lo.detach()),'overview':float(lov.detach()),'overview_global':float(lov_global.detach()),'overview_views':float(lov_views.detach())}

def move_batch(batch:Dict[str,Any],device:torch.device)->Dict[str,Any]:
    out={}
    for k,v in batch.items():out[k]=v.to(device,non_blocking=False) if torch.is_tensor(v) else v
    return out

def train_classifier(model:nn.Module,train_df:pd.DataFrame,val_df:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],out_dir:Path,args:argparse.Namespace,clock:GlobalClock,exp_idx:int,exp_name:str,foundation:bool=False,epochs_override:Optional[int]=None)->Path:
    ensure_dir(out_dir);done=out_dir/'DONE.json';bestp=out_dir/'best.pt';lastp=out_dir/'last.pt';logp=out_dir/'train_log.csv'
    if done.exists() and bestp.exists() and not args.overwrite:
        clock.complete(0.01);return bestp
    seed=stable_seed(exp_name,base=args.seed);seed_everything(seed);dry=args.dry_run;limit=args.dry_per_class if dry else 0
    trds=make_model_dataset(model,train_df,labels,timelines,True,seed,args,limit);vads=make_model_dataset(model,val_df,labels,timelines,False,seed,args,limit);trl=make_loader(trds,args.batch,0 if dry else args.workers,True,seed);val=make_loader(vads,args.eval_batch,0 if dry else args.workers,False,seed)
    device=torch.device(args.device);model=model.to(device);cw=action_class_weights(train_df,labels).to(device)
    if foundation:
        frames,_,_=model_data_spec(model,args);ps,po=evidence_pos_weights(train_df,timelines,frames);ps=ps.to(device);po=po.to(device)
    else:
        ps=torch.ones(4,device=device);po=torch.ones(3,device=device)

    enc_params=[];head_params=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (enc_params if ('encoder.' in n or n.startswith('base.')) else head_params).append(p)
    groups=[]
    if enc_params:groups.append({'params':enc_params,'lr':args.lr_backbone})
    if head_params:groups.append({'params':head_params,'lr':args.lr_head})
    opt=torch.optim.AdamW(groups,weight_decay=args.weight_decay,betas=(.9,.999));scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    start=1;best=-1.;history=[]
    if args.resume and lastp.exists() and not args.overwrite:
        ck=torch.load(lastp,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start=int(ck['epoch'])+1;best=float(ck['best']);history=list(ck.get('history',[]));log(f'Resume {exp_name} from epoch {start}')
    epochs=1 if dry else int(epochs_override if epochs_override is not None else args.epochs);bar=ExperimentBar(clock,exp_idx,exp_name,epochs,0)
    try:
        for ep in range(start,epochs+1):
            if hasattr(trds,'set_epoch'):trds.set_epoch(ep)
            t0=time.time();model.train();losses=[]
            for batch in trl:
                batch=move_batch(batch,device);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=args.amp and device.type=='cuda'):
                    out=model(batch['x'])
                    if foundation:loss,parts=foundation_loss(out,batch,cw,ps,po,args.lambda_evidence,args.lambda_occlusion,args.lambda_overview)
                    else:loss=F.cross_entropy(out['logits'],batch['y'],weight=cw);parts={'action':float(loss.detach())}
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
                if dry:break
            vm=evaluate_classifier(model,val,labels,device,args.amp,foundation=foundation,max_batches=1 if dry else 0);vf=vm['metrics']['macro_f1'];row={'epoch':ep,'loss':float(np.mean(losses)),'val_macro_f1':vf,'val_acc':vm['metrics']['acc'],'val_false_safe':vm['metrics']['false_safe'],'sec':time.time()-t0,**{f'loss_{k}':v for k,v in parts.items()}};history.append(row)
            if vf>best:
                best=vf;atomic_torch_save({'state_dict':model.state_dict(),'labels':list(labels),'best_val_macro_f1':best,'epoch':ep,'seed':seed,'foundation':foundation},bestp)
            atomic_torch_save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'labels':list(labels),'best':best,'epoch':ep,'history':history},lastp);pd.DataFrame(history).to_csv(logp,index=False);bar.epoch(ep,row['sec'],row['loss'],vf,best)
        if not bestp.exists():raise RuntimeError(f'No best checkpoint for {exp_name}')
        write_json(done,{'experiment':exp_name,'completed':True,'best_val_macro_f1':best,'duration_sec':time.time()-bar.start,'seed':seed});return bestp
    finally:bar.close();del trl,val;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_classifier(model:nn.Module,loader:DataLoader,labels:Sequence[str],device:torch.device,amp:bool,active:Sequence[int]=(0,1,2),foundation:bool=False,max_batches:int=0,return_aux:bool=True)->Dict[str,Any]:
    model.eval();ys=[];logits=[];sem=[];occ=[];valid=[];contr=[];evs=[];fused=[];over=[]
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device)
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            try:o=model(b['x'],active=active)
            except TypeError:o=model(b['x'])
        z=o['logits'];ys.append(b['y'].cpu().numpy());logits.append(z.float().cpu().numpy());sem.append(b['semantic'].cpu().numpy());occ.append(b['occlusion'].cpu().numpy());valid.append(b['valid'].cpu().numpy())
        if 'contradiction' in o:contr.append(o['contradiction'].float().cpu().numpy())
        if 'evidence' in o:evs.append(o['evidence'].float().cpu().numpy())
        if 'fused' in o:fused.append(o['fused'].float().cpu().numpy())
        if 'overview' in o:over.append(o['overview']['global'].float().cpu().numpy())
    y=np.concatenate(ys);zz=np.concatenate(logits);prob=torch.from_numpy(zz).softmax(1).numpy();pred=zz.argmax(1);m=classification_metrics(y,pred,prob,labels)
    return {'metrics':m,'y':y,'logits':zz,'prob':prob,'pred':pred,'semantic':np.concatenate(sem),'occlusion':np.concatenate(occ),'valid':np.concatenate(valid),'contradiction':np.concatenate(contr) if contr else None,'evidence':np.concatenate(evs) if evs else None,'fused':np.concatenate(fused) if fused else None,'overview':np.concatenate(over) if over else None}

@torch.no_grad()
def collect_logits_only(model:nn.Module,loader:DataLoader,device:torch.device,amp:bool,active:Sequence[int]=(0,1,2),max_batches:int=0)->np.ndarray:

    model.eval();out=[]
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device)
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            try:o=model(b['x'],active=active)
            except TypeError:o=model(b['x'])
        out.append(o['logits'].float().cpu().numpy())
    if not out:raise RuntimeError('No batches available while collecting logits')
    return np.concatenate(out)

def load_model_state(model:nn.Module,ckpt:Path,device:torch.device)->Dict[str,Any]:
    ck=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True);model.to(device);model.eval();return ck


class SelectiveFusionHead(nn.Module):
    def __init__(self,ncls:int,overview_dim:int=128,detail_dim:int=384,hid:int=256):
        super().__init__();self.proj=nn.Linear(detail_dim,overview_dim);self.view_embed=nn.Parameter(torch.randn(3,overview_dim)*.02);self.net=nn.Sequential(nn.Linear(overview_dim*2+3,hid),nn.GELU(),nn.LayerNorm(hid),nn.Linear(hid,ncls))
    def forward(self,overview:torch.Tensor,pools:torch.Tensor,mask:torch.Tensor):

        pp=self.proj(pools)+self.view_embed[None];den=mask.sum(1,keepdim=True).clamp_min(1.);agg=(pp*mask[:,:,None]).sum(1)/den;agg=torch.where((mask.sum(1,keepdim=True)>0),agg,torch.zeros_like(agg));return self.net(torch.cat([overview,agg,mask],1))
class SelectiveEvidenceAcquisitionPolicy(nn.Module):
    def __init__(self,ncls:int,overview_dim:int=128,hid:int=256):
        super().__init__();self.net=nn.Sequential(nn.Linear(overview_dim*4+ncls+3,hid),nn.GELU(),nn.LayerNorm(hid),nn.Linear(hid,3))
    def forward(self,overview_global:torch.Tensor,overview_views:torch.Tensor,current_logits:torch.Tensor,mask:torch.Tensor):
        return self.net(torch.cat([overview_global,overview_views.flatten(1),current_logits,mask],1))


def make_light_zip(out:Path,script_path:Path)->Path:
    zp=out.with_name(out.name+'_results.zip');
    if zp.exists():zp.unlink()
    keep={'.csv','.json','.txt','.log','.py'}
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(script_path,arcname='CODE/'+script_path.name)
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix.lower() in keep:z.write(p,arcname=str(p.relative_to(out.parent)))
    return zp

def common_parser(description:str)->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--root',default='../dataset/DMD_distraction_extracted');p.add_argument('--split_dir',default='dmd_evidms_final_split_v1');p.add_argument('--out_dir',default='')
    p.add_argument('--checkpoint',default='checkpoints/vit_s_k710_dl_from_giant.pth');p.add_argument('--no_auto_download',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4);p.add_argument('--batch',type=int,default=2);p.add_argument('--eval_batch',type=int,default=2);p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--frames',type=int,default=16);p.add_argument('--size',type=int,default=224);p.add_argument('--lr_backbone',type=float,default=2e-5);p.add_argument('--lr_head',type=float,default=2e-4);p.add_argument('--weight_decay',type=float,default=.05);p.add_argument('--grad_clip',type=float,default=5.)
    p.add_argument('--lambda_evidence',type=float,default=.25);p.add_argument('--lambda_occlusion',type=float,default=.05);p.add_argument('--lambda_overview',type=float,default=.15)
    p.add_argument('--seed',type=int,default=DEFAULT_SEED);p.add_argument('--amp',action='store_true',default=True);p.add_argument('--no_amp',dest='amp',action='store_false');p.add_argument('--resume',action='store_true',default=True);p.add_argument('--no_resume',dest='resume',action='store_false');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry_run',action='store_true');p.add_argument('--dry_per_class',type=int,default=2);p.add_argument('--overwrite_annotation_cache',action='store_true')
    return p

def dryrun_lock_path(args:argparse.Namespace)->Path:
    return Path(str(Path(args.out_dir).expanduser())+'_DRYRUN')/'DRY_RUN_OK.json'

def dryrun_signature(args:argparse.Namespace,script_path:Path,split_csv:Path,checkpoint:Optional[Path],extra:Optional[Dict[str,Any]]=None)->Dict[str,Any]:

    ignored={'dry_run','dry_per_class','resume','overwrite','overwrite_annotation_cache','workers','out_dir'}
    scientific_args={k:v for k,v in vars(args).items() if k not in ignored}
    sig={'script_sha256':sha256_file(script_path),'split_sha256':sha256_file(split_csv),'checkpoint_sha256':sha256_file(checkpoint) if checkpoint is not None else None,'scientific_args':scientific_args}
    if extra:sig.update(extra)
    return sig

def require_matching_dryrun(args:argparse.Namespace,script_path:Path,split_csv:Path,checkpoint:Optional[Path],extra:Optional[Dict[str,Any]]=None)->None:
    lp=dryrun_lock_path(args)
    if not lp.exists():raise RuntimeError(f'Full run is locked until this exact script/config passes --dry_run. Missing: {lp}')
    old=json.loads(lp.read_text(encoding='utf-8'));cur=dryrun_signature(args,script_path,split_csv,checkpoint,extra)
    if old.get('signature')!=cur:raise RuntimeError(f'Dry-run lock does not match current script/config. Re-run with --dry_run first. Lock={lp}')

def write_dryrun_lock(args:argparse.Namespace,script_path:Path,split_csv:Path,checkpoint:Optional[Path],extra:Optional[Dict[str,Any]]=None)->None:
    lp=dryrun_lock_path(args);ensure_dir(lp.parent);write_json(lp,{'status':'DRY_RUN_OK','created_at':now(),'signature':dryrun_signature(args,script_path,split_csv,checkpoint,extra)})

def preflight(args:argparse.Namespace,script_path:Path,need_checkpoint:bool=True)->Tuple[Path,Path,pd.DataFrame,Dict[str,EvidenceTimeline],Optional[Path]]:
    if args.frames!=16:raise RuntimeError('This fixed VideoMAE V2-S protocol requires --frames 16.')
    if args.size!=224:raise RuntimeError('This fixed main protocol requires --size 224.')
    root=Path(args.root).expanduser().resolve();split=Path(args.split_dir).expanduser().resolve();
    if not root.exists():raise FileNotFoundError(f'Dataset root missing: {root}')
    if not split.exists():raise FileNotFoundError(f'Split dir missing: {split}')
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError(f'CUDA device requested ({args.device}) but torch.cuda.is_available() is False')
        free_b,total_b=torch.cuda.mem_get_info(device);log(f'GPU preflight: {torch.cuda.get_device_name(device)} free={free_b/2**30:.1f}GiB total={total_b/2**30:.1f}GiB')
    df=load_interval_csv(split,root);validate_split(df);validate_split_manifest(split,df)
    if (df['_start']<0).any() or (df['_end']<0).any(): raise RuntimeError('Negative frame index found in fixed split CSV')

    r=df.iloc[0];cap=cv2.VideoCapture(str(r['_video_path']));ok,bgr=cap.read();cap.release()
    if not ok or bgr is None:raise RuntimeError(f'Preflight cannot decode {r["_video_path"]}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if rgb.shape[:2]!=(720,1280):raise RuntimeError(f'Expected DMD mosaic 1280x720, got {rgb.shape}')
    recon=np.zeros_like(rgb);recon[0:360,0:640]=rgb[0:360,0:640];recon[360:720,0:640]=rgb[360:720,0:640];recon[360:720,640:1280]=rgb[360:720,640:1280]

    if not np.array_equal(recon[0:360,0:640],rgb[0:360,0:640]) or not np.array_equal(recon[360:720],rgb[360:720]):raise RuntimeError('Crop pixel audit failed')
    out_base=Path(args.out_dir).expanduser() if args.out_dir else Path('runs')/('unnamed')
    out=Path(str(out_base)+('_DRYRUN' if args.dry_run else ''));ensure_dir(out)
    cache=out/'annotation_cache.pkl';timelines=build_annotation_cache(df,cache,args.overwrite_annotation_cache)
    ckpt=None
    if need_checkpoint:
        ckpt=ensure_videomae_checkpoint(Path(args.checkpoint),not args.no_auto_download);tmp=VideoMAEv2Small(args.frames);info=load_videomae_pretrained(tmp,ckpt);del tmp;log(f'VideoMAE V2-S checkpoint OK, loaded ratio={info["loaded_parameter_ratio"]:.4f}')

    ds=DMDMultiViewDataset(filter_df(df,'fold0','train',MAIN11).iloc[:1],MAIN11,timelines,args.frames,args.size,False,args.seed,True);b=ds[0]
    if tuple(b['x'].shape)!=(3,3,16,224,224):raise RuntimeError(f'Decode shape mismatch {tuple(b["x"].shape)}')
    log(f'PREFLIGHT OK: fixed split, no subject leakage, crop=hands/face/body only, decoded={tuple(b["x"].shape)}, decord={HAS_DECORD}')
    return root,split,df,timelines,ckpt


@torch.no_grad()
def extract_foundation_cache(model:EvidenceFoundation,loader:DataLoader,device:torch.device,amp:bool,max_batches:int=0)->Dict[str,np.ndarray]:
    model.eval();ys=[];og=[];ovf=[];ovl=[];ovocc=[];pools=[];evs=[];unc=[];occ=[];sem=[];valid=[];base_logits=[];agg_evidence=[];sem_pred=[];rel_weights=[]
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device);x=b['x']
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            o=model.overview(x); enc=model.encode_all(x,(0,1,2)); agg=model.aggregate(enc,o)
        pp=[];ee=[];uu=[];oo=[]
        for v in range(3):pp.append(enc[v]['pool']);ee.append(enc[v]['evidence']);uu.append(enc[v]['unc']);oo.append(enc[v]['occ_logit'])
        sl=agg['semantic_logits']; sp=torch.cat([F.softmax(sl['gaze'],1),F.softmax(sl['hands'],1),torch.sigmoid(sl['talk'])[:,None],torch.sigmoid(sl['objects'])],1)
        ys.append(b['y'].cpu().numpy());og.append(o['global'].float().cpu().numpy());ovf.append(o['view_features'].float().cpu().numpy());ovl.append(o['view_logits'].float().cpu().numpy());ovocc.append(o['occ_logits'].float().cpu().numpy());pools.append(torch.stack(pp,1).float().cpu().numpy());evs.append(torch.stack(ee,1).float().cpu().numpy());unc.append(torch.stack(uu,1).float().cpu().numpy());occ.append(torch.stack(oo,1).float().cpu().numpy());sem.append(b['semantic'].cpu().numpy());valid.append(b['valid'].cpu().numpy());base_logits.append(agg['logits'].float().cpu().numpy());agg_evidence.append(agg['evidence'].float().cpu().numpy());sem_pred.append(sp.float().cpu().numpy());rel_weights.append(agg['weights'].float().cpu().numpy())
    return {'y':np.concatenate(ys),'overview':np.concatenate(og),'overview_views':np.concatenate(ovf),'overview_view_logits':np.concatenate(ovl),'overview_occ_logits':np.concatenate(ovocc),'pools':np.concatenate(pools),'evidence_by_view':np.concatenate(evs),'unc':np.concatenate(unc),'occ_logits':np.concatenate(occ),'semantic':np.concatenate(sem),'valid':np.concatenate(valid),'base_logits':np.concatenate(base_logits),'agg_evidence':np.concatenate(agg_evidence),'semantic_pred':np.concatenate(sem_pred),'rel_weights':np.concatenate(rel_weights)}

def all_subset_masks(include_zero:bool=True)->np.ndarray:
    out=[]
    if include_zero:out.append([0,0,0])
    out.extend(VIEW_MASKS.astype(int).tolist())
    return np.asarray(out,dtype=np.float32)

def train_selective_head(cache_tr:Dict[str,np.ndarray],cache_va:Dict[str,np.ndarray],ncls:int,out_dir:Path,args:argparse.Namespace,clock:GlobalClock,exp_idx:int,exp_name:str,epochs:int=20)->Path:
    ensure_dir(out_dir);bp=out_dir/'best.pt';done=out_dir/'DONE.json'
    if bp.exists() and done.exists() and not args.overwrite:clock.complete(.01);return bp
    seed=stable_seed(exp_name,base=args.seed);seed_everything(seed);device=torch.device(args.device);head=SelectiveFusionHead(ncls).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01);best=-1.;hist=[];masks=all_subset_masks(True);nep=1 if args.dry_run else int(epochs);bar=ExperimentBar(clock,exp_idx,exp_name,nep,0)
    Xo=torch.tensor(cache_tr['overview'],dtype=torch.float32);Xp=torch.tensor(cache_tr['pools'],dtype=torch.float32);Y=torch.tensor(cache_tr['y'],dtype=torch.long);N=len(Y);bs=256
    try:
        for ep in range(1,nep+1):
            t0=time.time();head.train();order=torch.randperm(N);losses=[]
            for st in range(0,N,bs):
                ids=order[st:st+bs];o=Xo[ids].to(device);p=Xp[ids].to(device);y=Y[ids].to(device);rng=np.random.RandomState(seed+ep*1009+st);mi=rng.randint(0,len(masks),size=len(ids));m=torch.tensor(masks[mi],dtype=torch.float32,device=device);opt.zero_grad(set_to_none=True);z=head(o,p,m);loss=F.cross_entropy(z,y);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step();losses.append(float(loss.detach()))
                if args.dry_run:break

            head.eval();vals=[]
            with torch.no_grad():
                vo=torch.tensor(cache_va['overview'],dtype=torch.float32,device=device);vp=torch.tensor(cache_va['pools'],dtype=torch.float32,device=device);vy=np.asarray(cache_va['y'])
                for m0 in masks:
                    mm=torch.tensor(np.repeat(m0[None],len(vy),0),dtype=torch.float32,device=device);z=head(vo,vp,mm).cpu().numpy();vals.append(f1_score(vy,z.argmax(1),labels=list(range(ncls)),average='macro',zero_division=0))
            vf=float(np.mean(vals));hist.append({'epoch':ep,'loss':float(np.mean(losses)),'val_subset_macro_f1_mean':vf,'sec':time.time()-t0})
            if vf>best:best=vf;atomic_torch_save({'state_dict':head.state_dict(),'best':best,'epoch':ep,'seed':seed},bp)
            pd.DataFrame(hist).to_csv(out_dir/'train_log.csv',index=False);bar.epoch(ep,hist[-1]['sec'],hist[-1]['loss'],vf,best)
        write_json(done,{'completed':True,'best':best,'seed':seed});return bp
    finally:bar.close()

def load_selective_head(path:Path,ncls:int,device:torch.device)->SelectiveFusionHead:
    h=SelectiveFusionHead(ncls).to(device);ck=torch.load(path,map_location='cpu',weights_only=False);h.load_state_dict(ck['state_dict'],strict=True);h.eval();return h

def videomae_v2s_gflops_per_view()->float:

    T=16;H=W=224;tube=2;patch=16;D=384;N=(T//tube)*(H//patch)*(W//patch);depth=12
    patch_flops=(T//tube)*(H//patch)*(W//patch)*D*(3*tube*patch*patch)
    block=12*N*D*D + 2*N*N*D
    return float((patch_flops+depth*block)/1e9)

def entropy_from_logits_np(z:np.ndarray)->np.ndarray:
    x=np.asarray(z,dtype=np.float64);x=x-x.max(-1,keepdims=True);p=np.exp(x);p/=p.sum(-1,keepdims=True);return -(p*np.log(p+1e-12)).sum(-1)

@torch.no_grad()
def evaluate_selective_baseline(model:EvidenceFoundation,selective:SelectiveFusionHead,loader:DataLoader,labels:Sequence[str],device:torch.device,amp:bool,policy:str,seed:int,max_batches:int=0)->pd.DataFrame:

    model.eval();selective.eval();rng=np.random.RandomState(seed);by_budget={k:{'y':[],'logits':[],'lat_ms':0.0,'n':0} for k in range(4)}
    fixed=[2,0,1]
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device);x=b['x'];y=b['y'];B=len(y)
        if device.type=='cuda':torch.cuda.synchronize()
        t0=time.time()
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):ov=model.overview(x)
        pools=torch.zeros(B,3,384,device=device);mask=torch.zeros(B,3,device=device);current=selective(ov['global'],pools,mask)
        if device.type=='cuda':torch.cuda.synchronize()
        base_ms=(time.time()-t0)*1000
        by_budget[0]['y']+=y.cpu().tolist();by_budget[0]['logits'].append(current.float().cpu().numpy());by_budget[0]['lat_ms']+=base_ms;by_budget[0]['n']+=B
        for budget in range(1,4):

            if policy=='FixedBodyFirst': choice=torch.tensor([next(v for v in fixed if mask[i,v]<.5) for i in range(B)],device=device)
            elif policy=='Random':
                ch=[]
                for i in range(B):
                    av=np.where(mask[i].cpu().numpy()<.5)[0];ch.append(int(rng.choice(av)))
                choice=torch.tensor(ch,device=device)
            elif policy=='UncertaintyFirst':
                unc=torch.sigmoid(ov['occ_logits']).float().masked_fill(mask>.5,-1e9);choice=unc.argmax(1)
            elif policy=='EntropyFirst':
                ent=torch.tensor(entropy_from_logits_np(ov['view_logits'].float().cpu().numpy()),device=device);ent=ent.masked_fill(mask>.5,-1e9);choice=ent.argmax(1)
            elif policy=='Oracle':

                gains=torch.full((B,3),-1e9,device=device);cur_lp=F.log_softmax(current,1)[torch.arange(B,device=device),y]
                for v in range(3):
                    ids=torch.where(mask[:,v]<.5)[0]
                    if len(ids)==0:continue
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=model.encode_view(x[ids,v],v)
                    pp=pools.clone();mm=mask.clone();pp[ids,v]=enc['pool'];mm[ids,v]=1;zz=selective(ov['global'],pp,mm);lp=F.log_softmax(zz,1)[torch.arange(B,device=device),y];gains[:,v]=torch.where(mm[:,v]>mask[:,v],lp-cur_lp,gains[:,v])
                choice=gains.argmax(1)
            else:raise ValueError(policy)
            for v in range(3):
                ids=torch.where(choice==v)[0]
                if len(ids):
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=model.encode_view(x[ids,v],v)
                    pools[ids,v]=enc['pool'];mask[ids,v]=1
            with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):current=selective(ov['global'],pools,mask)
            if device.type=='cuda':torch.cuda.synchronize()
            base_ms+=(time.time()-q0)*1000
            by_budget[budget]['y']+=y.cpu().tolist();by_budget[budget]['logits'].append(current.float().cpu().numpy());by_budget[budget]['lat_ms']+=base_ms;by_budget[budget]['n']+=B
    rows=[];gf=videomae_v2s_gflops_per_view()
    for budget,d in by_budget.items():
        yy=np.asarray(d['y']);zz=np.concatenate(d['logits']);pp=zz.argmax(1);prob=torch.from_numpy(zz).softmax(1).numpy();m=classification_metrics(yy,pp,prob,labels);rows.append({'policy':policy,'max_detailed_queries':budget,**m,'detail_gflops_approx':(float('nan') if policy=='Oracle' else budget*gf),'model_latency_ms_per_sample':d['lat_ms']/max(1,d['n']),'oracle_is_diagnostic':policy=='Oracle','deployable_cost_valid':policy!='Oracle'})
    return pd.DataFrame(rows)




class PrototypeFalsification(nn.Module):

    def __init__(self,ncls:int,qnum:int=4,max_revision:float=3.0):
        super().__init__();self.ncls=ncls;self.qnum=qnum;self.max_revision=float(max_revision)
        self.query_logits=nn.Parameter(torch.zeros(qnum));self.semantic_raw=nn.Parameter(torch.tensor(0.0));self.gate=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,1));self.alpha_raw=nn.Parameter(torch.tensor(0.0));self.temperature_raw=nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-1.0)
    def forward(self,base_logits:torch.Tensor,evidence:torch.Tensor,semantic_prob:torch.Tensor,proto_evidence:torch.Tensor,proto_semantic:torch.Tensor)->Dict[str,torch.Tensor]:

        en=F.normalize(evidence,dim=-1);pn=F.normalize(proto_evidence,dim=-1);cos=torch.einsum('bqd,cqd->bcq',en,pn);qw=torch.softmax(self.query_logits,0);df=((1-cos)*qw[None,None,:]).sum(-1)
        ds=(semantic_prob[:,None,:]-proto_semantic[None,:,:]).abs().mean(-1);sw=F.softplus(self.semantic_raw);dist=df+sw*ds
        hyp=base_logits.argmax(1);hd=dist.gather(1,hyp[:,None]).squeeze(1);prob=F.softmax(base_logits,1);ent=-(prob*torch.log(prob.clamp_min(1e-8))).sum(1)/math.log(max(2,self.ncls));top=base_logits.topk(min(2,self.ncls),1).values;margin=(top[:,0]-top[:,1]) if self.ncls>1 else torch.ones_like(hd);margin_conf=torch.sigmoid(margin)
        gate=torch.sigmoid(self.gate(torch.stack([hd,ent,1-margin_conf],1)).squeeze(1));temp=.25+F.softplus(self.temperature_raw);compat=-dist/temp;direction=compat-compat.mean(1,keepdim=True);alpha=self.max_revision*torch.tanh(self.alpha_raw);final=base_logits+alpha*gate[:,None]*direction
        return {'logits':final,'base_logits':base_logits,'distance':dist,'hypothesis':hyp,'contradiction':hd,'gate':gate,'alpha':alpha,'compatibility':compat}

def cache_has_all_classes(cache:Dict[str,np.ndarray],ncls:int)->bool:
    if 'y' not in cache:
        return False
    y=np.asarray(cache['y'],dtype=int)
    return set(range(int(ncls))).issubset(set(y.tolist()))

def build_prototypes(cache:Dict[str,np.ndarray],ncls:int)->Tuple[np.ndarray,np.ndarray]:
    y=np.asarray(cache['y'],dtype=int);ev=np.asarray(cache['agg_evidence'],dtype=np.float32);sem=np.asarray(cache['semantic'],dtype=np.float32);valid=np.asarray(cache['valid'],dtype=np.float32);pe=[];ps=[]
    for c in range(ncls):
        m=y==c
        if not m.any():raise RuntimeError(f'No train samples for prototype class {c}')
        pe.append(ev[m].mean(0));p=np.zeros(SEM_DIM,dtype=np.float32);mg=m&(valid[:,0]>0)
        p[0:2]=sem[mg,0:2].mean(0) if mg.any() else np.array([.5,.5],dtype=np.float32);mh=m&(valid[:,1]>0);p[2:6]=sem[mh,2:6].mean(0) if mh.any() else np.full(4,.25,dtype=np.float32);p[6:10]=sem[m,6:10].mean(0);ps.append(p)
    pe=np.stack(pe);ps=np.stack(ps);return pe.astype(np.float32),ps.astype(np.float32)

def apply_falsification_np(side:PrototypeFalsification,cache:Dict[str,np.ndarray],pe:np.ndarray,ps:np.ndarray,device:torch.device,batch:int=512)->Dict[str,np.ndarray]:
    side.eval();outs={k:[] for k in ('logits','base_logits','contradiction','gate')};N=len(cache['y'])
    with torch.no_grad():
        PEt=torch.tensor(pe,dtype=torch.float32,device=device);PSt=torch.tensor(ps,dtype=torch.float32,device=device)
        for st in range(0,N,batch):
            sl=slice(st,min(N,st+batch));b=torch.tensor(cache['base_logits'][sl],dtype=torch.float32,device=device);e=torch.tensor(cache['agg_evidence'][sl],dtype=torch.float32,device=device);s=torch.tensor(cache['semantic_pred'][sl],dtype=torch.float32,device=device);o=side(b,e,s,PEt,PSt)
            for k in outs:outs[k].append(o[k].float().cpu().numpy())
    return {k:np.concatenate(v) for k,v in outs.items()}

def metrics_from_logits(y:np.ndarray,logits:np.ndarray,labels:Sequence[str])->Dict[str,float]:
    z=np.asarray(logits);p=torch.from_numpy(z).softmax(1).numpy();return classification_metrics(np.asarray(y),z.argmax(1),p,labels)

def train_falsification_initial(cache_tr:Dict[str,np.ndarray],cache_va:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,out_dir:Path,args:argparse.Namespace,clock:GlobalClock,idx:int,exp_name:str)->Path:
    ensure_dir(out_dir);bp=out_dir/'best.pt';done=out_dir/'DONE.json'
    if bp.exists() and done.exists() and not args.overwrite:clock.complete(.01);return bp
    seed=reproduction_seed(exp_name,base=args.seed);seed_everything(seed);device=torch.device(args.device);side=PrototypeFalsification(len(labels),4,args.max_revision).to(device);opt=torch.optim.AdamW(side.parameters(),lr=args.falsification_lr,weight_decay=.01);cw=action_class_weights(pd.DataFrame({'_label':[labels[int(i)] for i in cache_tr['y']]}),labels).to(device);PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device)

    base_vm=metrics_from_logits(cache_va['y'],cache_va['base_logits'],labels);best_score=float(base_vm['macro_f1']);best_fs=float(base_vm['false_safe']);atomic_torch_save({'state_dict':side.state_dict(),'epoch':0,'best_val':base_vm,'selected':'baseline_identity','seed':seed,'proto_evidence':pe,'proto_semantic':ps},bp)
    N=len(cache_tr['y']);bs=512;epochs=1 if args.dry_run else args.falsification_epochs;hist=[];bar=ExperimentBar(clock,idx,exp_name,epochs,0)
    try:
        for ep in range(1,epochs+1):
            t0=time.time();side.train();order=torch.randperm(N);losses=[]
            for st in range(0,N,bs):
                ids=order[st:st+bs].numpy();y=torch.tensor(cache_tr['y'][ids],dtype=torch.long,device=device);b=torch.tensor(cache_tr['base_logits'][ids],dtype=torch.float32,device=device);e=torch.tensor(cache_tr['agg_evidence'][ids],dtype=torch.float32,device=device);sp=torch.tensor(cache_tr['semantic_pred'][ids],dtype=torch.float32,device=device);opt.zero_grad(set_to_none=True);o=side(b,e,sp,PE,PS);ce=F.cross_entropy(o['logits'],y,weight=cw)
                bc=b.argmax(1)==y;keep=torch.tensor(0.,device=device)
                if bc.any():keep=F.kl_div(F.log_softmax(o['logits'][bc],1),F.softmax(b[bc],1),reduction='batchmean')
                fs=torch.tensor(0.,device=device)
                if SAFE_LABEL in labels:
                    si=labels.index(SAFE_LABEL);non=y!=si
                    if non.any():true=o['logits'][torch.arange(len(y),device=device),y];fs=F.relu(o['logits'][:,si]-true+args.false_safe_margin)[non].mean()
                loss=ce+args.lambda_keep*keep+args.lambda_false_safe*fs+args.lambda_alpha*o['alpha'].pow(2);loss.backward();nn.utils.clip_grad_norm_(side.parameters(),5.);opt.step();losses.append(float(loss.detach()))
                if args.dry_run:break
            vr=apply_falsification_np(side,cache_va,pe,ps,device);vm=metrics_from_logits(cache_va['y'],vr['logits'],labels)
            eligible=(vm['macro_f1']>=base_vm['macro_f1']-args.val_tolerance and vm['acc']>=base_vm['acc']-args.val_tolerance and (not np.isfinite(base_vm['false_safe']) or vm['false_safe']<=base_vm['false_safe']+args.fs_tolerance))
            better=eligible and (vm['macro_f1']>best_score+1e-8 or (abs(vm['macro_f1']-best_score)<=1e-8 and vm['false_safe']<best_fs))
            if better:
                best_score=float(vm['macro_f1']);best_fs=float(vm['false_safe']);atomic_torch_save({'state_dict':side.state_dict(),'epoch':ep,'best_val':vm,'selected':'falsification','seed':seed,'proto_evidence':pe,'proto_semantic':ps},bp)
            hist.append({'epoch':ep,'loss':float(np.mean(losses)),'val_macro_f1':vm['macro_f1'],'val_acc':vm['acc'],'val_false_safe':vm['false_safe'],'eligible':bool(eligible),'selected_now':bool(better),'alpha':float(side.max_revision*torch.tanh(side.alpha_raw).detach().cpu()),'sec':time.time()-t0});pd.DataFrame(hist).to_csv(out_dir/'train_log.csv',index=False);bar.epoch(ep,hist[-1]['sec'],hist[-1]['loss'],vm['macro_f1'],best_score)
        ck=torch.load(bp,map_location='cpu',weights_only=False);write_json(done,{'completed':True,'selected_epoch':int(ck['epoch']),'selection':ck['selected'],'best_val':ck['best_val'],'baseline_val':base_vm,'seed':seed});return bp
    finally:bar.close()

def load_falsification(path:Path,ncls:int,device:torch.device,max_revision:float)->Tuple[PrototypeFalsification,np.ndarray,np.ndarray,Dict[str,Any]]:
    ck=torch.load(path,map_location='cpu',weights_only=False);s=PrototypeFalsification(ncls,4,max_revision).to(device);s.load_state_dict(ck['state_dict'],strict=True);s.eval();return s,np.asarray(ck['proto_evidence'],dtype=np.float32),np.asarray(ck['proto_semantic'],dtype=np.float32),ck

def extract_subset_cache(model:EvidenceFoundation,dfp:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],args:argparse.Namespace,active:Sequence[int],tag:str)->Dict[str,np.ndarray]:
    seed=stable_seed('subset',tag,base=args.seed);ds=DMDMultiViewDataset(dfp,labels,timelines,args.frames,args.size,False,seed,True,args.dry_per_class if args.dry_run else 0);ld=make_loader(ds,args.eval_batch,0 if args.dry_run else args.workers,False,seed);device=torch.device(args.device);model.eval();ys=[];bl=[];ev=[];sp=[];occ=[];valid=[]
    with torch.no_grad():
        for bi,b in enumerate(ld):
            if args.dry_run and bi>=1:break
            b=move_batch(b,device)
            with torch.autocast(device_type=device.type,enabled=args.amp and device.type=='cuda'):o=model(b['x'],active=active)
            sl=o['semantic_logits'];semprob=model.semantic_prob_from_evidence(o['evidence']) if sl is not None else torch.zeros(len(b['y']),SEM_DIM,device=device);ys.append(b['y'].cpu().numpy());bl.append(o['logits'].float().cpu().numpy());ev.append(o['evidence'].float().cpu().numpy());sp.append(semprob.float().cpu().numpy());occ.append(b['occlusion'].cpu().numpy());valid.append(b['valid'].cpu().numpy())
    return {'y':np.concatenate(ys),'base_logits':np.concatenate(bl),'agg_evidence':np.concatenate(ev),'semantic_pred':np.concatenate(sp),'occlusion':np.concatenate(occ),'valid':np.concatenate(valid)}

def evaluate_kofu_ood(side:PrototypeFalsification,pe:np.ndarray,ps:np.ndarray,known_cache:Dict[str,np.ndarray],unknown_cache:Dict[str,np.ndarray],val_known_cache:Dict[str,np.ndarray],labels:Sequence[str],device:torch.device,proto:str,fold:str,out_dir:Path)->List[Dict[str,Any]]:
    kr=apply_falsification_np(side,known_cache,pe,ps,device);ur=apply_falsification_np(side,unknown_cache,pe,ps,device);vr=apply_falsification_np(side,val_known_cache,pe,ps,device)
    ke=energy_np(kr['logits']);ue=energy_np(ur['logits']);ve=energy_np(vr['logits']);kc=kr['contradiction'];uc=ur['contradiction'];vc=vr['contradiction'];me,se=zfit(ve);mc,sc=zfit(vc);kf=zapply(ke,me,se)+zapply(kc,mc,sc);uf=zapply(ue,me,se)+zapply(uc,mc,sc)
    rows=[{'protocol':proto,'fold':fold,'method':'KOFU','ood_score':'Energy',**ood_metrics(ke,ue)},{'protocol':proto,'fold':fold,'method':'KOFU','ood_score':'FalsificationScore',**ood_metrics(kf,uf)}];ensure_dir(out_dir);pd.DataFrame(rows).to_csv(out_dir/'ood_metrics.csv',index=False);return rows

class SelectiveEvidenceAcquisitionDataset(Dataset):
    def __init__(self,cache:Dict[str,np.ndarray]):self.cache=cache;self.N=len(cache['y'])
    def __len__(self):return self.N
    def __getitem__(self,i):return i

def aggregate_selected_evidence_offline(model:EvidenceFoundation,ev_by_view:torch.Tensor,mask:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
    ev=model.aggregate_evidence_tensor(ev_by_view,mask);sp=model.semantic_prob_from_evidence(ev);return ev,sp

def train_selective_acquisition_policy_auxiliary(foundation:EvidenceFoundation,selective:SelectiveFusionHead,side:PrototypeFalsification,pe:np.ndarray,ps:np.ndarray,cache_tr:Dict[str,np.ndarray],out_dir:Path,args:argparse.Namespace,clock:GlobalClock,idx:int,exp_name:str)->Path:
    ensure_dir(out_dir);bp=out_dir/'best.pt';done=out_dir/'DONE.json'
    if bp.exists() and done.exists() and not args.overwrite:clock.complete(.01);return bp
    seed=stable_seed(exp_name,base=args.seed);seed_everything(seed);device=torch.device(args.device);pol=SelectiveEvidenceAcquisitionPolicy(len(pe)).to(device);opt=torch.optim.AdamW(pol.parameters(),lr=1e-3,weight_decay=.01);PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device);N=len(cache_tr['y']);epochs=1 if args.dry_run else args.acquisition_epochs;best=1e30;hist=[];bar=ExperimentBar(clock,idx,exp_name,epochs,0)

    O=torch.tensor(cache_tr['overview'],dtype=torch.float32);OV=torch.tensor(cache_tr['overview_views'],dtype=torch.float32);P=torch.tensor(cache_tr['pools'],dtype=torch.float32);EV=torch.tensor(cache_tr['evidence_by_view'],dtype=torch.float32);Y=torch.tensor(cache_tr['y'],dtype=torch.long);masks=all_subset_masks(True);bs=256
    try:
        for ep in range(1,epochs+1):
            t0=time.time();pol.train();order=torch.randperm(N);losses=[]
            for st in range(0,N,bs):
                ids=order[st:st+bs];o=O[ids].to(device);ov=OV[ids].to(device);p=P[ids].to(device);evv=EV[ids].to(device);y=Y[ids].to(device);rng=np.random.RandomState(seed+ep*1237+st);mi=rng.randint(0,len(masks)-1,size=len(ids));mask=torch.tensor(masks[mi],dtype=torch.float32,device=device);base=selective(o,p,mask);eva,sp=aggregate_selected_evidence_offline(foundation,evv,mask);side_cur=side(base,eva,sp,PE,PS)['logits'];cur=torch.where((mask.sum(1,keepdim=True)>0),side_cur,base);cur_lp=F.log_softmax(cur,1)[torch.arange(len(y),device=device),y];targets=torch.zeros(len(y),3,device=device);valid=1-mask
                with torch.no_grad():
                    for v in range(3):
                        m2=mask.clone();m2[:,v]=1;base2=selective(o,p,m2);e2,s2=aggregate_selected_evidence_offline(foundation,evv,m2);z2=side(base2,e2,s2,PE,PS)['logits'];lp=F.log_softmax(z2,1)[torch.arange(len(y),device=device),y];targets[:,v]=lp-cur_lp
                pred=pol(o,ov,cur.detach(),mask);loss=((pred-targets).pow(2)*valid).sum()/valid.sum().clamp_min(1.);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
                if args.dry_run:break
            vl=float(np.mean(losses));hist.append({'epoch':ep,'mse':vl,'sec':time.time()-t0})
            if vl<best:best=vl;atomic_torch_save({'state_dict':pol.state_dict(),'epoch':ep,'mse':best,'seed':seed},bp)
            pd.DataFrame(hist).to_csv(out_dir/'train_log.csv',index=False);bar.epoch(ep,hist[-1]['sec'],vl,-vl,-best)
        write_json(done,{'completed':True,'best_mse':best,'seed':seed});return bp
    finally:bar.close()

def load_selective_acquisition_policy(path:Path,ncls:int,device:torch.device)->SelectiveEvidenceAcquisitionPolicy:
    p=SelectiveEvidenceAcquisitionPolicy(ncls).to(device);p.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True);p.eval();return p

@torch.no_grad()
def evaluate_selective_acquisition(foundation:EvidenceFoundation,selective:SelectiveFusionHead,side:PrototypeFalsification,acquisition_policy:SelectiveEvidenceAcquisitionPolicy,pe:np.ndarray,ps:np.ndarray,loader:DataLoader,labels:Sequence[str],device:torch.device,amp:bool,policy:str,seed:int,max_batches:int=0)->pd.DataFrame:
    foundation.eval();selective.eval();side.eval();acquisition_policy.eval();PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device);rng=np.random.RandomState(seed);fixed=[2,0,1];by={k:{'y':[],'logits':[],'lat':0.,'n':0} for k in range(4)}
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device);x=b['x'];y=b['y'];B=len(y)
        if device.type=='cuda':torch.cuda.synchronize()
        t0=time.time()
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):ov=foundation.overview(x)
        pools=torch.zeros(B,3,384,device=device,dtype=torch.float32);evv=torch.zeros(B,3,4,384,device=device,dtype=torch.float32);mask=torch.zeros(B,3,device=device,dtype=torch.float32);base=selective(ov['global'].float(),pools,mask);cur=base
        if device.type=='cuda':torch.cuda.synchronize()
        lat=(time.time()-t0)*1000;by[0]['y']+=y.cpu().tolist();by[0]['logits'].append(cur.float().cpu().numpy());by[0]['lat']+=lat;by[0]['n']+=B
        for budget in range(1,4):
            if device.type=='cuda':torch.cuda.synchronize()
            q0=time.time()
            if policy=='KOFU_LearnedUtility':scores=acquisition_policy(ov['global'].float(),ov['view_features'].float(),cur.float(),mask.float()).float().masked_fill(mask.float()>.5,-1e9);choice=scores.argmax(1)
            elif policy=='UncertaintyFirst':
                pv=torch.softmax(ov['view_logits'],dim=-1);choice=(1-pv.max(-1).values).float().masked_fill(mask.float()>.5,-1e9).argmax(1)
            elif policy=='EntropyFirst':
                pv=torch.softmax(ov['view_logits'],dim=-1);choice=(-(pv.float()*torch.log(pv.float().clamp_min(1e-8))).sum(-1)).masked_fill(mask.float()>.5,-1e9).argmax(1)
            elif policy=='FixedBodyFirst':choice=torch.tensor([next(v for v in fixed if mask[i,v]<.5) for i in range(B)],device=device)
            elif policy=='Random':
                ch=[]
                for i in range(B):ch.append(int(rng.choice(np.where(mask[i].cpu().numpy()<.5)[0])))
                choice=torch.tensor(ch,device=device)
            elif policy=='Oracle':
                gains=torch.full((B,3),-1e9,device=device);cur_lp=F.log_softmax(cur,1)[torch.arange(B,device=device),y]
                for v in range(3):
                    ids=torch.where(mask[:,v]<.5)[0]
                    if len(ids)==0:continue
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=foundation.encode_view(x[ids,v],v)
                    pp=pools.clone();ee=evv.clone();mm=mask.clone();pp[ids,v]=enc['pool'].float();ee[ids,v]=enc['evidence'].float();mm[ids,v]=1;b2=selective(ov['global'].float(),pp,mm);ea=foundation.aggregate_evidence_tensor(ee,mm);sp=foundation.semantic_prob_from_evidence(ea);z2=side(b2.float(),ea.float(),sp.float(),PE,PS)['logits'];lp=F.log_softmax(z2,1)[torch.arange(B,device=device),y];gains[:,v]=torch.where(mm[:,v]>mask[:,v],lp-cur_lp,gains[:,v])
                choice=gains.argmax(1)
            else:raise ValueError(policy)
            for v in range(3):
                ids=torch.where(choice==v)[0]
                if len(ids):
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=foundation.encode_view(x[ids,v],v)
                    pools[ids,v]=enc['pool'].float();evv[ids,v]=enc['evidence'].float();mask[ids,v]=1
            with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                base=selective(ov['global'].float(),pools,mask);ea=foundation.aggregate_evidence_tensor(evv,mask);sp=foundation.semantic_prob_from_evidence(ea);cur=side(base.float(),ea.float(),sp.float(),PE,PS)['logits']
            if device.type=='cuda':torch.cuda.synchronize()
            lat+=(time.time()-q0)*1000;by[budget]['y']+=y.cpu().tolist();by[budget]['logits'].append(cur.float().cpu().numpy());by[budget]['lat']+=lat;by[budget]['n']+=B
    rows=[];gf=videomae_v2s_gflops_per_view()
    for budget,d in by.items():
        yy=np.asarray(d['y']);zz=np.concatenate(d['logits']);pp=zz.argmax(1);prob=torch.from_numpy(zz).softmax(1).numpy();m=classification_metrics(yy,pp,prob,labels);rows.append({'policy':policy,'max_detailed_queries':budget,**m,'detail_gflops_approx':(float('nan') if policy=='Oracle' else budget*gf),'model_latency_ms_per_sample':(float('nan') if policy=='Oracle' else d['lat']/max(1,d['n'])),'oracle_is_diagnostic':policy=='Oracle','deployable_cost_valid':policy!='Oracle'})
    return pd.DataFrame(rows)

def train_foundation_and_falsification(df:pd.DataFrame,timelines:Dict[str,EvidenceTimeline],fold:str,labels:Sequence[str],proto_name:str,vmck:Path,args:argparse.Namespace,out:Path,clock:GlobalClock,idx:int)->Tuple[EvidenceFoundation,PrototypeFalsification,np.ndarray,np.ndarray,Dict[str,np.ndarray],Dict[str,np.ndarray],Path,int]:
    tr=filter_df(df,fold,'train',labels);va=filter_df(df,fold,'val',labels);exp=(f'A/{fold}/foundation' if proto_name=='protocol_a' else (f"{proto_name.split('/',1)[1]}/{fold}/foundation" if proto_name.startswith('protocol_c/') else f'{proto_name}/{fold}/foundation'));seed_everything(stable_seed(exp,base=args.seed));fm=EvidenceFoundation(len(labels),vmck,args.frames);fb=train_classifier(fm,tr,va,labels,timelines,out/proto_name/'foundation'/fold,args,clock,idx,exp,True);idx+=1;load_model_state(fm,fb,torch.device(args.device));cache_dir=out/proto_name/'foundation'/fold/'feature_cache';ensure_dir(cache_dir);tp=cache_dir/'train.npz';vp=cache_dir/'val.npz'
    use_cache=False
    if tp.exists() and vp.exists() and not args.overwrite:
        ctr={k:v for k,v in np.load(tp).items()};cva={k:v for k,v in np.load(vp).items()}
        use_cache=cache_has_all_classes(ctr,len(labels)) and cache_has_all_classes(cva,len(labels))
        if not use_cache:
            log(f'Ignoring incomplete feature cache for {proto_name}/{fold}; rebuilding balanced cache.')
    if not use_cache:
        ctr=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(tr,labels,timelines,args.frames,args.size,False,stable_seed(exp,'tr',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,stable_seed(exp,'tr',base=args.seed)),torch.device(args.device),args.amp,max_batches=0);cva=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(va,labels,timelines,args.frames,args.size,False,stable_seed(exp,'va',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,stable_seed(exp,'va',base=args.seed)),torch.device(args.device),args.amp,max_batches=0);np.savez_compressed(tp,**ctr);np.savez_compressed(vp,**cva)
    pe,ps=build_prototypes(ctr,len(labels));sp=train_falsification_initial(ctr,cva,labels,pe,ps,out/proto_name/'kofu'/fold,args,clock,idx,f'{proto_name}/{fold}/Falsification');idx+=1;side,pe,ps,sck=load_falsification(sp,len(labels),torch.device(args.device),args.max_revision);return fm,side,pe,ps,ctr,cva,sp,idx






def _load_npz_dict(path:Path)->Dict[str,np.ndarray]:
    if not path.exists(): raise FileNotFoundError(path)
    with np.load(path) as z: return {k:z[k] for k in z.files}

def _foundation_checkpoint_dir(foundation_root:Path,proto_name:str,fold:str)->Path:
    if proto_name=='protocol_a': return foundation_root/'protocol_a'/'foundation'/fold
    if proto_name.startswith('protocol_c/'):
        return foundation_root/'protocol_c'/proto_name.split('/',1)[1]/'foundation'/fold
    raise ValueError(proto_name)

def load_frozen_foundation(df:pd.DataFrame,timelines:Dict[str,EvidenceTimeline],fold:str,labels:Sequence[str],proto_name:str,vmck:Path,args:argparse.Namespace,foundation_root:Path)->Tuple[EvidenceFoundation,Dict[str,np.ndarray],Dict[str,np.ndarray],Path]:
    fd=_foundation_checkpoint_dir(foundation_root,proto_name,fold); bp=fd/'best.pt'
    if not bp.exists():
        raise FileNotFoundError(f'Missing frozen Foundation checkpoint: {bp}. Falsification training does not retrain the Foundation.')
    device=torch.device(args.device); fm=EvidenceFoundation(len(labels),vmck,args.frames); ck=load_model_state(fm,bp,device)
    ck_labels=list(ck.get('labels',labels))
    if ck_labels!=list(labels): raise RuntimeError(f'{proto_name}/{fold}: Foundation labels differ. checkpoint={ck_labels}, expected={list(labels)}')
    for p in fm.parameters(): p.requires_grad_(False)
    fm.eval(); cd=fd/'feature_cache'; tp=cd/'train.npz'; vp=cd/'val.npz'
    if tp.exists() and vp.exists():
        ctr=_load_npz_dict(tp); cva=_load_npz_dict(vp)
    else:
        log(f'{proto_name}/{fold}: frozen feature cache missing; rebuilding it from the frozen checkpoint (no training).')
        tr=filter_df(df,fold,'train',labels); va=filter_df(df,fold,'val',labels)
        exp=f'FoundationReuse/{proto_name}/{fold}';
        ctr=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(tr,labels,timelines,args.frames,args.size,False,reproduction_seed(exp+'/tr',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,reproduction_seed(exp+'/tr',base=args.seed)),device,args.amp,max_batches=0)
        cva=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(va,labels,timelines,args.frames,args.size,False,reproduction_seed(exp+'/va',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,reproduction_seed(exp+'/va',base=args.seed)),device,args.amp,max_batches=0)
    validate_feature_cache(ctr,len(labels),f'{proto_name}/{fold}/train-cache')
    validate_feature_cache(cva,len(labels),f'{proto_name}/{fold}/val-cache')
    if not cache_has_all_classes(ctr,len(labels)) or not cache_has_all_classes(cva,len(labels)):
        raise RuntimeError(f'{proto_name}/{fold}: Foundation train/val cache does not cover every known class')
    return fm,ctr,cva,bp

@torch.no_grad()
def offline_partial_foundation(model:EvidenceFoundation,cache:Dict[str,np.ndarray],ids:np.ndarray,mask_np:Sequence[int],full_overview:bool,device:torch.device)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:





    mask=torch.tensor(np.asarray(mask_np,dtype=np.float32),device=device).view(1,3).expand(len(ids),-1)
    if full_overview:
        og=torch.tensor(cache['overview'][ids],dtype=torch.float32,device=device)
    else:
        vf=torch.tensor(cache['overview_views'][ids],dtype=torch.float32,device=device)*mask[:,:,None]
        og=model.overview.fuse(vf.flatten(1))
    active=[v for v in range(3) if float(mask_np[v])>.5]
    if not active:
        z=model.overview.head(og); ev=torch.zeros(len(ids),model.qnum,model.d,device=device,dtype=torch.float32)
        return z.float(),ev,model.semantic_prob_from_evidence(ev).float()
    pools=torch.tensor(cache['pools'][ids][:,active],dtype=torch.float32,device=device)
    unc=torch.tensor(cache['unc'][ids][:,active],dtype=torch.float32,device=device)
    occ=torch.tensor(cache['occ_logits'][ids][:,active],dtype=torch.float32,device=device)
    rel=-unc-F.softplus(occ); w=torch.softmax(rel,1); fused=(pools*w[:,:,None]).sum(1)
    evv=torch.tensor(cache['evidence_by_view'][ids][:,active],dtype=torch.float32,device=device)
    score=model.ev_view_score(evv).squeeze(-1).permute(0,2,1).float(); ew=torch.softmax(score,2); ev=(evv.permute(0,2,1,3)*ew[:,:,:,None]).sum(2)
    basefeat=model.fuse(torch.cat([fused,ev.mean(1)],1)); ov=model.overview_to_d(og); ff=model.final_fuse(torch.cat([basefeat,ov],1)); z=model.final_head(ff)
    sp=model.semantic_prob_from_evidence(ev)
    return z.float(),ev.float(),sp.float()

def _nonfull_masks()->List[np.ndarray]:
    return [np.asarray(m,dtype=np.float32) for m in VIEW_MASKS if int(np.asarray(m).sum())<3]

def build_robust_condition_cache(model:EvidenceFoundation,cache:Dict[str,np.ndarray],device:torch.device,batch:int=512)->Dict[str,Dict[str,np.ndarray]]:



    masks=_nonfull_masks(); out={}; N=len(cache['y'])
    for mode in ('D','E'):
        for mi,m in enumerate(masks):
            zz=[];ee=[];ss=[]
            for st in range(0,N,batch):
                ids=np.arange(st,min(N,st+batch));z,e,s=offline_partial_foundation(model,cache,ids,m,mode=='E',device);zz.append(z.cpu().numpy());ee.append(e.cpu().numpy());ss.append(s.cpu().numpy())
            out[f'{mode}_{mi}']={'y':np.asarray(cache['y']),'base_logits':np.concatenate(zz),'agg_evidence':np.concatenate(ee),'semantic_pred':np.concatenate(ss)}
    return out

def _false_safe_loss(logits:torch.Tensor,y:torch.Tensor,labels:Sequence[str],margin:float)->torch.Tensor:
    if SAFE_LABEL not in labels:return logits.sum()*0
    si=labels.index(SAFE_LABEL); non=y!=si
    if not non.any():return logits.sum()*0
    true=logits[torch.arange(len(y),device=y.device),y]
    return F.relu(logits[:,si]-true+margin)[non].mean()

def _robust_validation(side:PrototypeFalsification,clean:Dict[str,np.ndarray],hard:Dict[str,Dict[str,np.ndarray]],pe:np.ndarray,ps:np.ndarray,labels:Sequence[str],device:torch.device)->Dict[str,Any]:
    cr=apply_falsification_np(side,clean,pe,ps,device); cm=metrics_from_logits(clean['y'],cr['logits'],labels); vals=[]
    for name,c in hard.items():
        r=apply_falsification_np(side,c,pe,ps,device);m=metrics_from_logits(c['y'],r['logits'],labels);vals.append(float(m['macro_f1']))
    return {'full':cm,'robust_mean_macro_f1':float(np.mean(vals)),'hard_macro_f1':vals}

def train_falsification(model:EvidenceFoundation,cache_tr:Dict[str,np.ndarray],cache_va:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,out_dir:Path,args:argparse.Namespace,clock:GlobalClock,idx:int,exp_name:str)->Path:
    ensure_dir(out_dir);bp=out_dir/'best.pt';done=out_dir/'DONE.json'
    if bp.exists() and done.exists() and not args.overwrite:clock.complete(.01);return bp
    seed=reproduction_seed(exp_name,base=args.seed);seed_everything(seed);device=torch.device(args.device);side=PrototypeFalsification(len(labels),4,args.max_revision).to(device)
    opt=torch.optim.AdamW(side.parameters(),lr=args.falsification_lr,weight_decay=.01);cw=action_class_weights(pd.DataFrame({'_label':[labels[int(i)] for i in cache_tr['y']]}),labels).to(device);PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device)
    log(f'{exp_name}: building train/val hard banks from frozen per-view features (D missing-view + E selective-detail).')
    hard_tr=build_robust_condition_cache(model,cache_tr,device); hard_va=build_robust_condition_cache(model,cache_va,device)
    base_full=metrics_from_logits(cache_va['y'],cache_va['base_logits'],labels)

    base_hard=[metrics_from_logits(c['y'],c['base_logits'],labels)['macro_f1'] for c in hard_va.values()];best_rob=float(np.mean(base_hard));best_full=float(base_full['macro_f1']);best_fs=float(base_full['false_safe'])
    atomic_torch_save({'state_dict':side.state_dict(),'epoch':0,'best_val':base_full,'robust_mean_macro_f1':best_rob,'selected':'baseline_identity','seed':seed,'proto_evidence':pe,'proto_semantic':ps},bp)
    keys=sorted(hard_tr);N=len(cache_tr['y']);bs=512;epochs=1 if args.dry_run else args.falsification_epochs;hist=[];bar=ExperimentBar(clock,idx,exp_name,epochs,0)
    try:
        for ep in range(1,epochs+1):
            t0=time.time();side.train();order=torch.randperm(N);losses=[]
            for bi,st in enumerate(range(0,N,bs)):
                ids=order[st:st+bs].numpy();y=torch.tensor(cache_tr['y'][ids],dtype=torch.long,device=device)
                bc=torch.tensor(cache_tr['base_logits'][ids],dtype=torch.float32,device=device);ec=torch.tensor(cache_tr['agg_evidence'][ids],dtype=torch.float32,device=device);sc=torch.tensor(cache_tr['semantic_pred'][ids],dtype=torch.float32,device=device)
                hk=keys[(bi+ep-1)%len(keys)];hc=hard_tr[hk];bh=torch.tensor(hc['base_logits'][ids],dtype=torch.float32,device=device);eh=torch.tensor(hc['agg_evidence'][ids],dtype=torch.float32,device=device);sh=torch.tensor(hc['semantic_pred'][ids],dtype=torch.float32,device=device)
                opt.zero_grad(set_to_none=True);oc=side(bc,ec,sc,PE,PS);oh=side(bh,eh,sh,PE,PS)
                ce_clean=F.cross_entropy(oc['logits'],y,weight=cw);ce_hard=F.cross_entropy(oh['logits'],y,weight=cw)
                good=bc.argmax(1)==y;keep=F.kl_div(F.log_softmax(oc['logits'][good],1),F.softmax(bc[good],1),reduction='batchmean') if good.any() else ce_clean*0
                fs=.5*(_false_safe_loss(oc['logits'],y,labels,args.false_safe_margin)+_false_safe_loss(oh['logits'],y,labels,args.false_safe_margin))
                loss=ce_clean+args.lambda_hard*ce_hard+args.lambda_keep*keep+args.lambda_false_safe*fs+args.lambda_alpha*oc['alpha'].pow(2)
                loss.backward();nn.utils.clip_grad_norm_(side.parameters(),5.);opt.step();losses.append(float(loss.detach()))
                if args.dry_run:break
            side.eval();vr=_robust_validation(side,cache_va,hard_va,pe,ps,labels,device);vm=vr['full'];rob=float(vr['robust_mean_macro_f1'])
            eligible=(vm['macro_f1']>=base_full['macro_f1']-args.val_tolerance and vm['acc']>=base_full['acc']-args.val_tolerance and (not np.isfinite(base_full['false_safe']) or vm['false_safe']<=base_full['false_safe']+args.fs_tolerance))
            better=eligible and (rob>best_rob+1e-6 or (abs(rob-best_rob)<=1e-6 and vm['macro_f1']>best_full+1e-8))
            if better:
                best_rob=rob;best_full=float(vm['macro_f1']);best_fs=float(vm['false_safe']);atomic_torch_save({'state_dict':side.state_dict(),'epoch':ep,'best_val':vm,'robust_mean_macro_f1':rob,'selected':'falsification','seed':seed,'proto_evidence':pe,'proto_semantic':ps},bp)
            row={'epoch':ep,'loss':float(np.mean(losses)),'val_macro_f1':vm['macro_f1'],'val_acc':vm['acc'],'val_false_safe':vm['false_safe'],'val_robust_mean_macro_f1':rob,'eligible':bool(eligible),'selected_now':bool(better),'alpha':float(side.max_revision*torch.tanh(side.alpha_raw).detach().cpu()),'sec':time.time()-t0};hist.append(row);pd.DataFrame(hist).to_csv(out_dir/'train_log.csv',index=False);bar.epoch(ep,row['sec'],row['loss'],rob,best_rob)
        ck=torch.load(bp,map_location='cpu',weights_only=False);write_json(done,{'completed':True,'selected_epoch':int(ck['epoch']),'selection':ck['selected'],'best_val':ck['best_val'],'best_robust_mean_macro_f1':ck['robust_mean_macro_f1'],'baseline_full_val':base_full,'baseline_robust_mean_macro_f1':float(np.mean(base_hard)),'seed':seed});return bp
    finally:bar.close()

@torch.no_grad()
def partial_live_forward(model:EvidenceFoundation,ov:Dict[str,torch.Tensor],pools:torch.Tensor,evv:torch.Tensor,unc:torch.Tensor,occ:torch.Tensor,mask:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:

    m=mask.float(); rel=(-unc.float()-F.softplus(occ.float())).masked_fill(m<.5,-1e9);w=torch.softmax(rel,1);zero=m.sum(1)==0;w[zero]=0;fused=(pools.float()*w[:,:,None]).sum(1)
    ev=model.aggregate_evidence_tensor(evv.float(),m);basefeat=model.fuse(torch.cat([fused,ev.mean(1)],1));od=model.overview_to_d(ov['global'].float());ff=model.final_fuse(torch.cat([basefeat,od],1));z=model.final_head(ff)

    if zero.any(): z=z.clone(); z[zero]=model.overview.head(ov['global'].float())[zero]
    sp=model.semantic_prob_from_evidence(ev);return z.float(),ev.float(),sp.float()

def exact_partial_from_cache(model:EvidenceFoundation,cache:Dict[str,np.ndarray],ids:torch.Tensor,mask:torch.Tensor,device:torch.device)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    ii=ids.detach().cpu().numpy();B=len(ii);p=torch.tensor(cache['pools'][ii],dtype=torch.float32,device=device);e=torch.tensor(cache['evidence_by_view'][ii],dtype=torch.float32,device=device);u=torch.tensor(cache['unc'][ii],dtype=torch.float32,device=device);o=torch.tensor(cache['occ_logits'][ii],dtype=torch.float32,device=device);og=torch.tensor(cache['overview'][ii],dtype=torch.float32,device=device);ov={'global':og};return partial_live_forward(model,ov,p,e,u,o,mask)

def train_selective_acquisition_policy(foundation:EvidenceFoundation,side:PrototypeFalsification,pe:np.ndarray,ps:np.ndarray,cache_tr:Dict[str,np.ndarray],cache_va:Dict[str,np.ndarray],out_dir:Path,args:argparse.Namespace,clock:GlobalClock,idx:int,exp_name:str)->Path:
    ensure_dir(out_dir);bp=out_dir/'best.pt';done=out_dir/'DONE.json'
    if bp.exists() and done.exists() and not args.overwrite:clock.complete(.01);return bp
    seed=reproduction_seed(exp_name,base=args.seed);seed_everything(seed);device=torch.device(args.device);pol=SelectiveEvidenceAcquisitionPolicy(len(pe)).to(device);opt=torch.optim.AdamW(pol.parameters(),lr=1e-3,weight_decay=.01);PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device);masks=np.asarray([m for m in all_subset_masks(True) if np.sum(m)<3],dtype=np.float32);epochs=1 if args.dry_run else args.acquisition_epochs;best=1e30;hist=[];bar=ExperimentBar(clock,idx,exp_name,epochs,0)
    def batch_loss(cache,ids,mask_np,train:bool):
        y=torch.tensor(cache['y'][ids.cpu().numpy()],dtype=torch.long,device=device);mask=torch.tensor(mask_np,dtype=torch.float32,device=device);z,e,s=exact_partial_from_cache(foundation,cache,ids,mask,device);cur=torch.where((mask.sum(1,keepdim=True)>0),side(z,e,s,PE,PS)['logits'],z);cur_lp=F.log_softmax(cur,1)[torch.arange(len(y),device=device),y];targets=torch.full((len(y),3),-1e9,device=device)
        with torch.no_grad():
            for v in range(3):
                valid=mask[:,v]<.5
                if valid.any():
                    m2=mask.clone();m2[:,v]=1;z2,e2,s2=exact_partial_from_cache(foundation,cache,ids,m2,device);q=side(z2,e2,s2,PE,PS)['logits'];lp=F.log_softmax(q,1)[torch.arange(len(y),device=device),y];targets[:,v]=lp-cur_lp
        og=torch.tensor(cache['overview'][ids.cpu().numpy()],dtype=torch.float32,device=device);ov=torch.tensor(cache['overview_views'][ids.cpu().numpy()],dtype=torch.float32,device=device);pred=pol(og,ov,cur.detach(),mask);valid=mask<.5;return ((pred-targets).pow(2)*valid).sum()/valid.sum().clamp_min(1.)
    N=len(cache_tr['y']);Nv=len(cache_va['y']);bs=256
    try:
        for ep in range(1,epochs+1):
            t0=time.time();pol.train();order=torch.randperm(N);ls=[]
            for bi,st in enumerate(range(0,N,bs)):
                ids=order[st:st+bs];rng=np.random.RandomState(seed+ep*997+bi);mi=rng.randint(0,len(masks),size=len(ids));mn=masks[mi];loss=batch_loss(cache_tr,ids,mn,True);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(float(loss.detach()));
                if args.dry_run:break
            pol.eval();vls=[]
            with torch.no_grad():
                for bi,st in enumerate(range(0,Nv,bs)):
                    ids=torch.arange(st,min(Nv,st+bs));rng=np.random.RandomState(seed+77777+bi);mi=rng.randint(0,len(masks),size=len(ids));vls.append(float(batch_loss(cache_va,ids,masks[mi],False).detach()));
                    if args.dry_run:break
            vl=float(np.mean(vls));row={'epoch':ep,'train_mse':float(np.mean(ls)),'val_mse':vl,'sec':time.time()-t0};hist.append(row)
            if vl<best:best=vl;atomic_torch_save({'state_dict':pol.state_dict(),'epoch':ep,'val_mse':best,'seed':seed},bp)
            pd.DataFrame(hist).to_csv(out_dir/'train_log.csv',index=False);bar.epoch(ep,row['sec'],row['train_mse'],-vl,-best)
        write_json(done,{'completed':True,'best_val_mse':best,'seed':seed});return bp
    finally:bar.close()

@torch.no_grad()
def evaluate_selective_acquisition_exact(foundation:EvidenceFoundation,side:PrototypeFalsification,acquisition_policy:SelectiveEvidenceAcquisitionPolicy,pe:np.ndarray,ps:np.ndarray,loader:DataLoader,labels:Sequence[str],device:torch.device,amp:bool,policy:str,seed:int,max_batches:int=0)->pd.DataFrame:
    foundation.eval();side.eval();acquisition_policy.eval();PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device);rng=np.random.RandomState(seed);fixed=[2,0,1];by={k:{'y':[],'logits':[],'lat':0.,'n':0} for k in range(4)};checked=False
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device);x=b['x'];y=b['y'];B=len(y)
        if device.type=='cuda':torch.cuda.synchronize()
        t0=time.time()
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):ov=foundation.overview(x)
        pools=torch.zeros(B,3,384,device=device);evv=torch.zeros(B,3,4,384,device=device);unc=torch.zeros(B,3,device=device);occ=torch.zeros(B,3,device=device);mask=torch.zeros(B,3,device=device);cur=ov['logits'].float()
        if device.type=='cuda':torch.cuda.synchronize()
        lat=(time.time()-t0)*1000;by[0]['y']+=y.cpu().tolist();by[0]['logits'].append(cur.cpu().numpy());by[0]['lat']+=lat;by[0]['n']+=B
        for budget in range(1,4):
            if device.type=='cuda':torch.cuda.synchronize()
            q0=time.time()
            if policy=='KOFU_LearnedUtility':scores=acquisition_policy(ov['global'].float(),ov['view_features'].float(),cur.float(),mask).float().masked_fill(mask>.5,-1e9);choice=scores.argmax(1)
            elif policy=='UncertaintyFirst':
                pv=torch.softmax(ov['view_logits'].float(),-1);choice=(1-pv.max(-1).values).masked_fill(mask>.5,-1e9).argmax(1)
            elif policy=='EntropyFirst':
                pv=torch.softmax(ov['view_logits'].float(),-1);choice=(-(pv*torch.log(pv.clamp_min(1e-8))).sum(-1)).masked_fill(mask>.5,-1e9).argmax(1)
            elif policy=='FixedBodyFirst':choice=torch.tensor([next(v for v in fixed if mask[i,v]<.5) for i in range(B)],device=device)
            elif policy=='Random':choice=torch.tensor([int(rng.choice(np.where(mask[i].cpu().numpy()<.5)[0])) for i in range(B)],device=device)
            elif policy=='Oracle':
                gains=torch.full((B,3),-1e9,device=device);cur_lp=F.log_softmax(cur,1)[torch.arange(B,device=device),y]
                for v in range(3):
                    ids=torch.where(mask[:,v]<.5)[0]
                    if not len(ids):continue
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=foundation.encode_view(x[ids,v],v)
                    pp=pools.clone();ee=evv.clone();uu=unc.clone();oo=occ.clone();mm=mask.clone();pp[ids,v]=enc['pool'].float();ee[ids,v]=enc['evidence'].float();uu[ids,v]=enc['unc'].float();oo[ids,v]=enc['occ_logit'].float();mm[ids,v]=1;z2,e2,s2=partial_live_forward(foundation,ov,pp,ee,uu,oo,mm);q=side(z2,e2,s2,PE,PS)['logits'];lp=F.log_softmax(q,1)[torch.arange(B,device=device),y];gains[:,v]=torch.where(mm[:,v]>mask[:,v],lp-cur_lp,gains[:,v])
                choice=gains.argmax(1)
            else:raise ValueError(policy)
            for v in range(3):
                ids=torch.where(choice==v)[0]
                if len(ids):
                    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):enc=foundation.encode_view(x[ids,v],v)
                    pools[ids,v]=enc['pool'].float();evv[ids,v]=enc['evidence'].float();unc[ids,v]=enc['unc'].float();occ[ids,v]=enc['occ_logit'].float();mask[ids,v]=1
            z,e,s=partial_live_forward(foundation,ov,pools,evv,unc,occ,mask);cur=side(z,e,s,PE,PS)['logits']
            if budget==3 and not checked:

                encd={v:{'pool':pools[:,v],'evidence':evv[:,v],'unc':unc[:,v],'occ_logit':occ[:,v]} for v in range(3)};direct=foundation.aggregate(encd,ov);ds=foundation.semantic_prob_from_evidence(direct['evidence']);dz=side(direct['logits'].float(),direct['evidence'].float(),ds.float(),PE,PS)['logits'];err=float((dz.float()-cur.float()).abs().max().cpu())
                if err>2e-4:raise RuntimeError(f'Protocol E invariant failed: 3-query != full inference, max_abs_diff={err:.6g}')
                checked=True
            if device.type=='cuda':torch.cuda.synchronize()
            lat+=(time.time()-q0)*1000;by[budget]['y']+=y.cpu().tolist();by[budget]['logits'].append(cur.float().cpu().numpy());by[budget]['lat']+=lat;by[budget]['n']+=B
    rows=[];gf=videomae_v2s_gflops_per_view()
    for budget,d in by.items():
        yy=np.asarray(d['y']);zz=np.concatenate(d['logits']);pp=zz.argmax(1);prob=torch.from_numpy(zz).softmax(1).numpy();m=classification_metrics(yy,pp,prob,labels);rows.append({'policy':policy,'max_detailed_queries':budget,**m,'detail_gflops_approx':(float('nan') if policy=='Oracle' else budget*gf),'model_latency_ms_per_sample':(float('nan') if policy=='Oracle' else d['lat']/max(1,d['n'])),'oracle_is_diagnostic':policy=='Oracle','deployable_cost_valid':policy!='Oracle','three_query_equals_full_checked':bool(checked if budget==3 else False)})
    return pd.DataFrame(rows)






def _falsification_checkpoint_path(falsification_root:Path,proto_name:str,fold:str)->Path:
    if proto_name=='protocol_a': return falsification_root/'protocol_a'/'kofu'/fold/'best.pt'
    if proto_name.startswith('protocol_c/'):
        return falsification_root/'protocol_c'/proto_name.split('/',1)[1]/'kofu'/fold/'best.pt'
    raise ValueError(proto_name)

def read_falsification_max_revision(falsification_root:Path)->float:
    mf=falsification_root/'AUDIT_MANIFEST.json'
    if mf.exists():
        try:
            d=json.loads(mf.read_text(encoding='utf-8'))
            return float(d.get('args',{}).get('max_revision',3.0))
        except Exception: pass
    return 3.0

def load_falsification_core(falsification_root:Path,proto_name:str,fold:str,ncls:int,device:torch.device,max_revision:float)->Tuple[PrototypeFalsification,np.ndarray,np.ndarray,Path,Dict[str,Any]]:
    bp=_falsification_checkpoint_path(falsification_root,proto_name,fold)
    if not bp.exists(): raise FileNotFoundError(f'Missing completed Falsification checkpoint: {bp}. Run the KOFU reproduction pipeline first.')
    ck=torch.load(bp,map_location='cpu',weights_only=False)
    core=PrototypeFalsification(ncls,4,max_revision).to(device); core.load_state_dict(ck['state_dict'],strict=True); core.eval()
    for p in core.parameters(): p.requires_grad_(False)
    pe=np.asarray(ck['proto_evidence'],dtype=np.float32); ps=np.asarray(ck['proto_semantic'],dtype=np.float32)
    return core,pe,ps,bp,ck

class ConservativeRevision(nn.Module):








    def __init__(self,core:PrototypeFalsification,policy:Dict[str,Any]):
        super().__init__(); self.core=core; self.policy=dict(policy); self.ncls=core.ncls
        for p in self.core.parameters(): p.requires_grad_(False)
    def _candidate(self,base_logits:torch.Tensor,dist:torch.Tensor,hyp:torch.Tensor)->torch.Tensor:
        mode=str(self.policy.get('candidate_mode','top3'))
        if mode=='top2':
            return base_logits.topk(min(2,self.ncls),1).indices[:,-1] if self.ncls>1 else hyp
        if mode=='top3':
            k=min(3,self.ncls); ids=base_logits.topk(k,1).indices
            if k<=1:return hyp
            alt=ids[:,1:]; dd=dist.gather(1,alt); return alt.gather(1,dd.argmin(1,keepdim=True)).squeeze(1)
        if mode=='all':
            dd=dist.clone(); dd.scatter_(1,hyp[:,None],float('inf')); return dd.argmin(1)
        raise ValueError(f'Unknown candidate_mode={mode}')
    def forward(self,base_logits:torch.Tensor,evidence:torch.Tensor,semantic_prob:torch.Tensor,proto_evidence:torch.Tensor,proto_semantic:torch.Tensor)->Dict[str,torch.Tensor]:
        r=self.core(base_logits.float(),evidence.float(),semantic_prob.float(),proto_evidence.float(),proto_semantic.float())
        dist=r['distance'].float(); hyp=base_logits.float().argmax(1); cand=self._candidate(base_logits.float(),dist,hyp)
        hd=dist.gather(1,hyp[:,None]).squeeze(1); cd=dist.gather(1,cand[:,None]).squeeze(1);adv=hd-cd
        top=base_logits.float().topk(min(2,self.ncls),1).values; margin=(top[:,0]-top[:,1]) if self.ncls>1 else torch.full_like(hd,float('inf'))
        enabled=bool(self.policy.get('enabled',False))
        if enabled:
            gate=(cand!=hyp)&(hd>=float(self.policy['contradiction_min']))&(adv>=float(self.policy['advantage_min']))&(margin<=float(self.policy['margin_max']))
        else: gate=torch.zeros_like(hyp,dtype=torch.bool)
        final=base_logits.float().clone()
        if gate.any():
            rows=torch.where(gate)[0]; cc=cand[rows]; mx=final[rows].max(1).values; cur=final[rows,cc];eps=float(self.policy.get('flip_eps',0.05));final[rows,cc]=mx+torch.clamp(torch.full_like(mx,eps),min=1e-4)
        compat=-dist
        return {'logits':final,'base_logits':base_logits.float(),'distance':dist,'hypothesis':hyp,'candidate':cand,'contradiction':hd,'advantage':adv,'margin':margin,'gate':gate.float(),'alpha':torch.tensor(1.0 if enabled else 0.0,device=base_logits.device),'compatibility':compat}

def _core_signal_arrays(core:PrototypeFalsification,cache:Dict[str,np.ndarray],pe:np.ndarray,ps:np.ndarray,device:torch.device,batch:int=512)->Dict[str,np.ndarray]:
    core.eval();out={'base_logits':[],'distance':[],'contradiction':[]};N=len(cache['y']);PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device)
    with torch.no_grad():
        for st in range(0,N,batch):
            sl=slice(st,min(N,st+batch));b=torch.tensor(cache['base_logits'][sl],dtype=torch.float32,device=device);e=torch.tensor(cache['agg_evidence'][sl],dtype=torch.float32,device=device);sp=torch.tensor(cache['semantic_pred'][sl],dtype=torch.float32,device=device);r=core(b,e,sp,PE,PS)
            out['base_logits'].append(b.cpu().numpy());out['distance'].append(r['distance'].float().cpu().numpy());out['contradiction'].append(r['contradiction'].float().cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def _candidate_np(base:np.ndarray,dist:np.ndarray,mode:str)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    base=np.asarray(base,dtype=np.float32);dist=np.asarray(dist,dtype=np.float32);N,C=base.shape;hyp=base.argmax(1);order=np.argsort(-base,axis=1)
    if mode=='top2': cand=order[:,1] if C>1 else hyp.copy()
    elif mode=='top3':
        k=min(3,C);alts=order[:,1:k]
        if alts.shape[1]==0:cand=hyp.copy()
        else:
            dd=np.take_along_axis(dist,alts,axis=1);j=dd.argmin(1);cand=alts[np.arange(N),j]
    elif mode=='all':
        dd=dist.copy();dd[np.arange(N),hyp]=np.inf;cand=dd.argmin(1)
    else:raise ValueError(mode)
    hd=dist[np.arange(N),hyp];cd=dist[np.arange(N),cand];adv=hd-cd
    if C>1: margin=base[np.arange(N),order[:,0]]-base[np.arange(N),order[:,1]]
    else: margin=np.full(N,np.inf,dtype=np.float32)
    return hyp,cand,adv,margin

def _fast_macro_metrics(y:np.ndarray,pred:np.ndarray,labels:Sequence[str])->Dict[str,float]:
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64);C=len(labels);cm=np.zeros((C,C),dtype=np.int64);np.add.at(cm,(y,pred),1);tp=np.diag(cm).astype(np.float64);fp=cm.sum(0)-tp;fn=cm.sum(1)-tp;den=2*tp+fp+fn;f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0);acc=float(tp.sum()/max(1,len(y)));fs=float('nan')
    if SAFE_LABEL in labels:
        si=labels.index(SAFE_LABEL);non=y!=si;fs=float(np.sum((pred==si)&non)/max(1,int(non.sum())))
    return {'acc':acc,'macro_f1':float(f1.mean()),'false_safe':fs}

def _policy_pred(sig:Dict[str,np.ndarray],policy:Dict[str,Any])->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    base=sig['base_logits'];dist=sig['distance'];hyp,cand,adv,margin=_candidate_np(base,dist,str(policy.get('candidate_mode','top3')));hd=sig['contradiction']
    if bool(policy.get('enabled',False)):
        gate=(cand!=hyp)&(hd>=float(policy['contradiction_min']))&(adv>=float(policy['advantage_min']))&(margin<=float(policy['margin_max']))
    else:gate=np.zeros(len(hyp),dtype=bool)
    pred=hyp.copy();pred[gate]=cand[gate];return pred,gate,hyp,cand,adv

def _revision_stats(y:np.ndarray,hyp:np.ndarray,pred:np.ndarray,gate:np.ndarray)->Dict[str,float]:
    y=np.asarray(y);rev=np.asarray(gate,dtype=bool);corrected=rev&(hyp!=y)&(pred==y);corrupted=rev&(hyp==y)&(pred!=y);wrong_to_wrong=rev&(hyp!=y)&(pred!=y)
    nrev=int(rev.sum());return {'revision_rate':float(nrev/max(1,len(y))),'revision_count':nrev,'revision_corrected':int(corrected.sum()),'revision_corrupted':int(corrupted.sum()),'revision_wrong_to_wrong':int(wrong_to_wrong.sum()),'revision_precision':float(corrected.sum()/max(1,nrev)),'revision_net_corrections':int(corrected.sum()-corrupted.sum())}

def _quantile_grid(x:np.ndarray,qs:Sequence[float],extra:Sequence[float]=())->List[float]:
    x=np.asarray(x,dtype=np.float64);x=x[np.isfinite(x)];vals=list(extra)
    if len(x):vals.extend(np.quantile(x,np.asarray(qs,dtype=float)).tolist())
    return sorted(set(float(v) for v in vals if np.isfinite(v)))

def calibrate_conservative_revision(core:PrototypeFalsification,model:EvidenceFoundation,cache_va:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,out_dir:Path,args:argparse.Namespace,clock:GlobalClock,idx:int,exp_name:str)->Dict[str,Any]:
    ensure_dir(out_dir);pp=out_dir/'policy.json';done=out_dir/'DONE.json'
    if pp.exists() and done.exists() and not args.overwrite:
        clock.complete(.01);return json.loads(pp.read_text(encoding='utf-8'))
    device=torch.device(args.device);t0=time.time();log(f'{exp_name}: calibrating discrete revision on validation only (clean + D/E hard banks).')
    hard=build_robust_condition_cache(model,cache_va,device);banks={'clean':cache_va,**hard};sigs={k:_core_signal_arrays(core,c,pe,ps,device) for k,c in banks.items()};ys={k:np.asarray(c['y'],dtype=np.int64) for k,c in banks.items()}
    base_metrics={k:_fast_macro_metrics(ys[k],sigs[k]['base_logits'].argmax(1),labels) for k in banks};base_rob=float(np.mean([base_metrics[k]['macro_f1'] for k in hard]));base_clean=base_metrics['clean']
    identity={'enabled':False,'candidate_mode':'top3','contradiction_min':float('inf'),'advantage_min':float('inf'),'margin_max':0.0,'flip_eps':args.revision_flip_eps}
    best=dict(identity);best_score=(base_rob,float(base_clean['macro_f1']),0.0,0.0);rows=[]
    modes=('top2','top3','all')
    for mode in modes:
        pool_hd=[];pool_adv=[];pool_margin=[]
        for k,sig in sigs.items():
            hyp,cand,adv,margin=_candidate_np(sig['base_logits'],sig['distance'],mode);m=(cand!=hyp)&np.isfinite(adv)&np.isfinite(margin);pool_hd.append(sig['contradiction'][m]);pool_adv.append(adv[m]);pool_margin.append(margin[m])
        hd_all=np.concatenate(pool_hd) if pool_hd else np.array([]);adv_all=np.concatenate(pool_adv) if pool_adv else np.array([]);mg_all=np.concatenate(pool_margin) if pool_margin else np.array([])
        hgrid=_quantile_grid(hd_all,[0,.25,.5,.7,.85,.95]);agrid=_quantile_grid(adv_all[adv_all>=0],[0,.2,.4,.6,.8,.9],[0.0]);mgrid=_quantile_grid(mg_all,[.15,.3,.5,.7,.85,1.0])
        if args.dry_run:
            hgrid=hgrid[::max(1,len(hgrid)//3)][:4] or hgrid;agrid=agrid[::max(1,len(agrid)//3)][:4] or agrid;mgrid=mgrid[::max(1,len(mgrid)//3)][:4] or mgrid
        for ht in hgrid:
            for at in agrid:
                for mt in mgrid:
                    pol={'enabled':True,'candidate_mode':mode,'contradiction_min':float(ht),'advantage_min':float(at),'margin_max':float(mt),'flip_eps':float(args.revision_flip_eps)}
                    cp,cg,ch,cc,ca=_policy_pred(sigs['clean'],pol);cm=_fast_macro_metrics(ys['clean'],cp,labels);cs=_revision_stats(ys['clean'],ch,cp,cg)
                    eligible=(cm['macro_f1']>=base_clean['macro_f1']-args.val_tolerance and cm['acc']>=base_clean['acc']-args.val_tolerance and (not np.isfinite(base_clean['false_safe']) or cm['false_safe']<=base_clean['false_safe']+args.fs_tolerance))
                    hfs=[];revs=[];nets=[]
                    for k in hard:
                        pr,ga,hy,ca,_=_policy_pred(sigs[k],pol);mm=_fast_macro_metrics(ys[k],pr,labels);rs=_revision_stats(ys[k],hy,pr,ga);hfs.append(mm['macro_f1']);revs.append(rs['revision_rate']);nets.append(rs['revision_net_corrections'])
                    rob=float(np.mean(hfs));rrev=float(np.mean(revs));net=float(np.sum(nets));gain=rob-base_rob
                    row={'candidate_mode':mode,'contradiction_min':ht,'advantage_min':at,'margin_max':mt,'eligible':bool(eligible),'clean_macro_f1':cm['macro_f1'],'clean_acc':cm['acc'],'clean_false_safe':cm['false_safe'],'clean_revision_rate':cs['revision_rate'],'clean_net_corrections':cs['revision_net_corrections'],'robust_mean_macro_f1':rob,'robust_gain':gain,'robust_mean_revision_rate':rrev,'robust_net_corrections':net};rows.append(row)

                    if eligible and gain>=args.revision_min_robust_gain and rrev>0:
                        score=(rob,float(cm['macro_f1']),net,-float(cs['revision_rate']))
                        if score>best_score:best_score=score;best=pol

    cp,cg,ch,_,_=_policy_pred(sigs['clean'],best);chosen_clean=_fast_macro_metrics(ys['clean'],cp,labels);chosen_clean_stats=_revision_stats(ys['clean'],ch,cp,cg);hf=[];hrs=[]
    for k in hard:
        pr,ga,hy,_,_=_policy_pred(sigs[k],best);hf.append(_fast_macro_metrics(ys[k],pr,labels)['macro_f1']);hrs.append(_revision_stats(ys[k],hy,pr,ga))
    chosen_rob=float(np.mean(hf));best.update({'calibration':'conservative_revision','baseline_clean':base_clean,'baseline_robust_mean_macro_f1':base_rob,'selected_clean':chosen_clean,'selected_clean_revision':chosen_clean_stats,'selected_robust_mean_macro_f1':chosen_rob,'selected_robust_gain':chosen_rob-base_rob,'selected_robust_mean_revision_rate':float(np.mean([r['revision_rate'] for r in hrs])),'selected_robust_net_corrections':int(np.sum([r['revision_net_corrections'] for r in hrs])),'selection_uses_test':False})
    pp.write_text(json.dumps(best,indent=2,ensure_ascii=False),encoding='utf-8');pd.DataFrame(rows).sort_values(['eligible','robust_mean_macro_f1','clean_macro_f1'],ascending=[False,False,False]).head(200).to_csv(out_dir/'calibration_top200.csv',index=False);write_json(done,{'completed':True,'policy':best,'seconds':time.time()-t0})

    bar=ExperimentBar(clock,idx,exp_name,1,0);bar.epoch(1,time.time()-t0,0.0,chosen_rob,chosen_rob);bar.close();return best

def evaluate_kofu_ood_calibrated(side:ConservativeRevision,pe:np.ndarray,ps:np.ndarray,known_cache:Dict[str,np.ndarray],unknown_cache:Dict[str,np.ndarray],val_known_cache:Dict[str,np.ndarray],labels:Sequence[str],device:torch.device,proto:str,fold:str,out_dir:Path)->List[Dict[str,Any]]:

    kr=apply_falsification_np(side.core,known_cache,pe,ps,device);ur=apply_falsification_np(side.core,unknown_cache,pe,ps,device);vr=apply_falsification_np(side.core,val_known_cache,pe,ps,device)
    ke=energy_np(kr['logits']);ue=energy_np(ur['logits']);ve=energy_np(vr['logits']);kc=kr['contradiction'];uc=ur['contradiction'];vc=vr['contradiction'];me,se=zfit(ve);mc,sc=zfit(vc);kf=zapply(ke,me,se)+zapply(kc,mc,sc);uf=zapply(ue,me,se)+zapply(uc,mc,sc)
    rows=[{'protocol':proto,'fold':fold,'method':'KOFU','ood_score':'Energy',**ood_metrics(ke,ue)},{'protocol':proto,'fold':fold,'method':'KOFU','ood_score':'FalsificationScore',**ood_metrics(kf,uf)}];ensure_dir(out_dir);pd.DataFrame(rows).to_csv(out_dir/'ood_metrics.csv',index=False);return rows

def _test_revision_row(y:np.ndarray,base_logits:np.ndarray,revised_logits:np.ndarray,gate:np.ndarray)->Dict[str,Any]:
    hyp=np.asarray(base_logits).argmax(1);pred=np.asarray(revised_logits).argmax(1);return _revision_stats(np.asarray(y),hyp,pred,np.asarray(gate)>.5)





class AnalysisBar:

    def __init__(self, clock:GlobalClock, idx:int, name:str, total:int):
        self.clock=clock; self.idx=idx; self.name=name; self.total=max(1,int(total)); self.start=time.time(); self.last=self.start; self.times=[]
        self.bar=tqdm(total=self.total,desc=f'[EXP {idx}/{clock.total}] {name}',dynamic_ncols=True,leave=True)
    def step(self,label:str=''):
        t=time.time(); self.times.append(t-self.last); self.last=t
        done=self.bar.n+1; av=float(np.mean(self.times)) if self.times else 1.0; rem=max(0,self.total-done)*av; full=max(av*self.total,t-self.start)
        self.bar.set_postfix_str(f'{label} exp_eta={format_seconds(rem)} all_eta={format_seconds(self.clock.all_eta(rem,full))}',refresh=False)
        self.bar.update(1)
    def close(self):

        self.bar.close(); self.clock.complete(time.time()-self.start)


def _manifest_max_revision(root:Path)->float:
    mf=root/'AUDIT_MANIFEST.json'
    if mf.exists():
        try:return float(json.loads(mf.read_text(encoding='utf-8')).get('args',{}).get('max_revision',3.0))
        except Exception:pass
    return 3.0


def _clean_falsification_checkpoint_path(foundation_root:Path,proto_name:str,fold:str)->Path:
    if proto_name=='protocol_a':return foundation_root/'protocol_a'/'kofu'/fold/'best.pt'
    if proto_name.startswith('protocol_c/'):
        return foundation_root/'protocol_c'/proto_name.split('/',1)[1]/'kofu'/fold/'best.pt'
    raise ValueError(proto_name)


def load_clean_falsification(foundation_root:Path,proto_name:str,fold:str,ncls:int,device:torch.device)->Tuple[PrototypeFalsification,np.ndarray,np.ndarray,Path]:
    bp=_clean_falsification_checkpoint_path(foundation_root,proto_name,fold)
    if not bp.exists():raise FileNotFoundError(f'Missing clean-only Falsification checkpoint: {bp}')
    ck=torch.load(bp,map_location='cpu',weights_only=False); mx=_manifest_max_revision(foundation_root)
    core=PrototypeFalsification(ncls,4,mx).to(device);core.load_state_dict(ck['state_dict'],strict=True);core.eval()
    for p in core.parameters():p.requires_grad_(False)
    return core,np.asarray(ck['proto_evidence'],dtype=np.float32),np.asarray(ck['proto_semantic'],dtype=np.float32),bp


@torch.no_grad()
def extract_foundation_cache_with_bar(model:EvidenceFoundation,loader:DataLoader,device:torch.device,amp:bool,bar:AnalysisBar,max_batches:int=0)->Dict[str,np.ndarray]:
    model.eval();ys=[];og=[];ovf=[];ovl=[];ovocc=[];pools=[];evs=[];unc=[];occ=[];sem=[];valid=[];base_logits=[];agg_evidence=[];sem_pred=[];rel_weights=[]
    lim=min(len(loader),max_batches) if max_batches else len(loader)
    for bi,b in enumerate(loader):
        if max_batches and bi>=max_batches:break
        b=move_batch(b,device);x=b['x']
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            o=model.overview(x); enc=model.encode_all(x,(0,1,2)); agg=model.aggregate(enc,o)
        pp=[];ee=[];uu=[];oo=[]
        for v in range(3):pp.append(enc[v]['pool']);ee.append(enc[v]['evidence']);uu.append(enc[v]['unc']);oo.append(enc[v]['occ_logit'])
        sl=agg['semantic_logits'];sp=torch.cat([F.softmax(sl['gaze'],1),F.softmax(sl['hands'],1),torch.sigmoid(sl['talk'])[:,None],torch.sigmoid(sl['objects'])],1)
        ys.append(b['y'].cpu().numpy());og.append(o['global'].float().cpu().numpy());ovf.append(o['view_features'].float().cpu().numpy());ovl.append(o['view_logits'].float().cpu().numpy());ovocc.append(o['occ_logits'].float().cpu().numpy());pools.append(torch.stack(pp,1).float().cpu().numpy());evs.append(torch.stack(ee,1).float().cpu().numpy());unc.append(torch.stack(uu,1).float().cpu().numpy());occ.append(torch.stack(oo,1).float().cpu().numpy());sem.append(b['semantic'].cpu().numpy());valid.append(b['valid'].cpu().numpy());base_logits.append(agg['logits'].float().cpu().numpy());agg_evidence.append(agg['evidence'].float().cpu().numpy());sem_pred.append(sp.float().cpu().numpy());rel_weights.append(agg['weights'].float().cpu().numpy())
        bar.step(f'batch={bi+1}/{lim}')
    return {'y':np.concatenate(ys),'overview':np.concatenate(og),'overview_views':np.concatenate(ovf),'overview_view_logits':np.concatenate(ovl),'overview_occ_logits':np.concatenate(ovocc),'pools':np.concatenate(pools),'evidence_by_view':np.concatenate(evs),'unc':np.concatenate(unc),'occ_logits':np.concatenate(occ),'semantic':np.concatenate(sem),'valid':np.concatenate(valid),'base_logits':np.concatenate(base_logits),'agg_evidence':np.concatenate(agg_evidence),'semantic_pred':np.concatenate(sem_pred),'rel_weights':np.concatenate(rel_weights)}


def get_test_cache(model:EvidenceFoundation,dfp:pd.DataFrame,labels:Sequence[str],timelines:Dict[str,EvidenceTimeline],args:argparse.Namespace,path:Path,clock:GlobalClock,idx:int,name:str)->Dict[str,np.ndarray]:
    ensure_dir(path.parent)
    if path.exists() and not args.overwrite:
        bar=AnalysisBar(clock,idx,name,1)
        try:
            c=_load_npz_dict(path);validate_feature_cache(c,model.ncls,name);bar.step('cache-hit');return c
        finally:bar.close()
    seed=stable_seed('EvaluationCache',name,base=args.seed);ds=DMDMultiViewDataset(dfp,labels,timelines,args.frames,args.size,False,seed,True,args.dry_per_class if args.dry_run else 0);ld=make_loader(ds,args.eval_batch,0 if args.dry_run else args.workers,False,seed);lim=1 if args.dry_run else 0
    bar=AnalysisBar(clock,idx,name,min(len(ld),lim) if lim else len(ld))
    try:
        c=extract_foundation_cache_with_bar(model,ld,torch.device(args.device),args.amp,bar,lim);validate_feature_cache(c,model.ncls,name);np.savez_compressed(path,**c);return c
    finally:bar.close()


def _distance_components_np(core:PrototypeFalsification,cache:Dict[str,np.ndarray],pe:np.ndarray,ps:np.ndarray,uniform_query:bool=False)->Dict[str,np.ndarray]:
    ev=np.asarray(cache['agg_evidence'],dtype=np.float32);sp=np.asarray(cache['semantic_pred'],dtype=np.float32);base=np.asarray(cache['base_logits'],dtype=np.float32);pe=np.asarray(pe,dtype=np.float32);ps=np.asarray(ps,dtype=np.float32)
    en=ev/(np.linalg.norm(ev,axis=-1,keepdims=True)+1e-8);pn=pe/(np.linalg.norm(pe,axis=-1,keepdims=True)+1e-8);cos=np.einsum('nqd,cqd->ncq',en,pn)
    if uniform_query:qw=np.full(ev.shape[1],1.0/ev.shape[1],dtype=np.float32)
    else:qw=torch.softmax(core.query_logits.detach().float().cpu(),0).numpy().astype(np.float32)
    df=((1.0-cos)*qw[None,None,:]).sum(-1);ds=np.abs(sp[:,None,:]-ps[None,:,:]).mean(-1);sw=float(F.softplus(core.semantic_raw.detach().float()).cpu());sem=sw*ds;full=df+sem;hyp=base.argmax(1);ii=np.arange(len(hyp))
    return {'evidence_distance':df,'semantic_distance':sem,'full_distance':full,'evidence_contradiction':df[ii,hyp],'semantic_contradiction':sem[ii,hyp],'full_contradiction':full[ii,hyp]}


def _combo_score(a:np.ndarray,b:np.ndarray,va:np.ndarray,vb:np.ndarray)->np.ndarray:
    ma,sa=zfit(va);mb,sb=zfit(vb);return zapply(a,ma,sa)+zapply(b,mb,sb)


def unknown_signal_analysis(robust_core:PrototypeFalsification,rpe:np.ndarray,rps:np.ndarray,clean_core:PrototypeFalsification,cpe:np.ndarray,cps:np.ndarray,known:Dict[str,np.ndarray],unknown:Dict[str,np.ndarray],val:Dict[str,np.ndarray],protocol:str,fold:str,bar:AnalysisBar,device:torch.device)->List[Dict[str,Any]]:
    rrk=apply_falsification_np(robust_core,known,rpe,rps,device);rru=apply_falsification_np(robust_core,unknown,rpe,rps,device);rrv=apply_falsification_np(robust_core,val,rpe,rps,device);bar.step('robust-core')
    rk=_distance_components_np(robust_core,known,rpe,rps);ru=_distance_components_np(robust_core,unknown,rpe,rps);rv=_distance_components_np(robust_core,val,rpe,rps);rku=_distance_components_np(robust_core,known,rpe,rps,True);ruu=_distance_components_np(robust_core,unknown,rpe,rps,True);rvu=_distance_components_np(robust_core,val,rpe,rps,True);bar.step('decompose')
    crk=apply_falsification_np(clean_core,known,cpe,cps,device);cru=apply_falsification_np(clean_core,unknown,cpe,cps,device);crv=apply_falsification_np(clean_core,val,cpe,cps,device);ck=_distance_components_np(clean_core,known,cpe,cps);cu=_distance_components_np(clean_core,unknown,cpe,cps);cv=_distance_components_np(clean_core,val,cpe,cps);bar.step('clean-core')
    be_k=energy_np(known['base_logits']);be_u=energy_np(unknown['base_logits']);be_v=energy_np(val['base_logits']);bm_k=msp_ood_np(known['base_logits']);bm_u=msp_ood_np(unknown['base_logits'])
    re_k=energy_np(rrk['logits']);re_u=energy_np(rru['logits']);re_v=energy_np(rrv['logits']);ce_k=energy_np(crk['logits']);ce_u=energy_np(cru['logits']);ce_v=energy_np(crv['logits'])
    variants={
        'Foundation_MSP':(bm_k,bm_u),
        'Foundation_Energy':(be_k,be_u),
        'Falsification_Energy':(re_k,re_u),
        'EvidenceContradiction_only':(rk['evidence_contradiction'],ru['evidence_contradiction']),
        'SemanticContradiction_only':(rk['semantic_contradiction'],ru['semantic_contradiction']),
        'FullContradiction_only':(rk['full_contradiction'],ru['full_contradiction']),
        'Energy+Evidence':(_combo_score(re_k,rk['evidence_contradiction'],re_v,rv['evidence_contradiction']),_combo_score(re_u,ru['evidence_contradiction'],re_v,rv['evidence_contradiction'])),
        'Energy+Semantic':(_combo_score(re_k,rk['semantic_contradiction'],re_v,rv['semantic_contradiction']),_combo_score(re_u,ru['semantic_contradiction'],re_v,rv['semantic_contradiction'])),
        'Full_Falsification':(_combo_score(re_k,rk['full_contradiction'],re_v,rv['full_contradiction']),_combo_score(re_u,ru['full_contradiction'],re_v,rv['full_contradiction'])),
        'Full_Falsification_uniform_query':(_combo_score(re_k,rku['full_contradiction'],re_v,rvu['full_contradiction']),_combo_score(re_u,ruu['full_contradiction'],re_v,rvu['full_contradiction'])),
        'CleanOnlyCore_Full_Falsification':(_combo_score(ce_k,ck['full_contradiction'],ce_v,cv['full_contradiction']),_combo_score(ce_u,cu['full_contradiction'],ce_v,cv['full_contradiction'])),
        'FoundationEnergy+FullContradiction':(_combo_score(be_k,rk['full_contradiction'],be_v,rv['full_contradiction']),_combo_score(be_u,ru['full_contradiction'],be_v,rv['full_contradiction'])),
    }
    rows=[]
    for name,(ks,us) in variants.items():rows.append({'fold':fold,'protocol':protocol,'ablation':name,**ood_metrics(ks,us)})
    bar.step('metrics');return rows


def _variant_policy_pred(sig:Dict[str,np.ndarray],policy:Dict[str,Any])->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    base=sig['base_logits'];dist=sig['distance'];hyp,cand,adv,margin=_candidate_np(base,dist,str(policy.get('candidate_mode','top3')));hd=sig['contradiction'];gate=(cand!=hyp)
    if bool(policy.get('enabled',False)):
        if policy.get('use_contradiction',True):gate&=hd>=float(policy.get('contradiction_min',-np.inf))
        if policy.get('use_advantage',True):gate&=adv>=float(policy.get('advantage_min',-np.inf))
        if policy.get('use_margin',True):gate&=margin<=float(policy.get('margin_max',np.inf))
    else:gate[:]=False
    pred=hyp.copy();pred[gate]=cand[gate];return pred,gate,hyp,cand,adv


def calibrate_revision_policy(core:PrototypeFalsification,model:EvidenceFoundation,cache_va:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,args:argparse.Namespace,use_advantage:bool,use_margin:bool,candidate_mode:str='top3',bank_mode:str='clean_hard')->Dict[str,Any]:
    device=torch.device(args.device);hard=build_robust_condition_cache(model,cache_va,device);banks={'clean':cache_va,**hard};sigs={k:_core_signal_arrays(core,c,pe,ps,device) for k,c in banks.items()};ys={k:np.asarray(c['y'],dtype=np.int64) for k,c in banks.items()};base={k:_fast_macro_metrics(ys[k],sigs[k]['base_logits'].argmax(1),labels) for k in banks};base_hard=float(np.mean([base[k]['macro_f1'] for k in hard]));base_clean=base['clean']
    identity={'enabled':False,'candidate_mode':candidate_mode,'use_contradiction':True,'use_advantage':use_advantage,'use_margin':use_margin,'contradiction_min':float('inf'),'advantage_min':float('inf'),'margin_max':0.0,'flip_eps':args.revision_flip_eps,'bank_mode':bank_mode}
    best=dict(identity);best_obj=base_clean['macro_f1'] if bank_mode=='clean_only' else base_hard;best_score=(float(best_obj),float(base_clean['macro_f1']),0.0,0.0)
    pool_hd=[];pool_adv=[];pool_margin=[]


    grid_keys=['clean'] if bank_mode=='clean_only' else list(sigs.keys())
    for k in grid_keys:
        sig=sigs[k];hyp,cand,adv,margin=_candidate_np(sig['base_logits'],sig['distance'],candidate_mode);m=(cand!=hyp)&np.isfinite(adv)&np.isfinite(margin);pool_hd.append(sig['contradiction'][m]);pool_adv.append(adv[m]);pool_margin.append(margin[m])
    hd=np.concatenate(pool_hd) if pool_hd else np.array([]);ad=np.concatenate(pool_adv) if pool_adv else np.array([]);mg=np.concatenate(pool_margin) if pool_margin else np.array([])
    hgrid=_quantile_grid(hd,[.25,.5,.7,.85,.95]);agrid=_quantile_grid(ad[ad>=0],[0,.25,.5,.7,.85,.95],[0.0]) if use_advantage else [-np.inf];mgrid=_quantile_grid(mg,[.15,.3,.5,.7,.85,1.0]) if use_margin else [np.inf]
    if args.dry_run:hgrid=hgrid[:3] or hgrid;agrid=agrid[:3] or agrid;mgrid=mgrid[:3] or mgrid
    for ht in hgrid:
        for at in agrid:
            for mt in mgrid:
                pol={'enabled':True,'candidate_mode':candidate_mode,'use_contradiction':True,'use_advantage':use_advantage,'use_margin':use_margin,'contradiction_min':float(ht),'advantage_min':float(at),'margin_max':float(mt),'flip_eps':args.revision_flip_eps,'bank_mode':bank_mode}
                cp,cg,ch,_,_=_variant_policy_pred(sigs['clean'],pol);cm=_fast_macro_metrics(ys['clean'],cp,labels);cs=_revision_stats(ys['clean'],ch,cp,cg)
                eligible=(cm['macro_f1']>=base_clean['macro_f1']-args.val_tolerance and cm['acc']>=base_clean['acc']-args.val_tolerance and (not np.isfinite(base_clean['false_safe']) or cm['false_safe']<=base_clean['false_safe']+args.fs_tolerance))
                hfs=[];revs=[];nets=[]
                for k in hard:
                    pr,ga,hy,_,_=_variant_policy_pred(sigs[k],pol);hfs.append(_fast_macro_metrics(ys[k],pr,labels)['macro_f1']);rs=_revision_stats(ys[k],hy,pr,ga);revs.append(rs['revision_rate']);nets.append(rs['revision_net_corrections'])
                rob=float(np.mean(hfs));rrev=float(np.mean(revs));net=float(np.sum(nets));obj=float(cm['macro_f1']) if bank_mode=='clean_only' else rob;base_obj=float(base_clean['macro_f1']) if bank_mode=='clean_only' else base_hard


                if bank_mode=='clean_only':
                    nonzero=cs['revision_rate']>0; tie_net=float(cs['revision_net_corrections'])
                else:
                    nonzero=(rrev>0 or cs['revision_rate']>0); tie_net=net
                if eligible and obj>=base_obj-1e-12 and nonzero:
                    score=(obj,float(cm['macro_f1']),tie_net,-float(cs['revision_rate']))
                    if score>best_score:best_score=score;best=pol
    best.update({'selection_uses_test':False,'baseline_clean':base_clean,'baseline_robust_mean_macro_f1':base_hard})
    return best


def apply_revision_policy_to_cache(core:PrototypeFalsification,cache:Dict[str,np.ndarray],pe:np.ndarray,ps:np.ndarray,labels:Sequence[str],policy:Dict[str,Any],device:torch.device)->Dict[str,Any]:
    sig=_core_signal_arrays(core,cache,pe,ps,device);pred,gate,hyp,cand,adv=_variant_policy_pred(sig,policy);z=np.asarray(sig['base_logits'],dtype=np.float32).copy();rows=np.where(gate)[0]
    if len(rows):
        mx=z[rows].max(1);z[rows,cand[rows]]=mx+max(1e-4,float(policy.get('flip_eps',.05)))
    m=metrics_from_logits(cache['y'],z,labels);rs=_revision_stats(cache['y'],hyp,pred,gate);return {'logits':z,'metrics':m,'stats':rs,'gate':gate,'pred':pred,'hyp':hyp,'candidate':cand,'advantage':adv}


def load_reference_revision_policy(foundation_root:Path,fold:str)->Dict[str,Any]:
    p=foundation_root/'protocol_a'/'kofu'/fold/'revision_calibration'/'policy.json'
    if not p.exists():raise FileNotFoundError(f'Missing completed Conservative Revision policy: {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def load_reference_acquisition_policy(foundation_root:Path,fold:str,ncls:int,device:torch.device)->SelectiveEvidenceAcquisitionPolicy:
    p=foundation_root/'protocol_e'/fold/'selective_evidence_acquisition'/'best.pt'
    if not p.exists():raise FileNotFoundError(f'Missing completed Selective Evidence Acquisition checkpoint: {p}')
    return load_selective_acquisition_policy(p,ncls,device)

def revision_component_rows(core:PrototypeFalsification,model:EvidenceFoundation,cva:Dict[str,np.ndarray],ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,args:argparse.Namespace,fold:str,bar:AnalysisBar,outdir:Path,reference_revision_policy:Dict[str,Any])->List[Dict[str,Any]]:
    rows=[];base=metrics_from_logits(ct['y'],ct['base_logits'],labels);rows.append({'fold':fold,'ablation':'Foundation',**base,'revision_rate':0.0,'revision_count':0,'revision_corrected':0,'revision_corrupted':0,'revision_net_corrections':0});bar.step('Foundation')
    cont=apply_falsification_np(core,ct,pe,ps,torch.device(args.device));cm=metrics_from_logits(ct['y'],cont['logits'],labels);hyp=ct['base_logits'].argmax(1);pred=cont['logits'].argmax(1);gate=pred!=hyp;rs=_revision_stats(ct['y'],hyp,pred,gate);rows.append({'fold':fold,'ablation':'ContinuousFalsificationRevision',**cm,**rs});bar.step('continuous')
    specs=[('Contradiction_only',False,False),('Contradiction+Advantage',True,False),('Contradiction+PrototypeAdvantage+Margin',True,True)]
    for name,ua,um in specs:
        pol=calibrate_revision_policy(core,model,cva,labels,pe,ps,args,ua,um,'top3','clean_hard');r=apply_revision_policy_to_cache(core,ct,pe,ps,labels,pol,torch.device(args.device));rows.append({'fold':fold,'ablation':name,**r['metrics'],**r['stats'],'policy_enabled':pol['enabled'],'candidate_mode':pol['candidate_mode'],'bank_mode':pol['bank_mode']});ensure_dir(outdir);write_json(outdir/f'{name}_policy.json',pol);bar.step(name)
    fr=apply_revision_policy_to_cache(core,ct,pe,ps,labels,reference_revision_policy,torch.device(args.device));rows.append({'fold':fold,'ablation':'ConservativeRevisionCandidateSearch',**fr['metrics'],**fr['stats'],'policy_enabled':reference_revision_policy.get('enabled',False),'candidate_mode':reference_revision_policy.get('candidate_mode',''),'bank_mode':'clean_hard'});bar.step('ConservativeRevision')
    return rows


def revision_design_rows(core:PrototypeFalsification,model:EvidenceFoundation,cva:Dict[str,np.ndarray],ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,args:argparse.Namespace,fold:str,bar:AnalysisBar,outdir:Path)->List[Dict[str,Any]]:
    rows=[]
    for mode,bank in [('top2','clean_hard'),('top3','clean_hard'),('all','clean_hard'),('top3','clean_only')]:
        name=f'candidate_{mode}__bank_{bank}';pol=calibrate_revision_policy(core,model,cva,labels,pe,ps,args,True,True,mode,bank);r=apply_revision_policy_to_cache(core,ct,pe,ps,labels,pol,torch.device(args.device));rows.append({'fold':fold,'ablation':name,**r['metrics'],**r['stats'],'policy_enabled':pol['enabled'],'candidate_mode':mode,'bank_mode':bank});ensure_dir(outdir);write_json(outdir/f'{name}_policy.json',pol);bar.step(name)
    return rows


def partial_cache_from_full(model:EvidenceFoundation,cache:Dict[str,np.ndarray],mask:Sequence[int],device:torch.device)->Dict[str,np.ndarray]:
    zz=[];ee=[];ss=[];N=len(cache['y']);bs=512
    for st in range(0,N,bs):
        ids=np.arange(st,min(N,st+bs));z,e,s=offline_partial_foundation(model,cache,ids,mask,False,device);zz.append(z.cpu().numpy());ee.append(e.cpu().numpy());ss.append(s.cpu().numpy())
    return {'y':np.asarray(cache['y']),'base_logits':np.concatenate(zz),'agg_evidence':np.concatenate(ee),'semantic_pred':np.concatenate(ss)}


def view_subset_analysis_rows(core:PrototypeFalsification,model:EvidenceFoundation,ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,policy:Dict[str,Any],fold:str,bar:AnalysisBar,device:torch.device)->List[Dict[str,Any]]:
    rows=[]
    for name,mask in zip(VIEW_MASK_NAMES,VIEW_MASKS):
        pc=partial_cache_from_full(model,ct,mask,device);r=apply_revision_policy_to_cache(core,pc,pe,ps,labels,policy,device);rows.append({'fold':fold,'condition':name,'n_views':int(np.sum(mask)),**r['metrics'],**r['stats']});bar.step(name)
    return rows


def validate_feature_cache(cache:Dict[str,np.ndarray],ncls:int,name:str)->None:
    req={
        'y':(None,), 'overview':(None,128), 'overview_views':(None,3,128),
        'overview_view_logits':(None,3,ncls), 'pools':(None,3,384),
        'evidence_by_view':(None,3,4,384), 'unc':(None,3), 'occ_logits':(None,3),
        'base_logits':(None,ncls), 'agg_evidence':(None,4,384), 'semantic_pred':(None,10),
    }
    miss=[k for k in req if k not in cache]
    if miss:raise RuntimeError(f'{name}: cache missing keys {miss}')
    N=len(cache['y'])
    if N<=0:raise RuntimeError(f'{name}: empty cache')
    for k,shape in req.items():
        a=np.asarray(cache[k]); exp=(N,)+tuple(shape[1:])
        if a.shape!=exp:raise RuntimeError(f'{name}: cache[{k}] shape={a.shape}, expected={exp}')
        if k!='y' and not np.isfinite(a).all():raise RuntimeError(f'{name}: cache[{k}] contains NaN/Inf')

@torch.no_grad()
def exact_full_revision_from_cache(foundation:EvidenceFoundation,side:nn.Module,pe:np.ndarray,ps:np.ndarray,cache:Dict[str,np.ndarray],device:torch.device)->torch.Tensor:




    N=len(cache['y']);ids=torch.arange(N,device=device);mask=torch.ones(N,3,dtype=torch.float32,device=device)
    z,e,sp=exact_partial_from_cache(foundation,cache,ids,mask,device)
    PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device)
    return side(z,e,sp,PE,PS)['logits'].float()

@torch.no_grad()
def evaluate_selective_cached(foundation:EvidenceFoundation,side:nn.Module,acquisition_policy:Optional[SelectiveEvidenceAcquisitionPolicy],pe:np.ndarray,ps:np.ndarray,cache:Dict[str,np.ndarray],labels:Sequence[str],device:torch.device,policy:str,seed:int)->pd.DataFrame:
    foundation.eval();side.eval()
    if acquisition_policy is not None:acquisition_policy.eval()
    validate_feature_cache(cache,len(labels),f'E/{policy}')
    PE=torch.tensor(pe,dtype=torch.float32,device=device);PS=torch.tensor(ps,dtype=torch.float32,device=device)
    N=len(cache['y']);ids=torch.arange(N,device=device);y=torch.tensor(cache['y'],dtype=torch.long,device=device)
    mask=torch.zeros(N,3,dtype=torch.float32,device=device)
    og=torch.tensor(cache['overview'],dtype=torch.float32,device=device);ovv=torch.tensor(cache['overview_views'],dtype=torch.float32,device=device);ovlog=torch.tensor(cache['overview_view_logits'],dtype=torch.float32,device=device)


    cur=foundation.overview.head(og).float();rng=np.random.RandomState(seed);fixed=[2,0,1];rows=[]
    def record(budget:int,z:torch.Tensor):
        a=z.float().cpu().numpy();m=metrics_from_logits(cache['y'],a,labels);rows.append({'policy':policy,'max_detailed_queries':budget,**m,'detail_gflops_approx':(float('nan') if policy=='Oracle' else budget*videomae_v2s_gflops_per_view()),'oracle_is_diagnostic':policy=='Oracle','selection_uses_falsification':policy not in ('FixedBodyFirst','Random','UncertaintyFirst','EntropyFirst','LearnedUtility_NoContradictionGuidance','Oracle')})
    record(0,cur)
    for budget in range(1,4):
        if policy in ('KOFU_LearnedUtility','LearnedUtility_NoContradictionGuidance'):
            if acquisition_policy is None:raise RuntimeError(f'{policy}: Selective evidence acquisition policy missing')
            score=acquisition_policy(og,ovv,cur.float(),mask).float();choice=score.masked_fill(mask>.5,-1e9).argmax(1)
        elif policy=='UncertaintyFirst':
            pv=torch.softmax(ovlog.float(),-1);choice=(1-pv.max(-1).values).float().masked_fill(mask>.5,-1e9).argmax(1)
        elif policy=='EntropyFirst':
            pv=torch.softmax(ovlog.float(),-1);ent=-(pv*torch.log(pv.clamp_min(1e-8))).sum(-1);choice=ent.float().masked_fill(mask>.5,-1e9).argmax(1)
        elif policy=='FixedBodyFirst':
            choice=torch.tensor([next(v for v in fixed if float(mask[i,v])<.5) for i in range(N)],dtype=torch.long,device=device)
        elif policy=='Random':
            choice=torch.tensor([int(rng.choice(np.flatnonzero(mask[i].detach().cpu().numpy()<.5))) for i in range(N)],dtype=torch.long,device=device)
        elif policy=='Oracle':
            gains=torch.full((N,3),-1e9,dtype=torch.float32,device=device);cur_lp=F.log_softmax(cur.float(),1)[torch.arange(N,device=device),y]
            for v in range(3):
                valid=mask[:,v]<.5
                if not bool(valid.any()):continue
                m2=mask.clone();m2[:,v]=1;z2,e2,s2=exact_partial_from_cache(foundation,cache,ids,m2,device);q=side(z2,e2,s2,PE,PS)['logits'].float();lp=F.log_softmax(q,1)[torch.arange(N,device=device),y];gains[:,v]=torch.where(valid,lp-cur_lp,gains[:,v])
            choice=gains.argmax(1)
        else:raise ValueError(policy)

        if choice.shape!=(N,) or choice.dtype!=torch.long:choice=choice.reshape(N).long()
        if bool(((choice<0)|(choice>2)).any()):raise RuntimeError(f'{policy}: invalid selected view index')
        if bool((mask[torch.arange(N,device=device),choice]>.5).any()):raise RuntimeError(f'{policy}: selected an already queried view at budget={budget}')
        mask[torch.arange(N,device=device),choice]=1
        if not bool((mask.sum(1)==budget).all()):raise RuntimeError(f'{policy}: query-mask cardinality invariant failed at budget={budget}')
        z,e,sp=exact_partial_from_cache(foundation,cache,ids,mask,device);cur=side(z,e,sp,PE,PS)['logits'].float();record(budget,cur)


    exact_full=exact_full_revision_from_cache(foundation,side,pe,ps,cache,device)
    exact_err=float((exact_full-cur.float()).abs().max().cpu())
    if exact_err>2e-5:raise RuntimeError(f'Cached E invariant failed: 3-query != exact cached full, err={exact_err}')
    amp_full=side(torch.tensor(cache['base_logits'],dtype=torch.float32,device=device),torch.tensor(cache['agg_evidence'],dtype=torch.float32,device=device),torch.tensor(cache['semantic_pred'],dtype=torch.float32,device=device),PE,PS)['logits'].float()
    amp_err=float((amp_full-exact_full).abs().max().cpu())
    amp_pred_agree=float((amp_full.argmax(1)==exact_full.argmax(1)).float().mean().cpu())
    for r in rows:
        r['three_query_equals_full_checked']=bool(r['max_detailed_queries']==3)
        r['three_query_exact_max_abs_diff']=exact_err if r['max_detailed_queries']==3 else float('nan')
        r['cached_amp_vs_exact_full_max_abs_diff']=amp_err if r['max_detailed_queries']==3 else float('nan')
        r['cached_amp_vs_exact_full_pred_agreement']=amp_pred_agree if r['max_detailed_queries']==3 else float('nan')
    return pd.DataFrame(rows)


def sensitivity_rows(core:PrototypeFalsification,ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,base_policy:Dict[str,Any],fold:str,bar:AnalysisBar,device:torch.device)->List[Dict[str,Any]]:
    rows=[]
    for strict in (0.75,0.90,1.00,1.10,1.25):
        pol=dict(base_policy)
        if pol.get('enabled',False):
            pol['contradiction_min']=float(pol['contradiction_min'])*strict
            if np.isfinite(float(pol['advantage_min'])):pol['advantage_min']=float(pol['advantage_min'])*strict
            if np.isfinite(float(pol['margin_max'])):pol['margin_max']=float(pol['margin_max'])/strict
        r=apply_revision_policy_to_cache(core,ct,pe,ps,labels,pol,device);rows.append({'fold':fold,'strictness':strict,**r['metrics'],**r['stats']});bar.step(f'x{strict:.2f}')
    return rows


def _summary(df:pd.DataFrame,groups:Sequence[str],metrics:Sequence[str],path:Path):
    if df.empty:return
    cols=[c for c in metrics if c in df.columns]
    df.groupby(list(groups))[cols].agg(['mean','std']).to_csv(path)



def validate_reproduction_dependencies(foundation_root:Path,falsification_root:Path,folds:Sequence[str])->None:
    missing=[]
    protos=['protocol_a']+[f'protocol_c/{p}' for p in C_PROTOCOLS]
    for fold in folds:
        for proto in protos:
            fd=_foundation_checkpoint_dir(foundation_root,proto,fold)
            if not (fd/'best.pt').exists():missing.append(str(fd/'best.pt'))
            vp=_falsification_checkpoint_path(falsification_root,proto,fold)
            if not vp.exists():missing.append(str(vp))
    if missing:raise FileNotFoundError('KOFU prerequisites are incomplete. Missing:\n  '+'\n  '.join(missing))


def unknown_detection_rows(core:PrototypeFalsification,pe:np.ndarray,ps:np.ndarray,known:Dict[str,np.ndarray],unknown:Dict[str,np.ndarray],protocol:str,fold:str)->Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:

    kc=_distance_components_np(core,known,pe,ps);uc=_distance_components_np(core,unknown,pe,ps)
    energy_k=energy_np(known['base_logits']);energy_u=energy_np(unknown['base_logits'])
    msp_k=msp_ood_np(known['base_logits']);msp_u=msp_ood_np(unknown['base_logits'])
    main=[
        {'fold':fold,'protocol':protocol,'method':'Foundation_Energy',**ood_metrics(energy_k,energy_u)},
        {'fold':fold,'protocol':protocol,'method':'EvidenceContradiction',**ood_metrics(kc['evidence_contradiction'],uc['evidence_contradiction'])},
    ]
    abl=[]
    for name,ks,us in [
        ('Foundation_MSP',msp_k,msp_u),
        ('Foundation_Energy',energy_k,energy_u),
        ('SemanticContradiction',kc['semantic_contradiction'],uc['semantic_contradiction']),
        ('FullPrototypeContradiction',kc['full_contradiction'],uc['full_contradiction']),
        ('EvidenceContradiction',kc['evidence_contradiction'],uc['evidence_contradiction']),
    ]:
        abl.append({'fold':fold,'protocol':protocol,'ablation':name,**ood_metrics(ks,us)})
    return main,abl


def revision_analysis_suite(core:PrototypeFalsification,model:EvidenceFoundation,cva:Dict[str,np.ndarray],ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,args:argparse.Namespace,fold:str,outdir:Path,bar:AnalysisBar)->Tuple[Dict[str,Any],List[Dict[str,Any]],List[Dict[str,Any]]]:

    ensure_dir(outdir)
    base=metrics_from_logits(ct['y'],ct['base_logits'],labels)
    rows=[{'fold':fold,'ablation':'Foundation',**base,'revision_rate':0.0,'revision_count':0,'revision_corrected':0,'revision_corrupted':0,'revision_wrong_to_wrong':0,'revision_net_corrections':0}]
    bar.step('Foundation')

    p_con=calibrate_revision_policy(core,model,cva,labels,pe,ps,args,False,False,'top3','clean_hard')
    r_con=apply_revision_policy_to_cache(core,ct,pe,ps,labels,p_con,torch.device(args.device))
    rows.append({'fold':fold,'ablation':'+Contradiction',**r_con['metrics'],**r_con['stats']});write_json(outdir/'01_contradiction.json',p_con);bar.step('+C')

    p_adv=calibrate_revision_policy(core,model,cva,labels,pe,ps,args,True,False,'top3','clean_hard')
    r_adv=apply_revision_policy_to_cache(core,ct,pe,ps,labels,p_adv,torch.device(args.device))
    rows.append({'fold':fold,'ablation':'+Contradiction+PrototypeAdvantage',**r_adv['metrics'],**r_adv['stats']});write_json(outdir/'02_plus_advantage.json',p_adv);bar.step('+A')

    revision_policy_record=calibrate_revision_policy(core,model,cva,labels,pe,ps,args,True,True,'top3','clean_hard')
    revision_result_record=apply_revision_policy_to_cache(core,ct,pe,ps,labels,revision_policy_record,torch.device(args.device))
    rows.append({'fold':fold,'ablation':'KOFU_Contradiction+PrototypeAdvantage+Margin',**revision_result_record['metrics'],**revision_result_record['stats']});write_json(outdir/'conservative_revision_policy.json',revision_policy_record);bar.step('+Margin KOFU')


    design=[]
    for mode,bank,label in [
        ('top2','clean_hard','Candidate_Top2'),
        ('all','clean_hard','Candidate_All'),
        ('top3','clean_only','Calibration_CleanOnly'),
        ('top3','clean_hard','KOFU_Top3_CleanHard'),
    ]:
        pol=revision_policy_record if (mode=='top3' and bank=='clean_hard') else calibrate_revision_policy(core,model,cva,labels,pe,ps,args,True,True,mode,bank)
        rr=revision_result_record if pol is revision_policy_record else apply_revision_policy_to_cache(core,ct,pe,ps,labels,pol,torch.device(args.device))
        design.append({'fold':fold,'ablation':label,**rr['metrics'],**rr['stats'],'candidate_mode':mode,'bank_mode':bank})
        if pol is not revision_policy_record:write_json(outdir/f'{label}.json',pol)
        bar.step(label)
    return {'policy':revision_policy_record,'result':revision_result_record},rows,design


def view_subset_rows(core:PrototypeFalsification,model:EvidenceFoundation,ct:Dict[str,np.ndarray],labels:Sequence[str],pe:np.ndarray,ps:np.ndarray,policy:Dict[str,Any],fold:str,bar:AnalysisBar,device:torch.device)->List[Dict[str,Any]]:
    rows=[]
    for name,mask in zip(VIEW_MASK_NAMES,VIEW_MASKS):
        pc=partial_cache_from_full(model,ct,mask,device)

        bm=metrics_from_logits(pc['y'],pc['base_logits'],labels)
        rr=apply_revision_policy_to_cache(core,pc,pe,ps,labels,policy,device)
        rows.append({'fold':fold,'condition':name,'n_views':int(np.sum(mask)),'method':'Foundation',**bm,'revision_rate':0.0})
        rows.append({'fold':fold,'condition':name,'n_views':int(np.sum(mask)),'method':'KOFU',**rr['metrics'],**rr['stats']})
        bar.step(name)
    return rows


def _save_partial(out:Path,protocol_a_rows,protocol_bc_rows,view_condition_rows,selective_acquisition_rows,revision_analysis_rows,revision_design_rows_list,unknown_signal_rows,revision_sensitivity_rows):
    pd.DataFrame(protocol_a_rows).to_csv(out/'protocol_a_all_folds_partial.csv',index=False)
    pd.DataFrame(protocol_bc_rows).to_csv(out/'protocol_bc_unknown_detection_all_folds_partial.csv',index=False)
    pd.DataFrame(view_condition_rows).to_csv(out/'protocol_d_view_subset_all_folds_partial.csv',index=False)
    pd.DataFrame(selective_acquisition_rows).to_csv(out/'protocol_e_selective_acquisition_all_folds_partial.csv',index=False)
    pd.DataFrame(revision_analysis_rows).to_csv(out/'revision_components_all_folds_partial.csv',index=False)
    pd.DataFrame(revision_design_rows_list).to_csv(out/'revision_design_all_folds_partial.csv',index=False)
    pd.DataFrame(unknown_signal_rows).to_csv(out/'unknown_signal_all_folds_partial.csv',index=False)
    pd.DataFrame(revision_sensitivity_rows).to_csv(out/'revision_sensitivity_all_folds_partial.csv',index=False)










classification_metrics_report = classification_metrics
ReportingExperimentBar = ExperimentBar

def classification_metrics_foundation(y:np.ndarray,pred:np.ndarray,prob:np.ndarray,labels:Sequence[str])->Dict[str,float]:
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); prob=np.asarray(prob)
    safe=labels.index(SAFE_LABEL) if SAFE_LABEL in labels else None; fs=float('nan')
    if safe is not None:
        non=y!=safe; fs=float(((pred==safe)&non).sum()/max(1,int(non.sum())))
    return {'acc':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_acc':float(balanced_accuracy_score(y,pred)),'false_safe':fs,'ece':expected_calibration_error(prob,y)}

class FoundationExperimentBar:

    def __init__(self, clock:GlobalClock, idx:int, name:str, epochs:int, eval_units:int=1):
        self.clock=clock; self.idx=idx; self.name=name; self.epochs=int(epochs); self.eval_units=int(eval_units)
        self.total=max(1,self.epochs+self.eval_units); self.start=time.time(); self.epoch_times=[]; self.eval_times=[]
        self.bar=tqdm(total=self.total, desc=f'[EXP {idx}/{clock.total}] {name}', dynamic_ncols=True, leave=True)
    def epoch(self, ep:int, sec:float, loss:float, val_f1:float, best:float):
        self.epoch_times.append(float(sec)); ae=float(np.mean(self.epoch_times)); av=float(np.mean(self.eval_times)) if self.eval_times else max(1.0,ae*.35)
        rem=max(0,self.epochs-ep)*ae + self.eval_units*av; full=self.epochs*ae+self.eval_units*av
        self.bar.set_postfix_str(f'ep={ep}/{self.epochs} loss={loss:.3f} valF1={val_f1:.3f} best={best:.3f} exp_eta={format_seconds(rem)} all_eta={format_seconds(self.clock.all_eta(rem,full))}', refresh=False)
        self.bar.update(1)
    def eval(self,phase:str,sec:float,remaining:int=0):
        self.eval_times.append(float(sec)); av=float(np.mean(self.eval_times)); rem=max(0,int(remaining))*av; ae=float(np.mean(self.epoch_times)) if self.epoch_times else av
        self.bar.set_postfix_str(f'phase={phase} exp_eta={format_seconds(rem)} all_eta={format_seconds(self.clock.all_eta(rem,self.epochs*ae+self.eval_units*av))}', refresh=False)
        self.bar.update(1)
    def close(self):
        if self.bar.n<self.bar.total: self.bar.update(self.bar.total-self.bar.n)
        self.bar.close(); self.clock.complete(time.time()-self.start)

def load_foundation_for_falsification(df:pd.DataFrame,timelines:Dict[str,EvidenceTimeline],fold:str,labels:Sequence[str],proto_name:str,vmck:Path,args:argparse.Namespace,foundation_root:Path)->Tuple[EvidenceFoundation,Dict[str,np.ndarray],Dict[str,np.ndarray],Path]:
    fd=_foundation_checkpoint_dir(foundation_root,proto_name,fold); bp=fd/'best.pt'
    if not bp.exists():
        raise FileNotFoundError(f'Missing frozen Foundation checkpoint: {bp}. Falsification training does not retrain the Foundation.')
    device=torch.device(args.device); fm=EvidenceFoundation(len(labels),vmck,args.frames); ck=load_model_state(fm,bp,device)
    ck_labels=list(ck.get('labels',labels))
    if ck_labels!=list(labels): raise RuntimeError(f'{proto_name}/{fold}: Foundation labels differ. checkpoint={ck_labels}, expected={list(labels)}')
    for p in fm.parameters(): p.requires_grad_(False)
    fm.eval(); cd=fd/'feature_cache'; tp=cd/'train.npz'; vp=cd/'val.npz'
    if tp.exists() and vp.exists():
        ctr=_load_npz_dict(tp); cva=_load_npz_dict(vp)
    else:
        log(f'{proto_name}/{fold}: frozen feature cache missing; rebuilding it from the frozen checkpoint (no training).')
        tr=filter_df(df,fold,'train',labels); va=filter_df(df,fold,'val',labels)
        exp=f'FoundationReuse/{proto_name}/{fold}';
        ctr=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(tr,labels,timelines,args.frames,args.size,False,reproduction_seed(exp+'/tr',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,reproduction_seed(exp+'/tr',base=args.seed)),device,args.amp,max_batches=0)
        cva=extract_foundation_cache(fm,make_loader(DMDMultiViewDataset(va,labels,timelines,args.frames,args.size,False,reproduction_seed(exp+'/va',base=args.seed),True,args.dry_per_class if args.dry_run else 0),args.eval_batch,0 if args.dry_run else args.workers,False,reproduction_seed(exp+'/va',base=args.seed)),device,args.amp,max_batches=0)
    if not cache_has_all_classes(ctr,len(labels)) or not cache_has_all_classes(cva,len(labels)):
        raise RuntimeError(f'{proto_name}/{fold}: Foundation train/val cache does not cover every known class')
    return fm,ctr,cva,bp


def _paper_stage_args(base: argparse.Namespace, stage: str) -> argparse.Namespace:

    a=argparse.Namespace(**vars(base))
    a.dry_run=False
    a.dry_per_class=0
    a.overwrite_annotation_cache=False
    if stage=='foundation':

        a.falsification_epochs=25; a.falsification_lr=2e-3; a.max_revision=3.0
        a.lambda_keep=.5; a.lambda_false_safe=.25; a.lambda_alpha=.01
        a.false_safe_margin=.5; a.val_tolerance=.001; a.fs_tolerance=.002
        a.selective_epochs=20; a.acquisition_epochs=15; a.e_batch=1; a.occlusion_threshold=.25
    elif stage=='falsification':

        a.falsification_epochs=35; a.falsification_lr=2e-3; a.max_revision=3.0
        a.lambda_hard=1.0; a.lambda_keep=.75; a.lambda_false_safe=.25; a.lambda_alpha=.0025
        a.false_safe_margin=.5; a.val_tolerance=.001; a.fs_tolerance=.002
        a.acquisition_epochs=20; a.e_batch=1; a.occlusion_threshold=.25
    elif stage=='reporting':

        a.max_revision=3.0; a.val_tolerance=.001; a.fs_tolerance=.002; a.revision_flip_eps=.05; a.acquisition_epochs=20
    else:
        raise ValueError(stage)
    return a


def _parser_kofu() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(
        description='KOFU: evidence-audited multi-view driver monitoring reproduction pipeline for Protocols A/B/C/D/E.'
    )
    p.add_argument('--root',default='../dataset/DMD_distraction_extracted')
    p.add_argument('--split_dir',default='dmd_evidms_final_split_v1')
    p.add_argument('--out_dir',default='')
    p.add_argument('--checkpoint',default='checkpoints/vit_s_k710_dl_from_giant.pth')
    p.add_argument('--no_auto_download',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--eval_batch',type=int,default=2)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--lr_backbone',type=float,default=2e-5)
    p.add_argument('--lr_head',type=float,default=2e-4)
    p.add_argument('--weight_decay',type=float,default=.05)
    p.add_argument('--grad_clip',type=float,default=5.)
    p.add_argument('--lambda_evidence',type=float,default=.25)
    p.add_argument('--lambda_occlusion',type=float,default=.05)
    p.add_argument('--lambda_overview',type=float,default=.15)
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--amp',action='store_true',default=True)
    p.add_argument('--no_amp',dest='amp',action='store_false')
    p.add_argument('--resume',action='store_true',default=True)
    p.add_argument('--no_resume',dest='resume',action='store_false')
    p.add_argument('--overwrite',action='store_true')
    return p


def _extract_evaluation_cache(model: EvidenceFoundation, dfp: pd.DataFrame, labels: Sequence[str], timelines: Dict[str,EvidenceTimeline], args: argparse.Namespace, seed: int) -> Dict[str,np.ndarray]:

    ds=DMDMultiViewDataset(dfp,labels,timelines,args.frames,args.size,False,seed,True,0)
    ld=make_loader(ds,args.eval_batch,args.workers,False,seed)
    return extract_foundation_cache(model,ld,torch.device(args.device),args.amp,max_batches=0)


def _write_result_bundle(out: Path, script_path: Path) -> Path:
    zp=out.with_name(out.name+'_results.zip')
    if zp.exists(): zp.unlink()
    keep_names={
        'protocol_a_all_folds.csv','protocol_a_summary_mean_std.csv',
        'protocol_bc_unknown_detection_all_folds.csv','protocol_bc_unknown_detection_summary_mean_std.csv',
        'protocol_d_view_subset_all_folds.csv','protocol_d_view_subset_summary_mean_std.csv',
        'protocol_e_selective_acquisition_all_folds.csv','protocol_e_selective_acquisition_summary_mean_std.csv',
    }
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(script_path,arcname='CODE/'+script_path.name)
        for name in sorted(keep_names):
            p=out/name
            if p.is_file(): z.write(p,arcname=name)
        rp=out/'conservative_revision'
        if rp.exists():
            for p in sorted(rp.rglob('conservative_revision_policy.json')):
                z.write(p,arcname=str(p.relative_to(out)))
    return zp


def main():
    global classification_metrics, ExperimentBar

    args=_parser_kofu().parse_args()
    seed_everything(args.seed)

    root=Path(args.root).expanduser().resolve()
    split=Path(args.split_dir).expanduser().resolve()
    out=Path(args.out_dir).expanduser() if args.out_dir else Path(f'runs/kofu_reproduction_seed{args.seed}')
    foundation_root=out/'foundation'
    falsification_root=out/'falsification'
    ensure_dir(out); ensure_dir(foundation_root); ensure_dir(falsification_root)


    df=load_interval_csv(split,root)
    timelines=build_annotation_cache(df,foundation_root/'annotation_cache.pkl',False)
    vmck=ensure_videomae_checkpoint(Path(args.checkpoint).expanduser(),not args.no_auto_download)
    folds=[f'fold{i}' for i in range(5)]
    device=torch.device(args.device)






    classification_metrics=classification_metrics_foundation
    ExperimentBar=FoundationExperimentBar
    foundation_args=_paper_stage_args(args,'foundation')
    foundation_clock=GlobalClock(40); foundation_index=1
    for fold in folds:
        fm,side,pe,ps,ctr,cva,sp,foundation_index=train_foundation_and_falsification(df,timelines,fold,MAIN11,'protocol_a',vmck,foundation_args,foundation_root,foundation_clock,foundation_index)
        del fm,side,pe,ps,ctr,cva,sp; gc.collect(); torch.cuda.empty_cache()
        for proto,spec in C_PROTOCOLS.items():
            known=list(spec['known'])
            fm,side,pe,ps,ctr,cva,sp,foundation_index=train_foundation_and_falsification(df,timelines,fold,known,'protocol_c/'+proto,vmck,foundation_args,foundation_root,foundation_clock,foundation_index)
            del fm,side,pe,ps,ctr,cva,sp; gc.collect(); torch.cuda.empty_cache()




    falsification_args=_paper_stage_args(args,'falsification')
    falsification_clock=GlobalClock(20); falsification_index=1
    for fold in folds:
        fm,ctr,cva,_=load_foundation_for_falsification(df,timelines,fold,MAIN11,'protocol_a',vmck,falsification_args,foundation_root)
        pe,ps=build_prototypes(ctr,len(MAIN11))
        sp=train_falsification(fm,ctr,cva,MAIN11,pe,ps,falsification_root/'protocol_a'/'kofu'/fold,falsification_args,falsification_clock,falsification_index,f'A/{fold}/Falsification'); falsification_index+=1
        del fm,ctr,cva,pe,ps,sp; gc.collect(); torch.cuda.empty_cache()
        for proto,spec in C_PROTOCOLS.items():
            known=list(spec['known'])
            fm,ctr,cva,_=load_foundation_for_falsification(df,timelines,fold,known,'protocol_c/'+proto,vmck,falsification_args,foundation_root)
            pe,ps=build_prototypes(ctr,len(known))
            sp=train_falsification(fm,ctr,cva,known,pe,ps,falsification_root/'protocol_c'/proto/'kofu'/fold,falsification_args,falsification_clock,falsification_index,f'{proto}/{fold}/Falsification'); falsification_index+=1
            del fm,ctr,cva,pe,ps,sp; gc.collect(); torch.cuda.empty_cache()




    classification_metrics=classification_metrics_report
    ExperimentBar=ReportingExperimentBar
    reporting_args=_paper_stage_args(args,'reporting')
    acquisition_clock=GlobalClock(15); acquisition_index=1
    protocol_a_rows=[]; protocol_bc_rows=[]; protocol_d_rows=[]; protocol_e_rows=[]

    for fold in folds:
        fm,ctr,cva,_=load_frozen_foundation(df,timelines,fold,MAIN11,'protocol_a',vmck,reporting_args,foundation_root)
        core,pe,ps,_,_=load_falsification_core(falsification_root,'protocol_a',fold,len(MAIN11),device,reporting_args.max_revision)

        te=filter_df(df,fold,'test',MAIN11)
        ct=_extract_evaluation_cache(fm,te,MAIN11,timelines,reporting_args,reproduction_seed(f'EvaluationCache/{fold}/A-test-cache',base=reporting_args.seed))

        revision_policy=calibrate_revision_policy(core,fm,cva,MAIN11,pe,ps,reporting_args,True,True,'top3','clean_hard')
        revision_result=apply_revision_policy_to_cache(core,ct,pe,ps,MAIN11,revision_policy,device)
        ensure_dir(out/'conservative_revision'/fold); write_json(out/'conservative_revision'/fold/'conservative_revision_policy.json',revision_policy)
        bm=metrics_from_logits(ct['y'],ct['base_logits'],MAIN11)
        protocol_a_rows.extend([
            {'fold':fold,'method':'Foundation',**bm,'revision_rate':0.0,'revision_count':0,'revision_corrected':0,'revision_corrupted':0,'revision_net_corrections':0},
            {'fold':fold,'method':'KOFU',**revision_result['metrics'],**revision_result['stats']},
        ])

        bu=filter_df(df,fold,'test',NONCANONICAL3)
        cb=_extract_evaluation_cache(fm,bu,NONCANONICAL3,timelines,reporting_args,reproduction_seed(f'ProtocolB/{fold}',base=reporting_args.seed))
        mr,_=unknown_detection_rows(core,pe,ps,ct,cb,'B_noncanonical3',fold); protocol_bc_rows.extend(mr)

        for proto,spec in C_PROTOCOLS.items():
            known=list(spec['known']); unknown=list(spec['unknown'])
            cfm,ctr2,cva2,_=load_frozen_foundation(df,timelines,fold,known,'protocol_c/'+proto,vmck,reporting_args,foundation_root)
            ccore,cpe,cps,_,_=load_falsification_core(falsification_root,'protocol_c/'+proto,fold,len(known),device,reporting_args.max_revision)
            tk=filter_df(df,fold,'test',known); tu=filter_df(df,fold,'test',unknown)
            sd=reproduction_seed(f'ProtocolC/{proto}/{fold}',base=reporting_args.seed)
            kc=_extract_evaluation_cache(cfm,tk,known,timelines,reporting_args,sd)
            uc=_extract_evaluation_cache(cfm,tu,unknown,timelines,reporting_args,sd+1)
            mr,_=unknown_detection_rows(ccore,cpe,cps,kc,uc,proto,fold); protocol_bc_rows.extend(mr)
            del cfm,ccore,ctr2,cva2,cpe,cps,kc,uc; gc.collect(); torch.cuda.empty_cache()


        for name,mask in zip(VIEW_MASK_NAMES,VIEW_MASKS):
            pc=partial_cache_from_full(fm,ct,mask,device)
            bdm=metrics_from_logits(pc['y'],pc['base_logits'],MAIN11)
            rr=apply_revision_policy_to_cache(core,pc,pe,ps,MAIN11,revision_policy,device)
            protocol_d_rows.append({'fold':fold,'condition':name,'n_views':int(np.sum(mask)),'method':'Foundation',**bdm,'revision_rate':0.0})
            protocol_d_rows.append({'fold':fold,'condition':name,'n_views':int(np.sum(mask)),'method':'KOFU',**rr['metrics'],**rr['stats']})


        conservative_revision=ConservativeRevision(core,revision_policy).to(device).eval()
        acquisition_checkpoint=train_selective_acquisition_policy(fm,conservative_revision,pe,ps,ctr,cva,out/'protocol_e'/fold/'selective_evidence_acquisition',reporting_args,acquisition_clock,acquisition_index,f'SelectiveAcquisition/{fold}/train'); acquisition_index+=1
        acquisition_policy=load_selective_acquisition_policy(acquisition_checkpoint,len(MAIN11),device)
        ed=evaluate_selective_cached(fm,conservative_revision,acquisition_policy,pe,ps,ct,MAIN11,device,'KOFU_LearnedUtility',reproduction_seed(f'ProtocolE/{fold}/KOFU_LearnedUtility',base=reporting_args.seed))
        ed.insert(0,'fold',fold); ed['method']='KOFU learned utility'; protocol_e_rows.extend(ed.to_dict('records'))

        del fm,core,ctr,cva,pe,ps,ct,cb,conservative_revision,acquisition_policy; gc.collect(); torch.cuda.empty_cache()

    A=pd.DataFrame(protocol_a_rows); O=pd.DataFrame(protocol_bc_rows); D=pd.DataFrame(protocol_d_rows); E=pd.DataFrame(protocol_e_rows)
    A.to_csv(out/'protocol_a_all_folds.csv',index=False)
    O.to_csv(out/'protocol_bc_unknown_detection_all_folds.csv',index=False)
    D.to_csv(out/'protocol_d_view_subset_all_folds.csv',index=False)
    E.to_csv(out/'protocol_e_selective_acquisition_all_folds.csv',index=False)
    _summary(A,['method'],['acc','macro_f1','balanced_acc','false_safe','ece','revision_rate','revision_corrected','revision_corrupted','revision_net_corrections'],out/'protocol_a_summary_mean_std.csv')
    _summary(O,['protocol','method'],['auroc','aupr','fpr95'],out/'protocol_bc_unknown_detection_summary_mean_std.csv')
    _summary(D,['condition','method'],['acc','macro_f1','balanced_acc','false_safe','ece','revision_rate'],out/'protocol_d_view_subset_summary_mean_std.csv')
    _summary(E,['method','max_detailed_queries'],['acc','macro_f1','balanced_acc','false_safe','ece','detail_gflops_approx'],out/'protocol_e_selective_acquisition_summary_mean_std.csv')

    z=_write_result_bundle(out,Path(__file__).resolve())
    print(f'\nDONE. Results bundle: {z}',flush=True)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('\nInterrupted by user.',file=sys.stderr); sys.exit(130)
    except Exception:
        traceback.print_exc(); sys.exit(1)
