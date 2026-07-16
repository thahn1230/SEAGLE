import json, os, sys
sys.path.insert(0,'src'); sys.path.insert(0,'third_party/EAGLE')
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import ConcatSelectiveDraftAdapter
from eagle_spinquant.embedding_scale_reparameterization import (
    capture_projection_inputs, alpha_objective, _shared_a4_fake_quant)
from eagle_spinquant import fake_w4a4_draft as fq
RD, tgt = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False)
dev='cuda:0'
cfg=experiment.load_config(None); paths=experiment.resolve_paths(cfg)
rr=cfg.get('paths',{}).get('rotations_root')
rot,quant,fhm = ('none','none','identity') if tgt=='fp16' else ('full','w4a4','gamma_R1')
model,stash,_=study.build_study_target(paths['target_path'],paths['draft_path'],
    cfg['model']['target'],rot,'learned_chat_w4a4kv16',quant,0,device=dev,rotations_root=rr)
if rot=='none':
    R=torch.load(study.r_bin_path('learned_chat_w4a4kv16',0,paths['target_path'],rr),map_location='cpu',weights_only=False)
    stash['R1']=R['R1'].clone()
    stash['gamma_f']=model.base_model.model.norm.weight.detach().float().cpu().clone()
    stash['lm_head_weight']=model.base_model.lm_head.weight.detach().float().cpu().clone()
tok=eagle_bridge.get_tokenizer(model)
from eagle.model.choices import mc_sim_7b_63 as tf_
tree=[list(p) for p in tf_]
study.set_draft_tree(model,tree,dev)
sys.path.insert(0,'scripts')
from calibrate_embedding_alpha import calib_prompts
build_prompt=eagle_bridge.PROMPT_BUILDERS[cfg.get('model',{}).get('chat_template','llama2')]
ids=[p.to(dev) for p in calib_prompts(tok,build_prompt)]
ad=ConcatSelectiveDraftAdapter(model,stash,dev,torch.float16,variant='folded',first_hidden_mode=fhm,trace=False)
ad.install()
Z=capture_projection_inputs(ad,model,ids,tree,max_rows=4096)
W_f=ad.split.projection_first_preR.weight.detach().float().cpu()
W_r=ad.split.projection_recurrent_preR.weight.detach().float().cpu()
bias=ad.split.projection_first_preR.bias.detach().float().cpu()
ad.uninstall()
alpha=json.load(open(os.path.join(RD,'alpha_selected.json')))[tgt]['alpha']
rows=[]
# P1 shared (alpha=1), P3 alpha
for name,a in (('P1_shared',1.0),('P3_alpha',alpha)):
    m=alpha_objective(W_f,W_r,bias,Z.get('first'),Z.get('recurrent'),a)
    for path,v in m.items():
        rows.append(dict(target=tgt,variant=name,alpha=a,path=path,**{k:round(x,6) for k,x in v.items()}))
# P2 separate scales (weights quantized identically, acts per-slice)
D=W_f.shape[1]//2
for path,W,Zc in (('first',W_f,Z.get('first')),('recurrent',W_r,Z.get('recurrent'))):
    if Zc is None: continue
    y_ref=Zc@W.t()+bias
    Wq=fq._weight_fake_quant(W.clone(),4).float()
    qe=_shared_a4_fake_quant(Zc[:,:D]); qh=_shared_a4_fake_quant(Zc[:,D:])
    y=torch.cat([qe,qh],-1)@Wq.t()+bias
    err=y-y_ref
    rows.append(dict(target=tgt,variant='P2_separate',alpha=1.0,path=path,
        nmse=round(float((err**2).sum()/(y_ref**2).sum()),6),
        rel_l2=round(float(err.norm()/y_ref.norm()),6),
        cos=round(float(torch.nn.functional.cosine_similarity(y.flatten(),y_ref.flatten(),0)),6),
        chan_max=round(float(err.abs().amax(0).max()),5), e_rail_frac=None, w_e_over_h_absmax=None))
out=os.path.join(RD,f'projection_scale_reconstruction__{tgt}.csv')
logging_utils.write_csv(out,rows)
print('[recon]',tgt,'written', flush=True)
