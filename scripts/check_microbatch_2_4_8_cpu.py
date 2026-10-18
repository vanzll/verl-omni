"""Native temporal loop and actual ART/NFT losses on fixed CPU inputs, MB2/4/8."""
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import json,copy,math,time
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl_omni.workers.engine.fsdp import diffusers_impl as impl
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss,OmniNFTLoss
R=Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
g=torch.Generator().manual_seed(173)
features=torch.randn(8,20,6,2,generator=g,dtype=torch.float64)
clean=torch.randn(8,6,2,generator=g,dtype=torch.float64)
guidance=torch.randn(8,6,2,generator=g,dtype=torch.float64);guidance[0]=0
prob=torch.linspace(.1,.9,8,dtype=torch.float64)
initial=torch.tensor([[.12,.03],[-.04,.15]],dtype=torch.float64)
reports={}
for variant,timesteps in [('A0',19),('N0',20)]:
 cfg=OmegaConf.load(R/f'preflight/configs/{variant}/resolved.yaml').actor_rollout_ref.actor
 rows=[]
 for micro in (2,4,8):
  module=torch.nn.Linear(2,2,bias=False,dtype=torch.float64);module.weight.data.copy_(initial)
  opt=torch.optim.AdamW(module.parameters(),lr=3e-5,weight_decay=1e-4,foreach=False)
  calls=[];losses=[];accums=[]
  data=TensorDict(dict(features=features[:,:timesteps].clone(),train_timesteps=torch.linspace(900,100,timesteps)[None].expand(8,-1),clean=clean.clone(),guidance=guidance.clone(),prob=prob.clone()),batch_size=[8])
  tu.assign_non_tensor(data,micro_batch_size_per_gpu=micro,use_dynamic_bsz=False,global_token_num=None)
  engine=SimpleNamespace(ulysses_sequence_parallel_size=1,get_data_parallel_group=lambda:None)
  def forward_step(batch,loss_function,forward_only,step):
   x=batch['features'][:,step];pred=module(x);old=x*.05;t=batch['train_timesteps'][:,step].double().view(-1,1,1)/1000
   if variant=='A0':
    loss,_=DiffusionARTLoss.compute_loss(forward_prediction=old+pred*1e-4,old_prediction=old,t_expanded=t,art_guidance=batch['guidance'],sigma_next=(t.flatten()-.03).clamp_min(0),config=cfg,video_seq_len=torch.full((len(batch),),4))
   else:
    kw={}
    for label,sl in [('video',slice(0,4)),('audio',slice(4,None))]:
     kw.update({f'{label}_forward_prediction':pred[:,sl],f'{label}_old_prediction':old[:,sl],f'{label}_ref_forward_prediction':torch.zeros_like(pred[:,sl]),f'{label}_x0':batch['clean'][:,sl],f'{label}_xt':x[:,sl],f'{label}_t_expanded':t,f'{label}_reward_prob':batch['prob']})
    loss,_=OmniNFTLoss.compute_loss(config=cfg,**kw)
   accum=tu.get_non_tensor_data(batch,'gradient_accumulation_steps',default=None);scaled=loss/accum
   calls.append((len(batch),step));accums.append(accum);losses.append(float(scaled.detach()))
   return scaled,{'loss':float(scaled.detach()),'metrics':{},'model_output':{}}
  engine.forward_step=forward_step;engine.postprocess_batch_func=lambda **kwargs:kwargs
  with patch.object(impl,'get_device_id',lambda:'cpu'):
   impl.DiffusersFSDPEngine._run_forward_backward_batch(engine,data,None,False,timesteps_key='train_timesteps')
  grad=module.weight.grad.clone();norm=torch.nn.utils.clip_grad_norm_(module.parameters(),1.0);opt.step()
  assert len(calls)==timesteps*(8//micro) and set(accums)=={timesteps*(8//micro)}
  rows.append(dict(micro=micro,trajectory_accumulation=8//micro,temporal_accumulation=timesteps*(8//micro),backward_calls=len(calls),loss=sum(losses),gradient=grad,parameter_update=module.weight.detach()-initial,grad_norm=float(norm),optimizer_step=float(opt.state[module.weight]['step'])))
 reference=copy.deepcopy(rows[0])
 for row in rows:
  assert math.isclose(row['loss'],reference['loss'],rel_tol=2e-6,abs_tol=1e-8)
  for key in ['gradient','parameter_update']:
   torch.testing.assert_close(row[key],reference[key],rtol=2e-5,atol=1e-8)
   diff=row[key]-reference[key];row[key+'_max_abs_difference']=float(diff.abs().max());row[key+'_relative_l2']=float(diff.norm()/reference[key].norm().clamp_min(1e-30))
  assert row['optimizer_step']==1 and 16*row['micro']*row['trajectory_accumulation']==128
  del row['gradient'],row['parameter_update']
 reports[variant]=rows
result=dict(status='passed_native_temporal_loop_actual_ART_NFT_loss_microbatch_contract',cpu_only=True,full_LTX_GPU_verified=False,world16_optimizer_batch=128,rows=reports,scope='Fixed supplied inputs, actual native temporal loop and actual losses; tiny linear model on CPU. Does not claim full-model BF16 bitwise equality.',observed_unix=time.time())
(R/'logs/microbatch_2_4_8_cpu_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
