"""
SevSeg-YOLO 工业缺陷检测训练工具 - Gradio Web界面
功能：环境检查、数据转换(含类别score分析)、模型训练、模型转换、批量推理+样本预览

修复:
- [Fix-1] 删除Guided Filter环境检测(MaskGeneratorV2不使用)
- [Fix-2] 全面参数保存/恢复: 数据转换、训练、转换、推理所有Tab
- [Fix-2] 训练历史(loss/mAP/score曲线)、日志恢复
- [Fix-3] 推理完成后自动刷新样本预览
- [Fix-4] TensorRT转换增加min_batch/max_batch参数
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import os, sys, threading, time, json, argparse, cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import gradio as gr
import numpy as np

SEVSEG_DIR = str(Path(__file__).parent)
sys.path.insert(0, SEVSEG_DIR)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════ DL-Hub ═══════
def _init_adapter():
    try:
        p = Path(__file__).parent.parent / 'dlhub_project' / 'dlhub'
        if p.exists(): sys.path.insert(0, str(p.parent))
        from dlhub.app_adapters.base_adapter import get_adapter; return get_adapter(default_port=7861)
    except: return None
def _init_params():
    try:
        from dlhub_params import get_dlhub_params; return get_dlhub_params()
    except: return None
dlhub_adapter = _init_adapter(); dlhub_params = _init_params()
def get_output_dir(d='./output'):
    if dlhub_params and dlhub_params.is_dlhub_mode: return dlhub_params.get_output_dir()
    if dlhub_adapter and dlhub_adapter.is_dlhub_mode: return dlhub_adapter.get_output_dir()
    return Path(d)
def get_saved_param(k, d=None): return dlhub_params.get(k, d) if dlhub_params else d

def _get_section(name, default=None):
    """[Fix-2] 安全获取section"""
    if default is None: default = {}
    if not dlhub_params: return default
    try: return dlhub_params.get_section(name) or default
    except: return dlhub_params.get(name, default) or default

def _save_section(name, data):
    """[Fix-2] 保存单个section (不覆盖其他section)"""
    if not dlhub_params: return
    try:
        cp = dlhub_params.get_all()
        cp[name] = data
        dlhub_params.save(cp)
    except Exception as e:
        print(f"[DL-Hub] 保存{name}失败: {e}")

CUSTOM_CSS = """
.gradio-container{max-width:100%!important;padding:0 20px!important;margin:0!important}
.log-box textarea{font-family:Consolas,Monaco,monospace!important;font-size:13px!important;background:#1a1a2e!important;color:#eee!important;border-radius:8px!important}
.primary-btn{background:linear-gradient(135deg,#667eea,#764ba2)!important;border:none!important;font-weight:bold!important;font-size:1.1em!important;padding:12px 24px!important}
.stop-btn{background:linear-gradient(135deg,#f43f5e,#ec4899)!important;border:none!important}
"""

# ═══════ 全局状态 ═══════
class TS:
    def __init__(self): self.reset()
    def reset(self):
        self.running=False;self.stop=False;self.epoch=0;self.total=0
        self.box_l=[];self.cls_l=[];self.score_l=[];self.map50=[];self.map95=[];self.mae=[]
        self.logs=[];self.best_map=0.;self.best_ep=0;self.out="";self.t0=None
    def log(self,m):
        self.logs.append(m)
        if len(self.logs)>200: self.logs=self.logs[-200:]
class CS:
    def __init__(self): self.running=False;self.logs=[]
    def log(self,m):
        self.logs.append(m)
        if len(self.logs)>200: self.logs=self.logs[-200:]
    def reset(self): self.running=False;self.logs=[]
class IS:
    def __init__(self): self.reset()
    def reset(self): self.running=False;self.total=0;self.done=0;self.results=[]

ts=TS();cs=CS();inf=IS()

# ═══════ 工具 ═══════
def check_env():
    try:
        from utils.env_validator import validate_environment
        _,m,_=validate_environment(); return m
    except Exception as e: return f"❌ {e}"
def get_gpus():
    try:
        from utils.env_validator import get_gpu_choices; return get_gpu_choices()
    except: return ["CPU"]
def fmt_t(s):
    if s<60: return f"{s:.0f}秒"
    elif s<3600: return f"{s/60:.1f}分"
    return f"{s/3600:.1f}时"
def scan_folders():
    b=get_output_dir()
    if not b.exists(): return gr.update(choices=[],value=None)
    fs=sorted([d.name for d in b.iterdir() if d.is_dir() and any(d.rglob('*.pt'))],reverse=True)
    return gr.update(choices=fs,value=fs[0] if fs else None)
def scan_models(n):
    if not n: return gr.update(choices=[],value=None)
    b=get_output_dir()/n
    if not b.exists(): return gr.update(choices=[],value=None)
    ms=sorted([str(m) for m in b.rglob('*.pt')],reverse=True)
    return gr.update(choices=ms,value=ms[0] if ms else None)

# ═══════ Tab1: 数据转换 ═══════
def validate_paths(i,j):
    try:
        from utils.data_validator import validate_images_jsons; return validate_images_jsons(i,j)
    except Exception as e: return f"❌ {e}"

def do_convert(img,jsn,out,vr,seed):
    if not img or not jsn or not out: return "❌ 请填写所有路径","",None,None,gr.update()
    _save_section('convert', {'images_dir':img.strip(),'jsons_dir':jsn.strip(),'output_dir':out.strip(),'val_ratio':float(vr),'seed':int(seed)})
    try:
        from sevseg_yolo.convert import convert_dataset
        r=convert_dataset(images_dir=img.strip(),jsons_dir=jsn.strip(),output_dir=out.strip(),val_ratio=float(vr),seed=int(seed))
        if not r: return "❌ 转换失败","",None,None,gr.update()
        msg=f"✅ 完成! 训练:{r['train_count']}张 验证:{r['val_count']}张 类别:{r['nc']}\n{r['yaml_path']}"
        f1,f2,cls_update=_data_figs(out.strip(),r['nc'])
        return msg,r['yaml_path'],f1,f2,cls_update
    except Exception as e:
        import traceback; return f"❌ {e}\n{traceback.format_exc()}","",None,None,gr.update()

def _data_figs(dd,nc):
    from collections import Counter
    dp=Path(dd); cls_all=[]; scores_all=[]; _cls_scores_cache.clear()
    for sp in ['train','val']:
        ld=dp/'labels'/sp
        if not ld.exists(): continue
        for t in ld.glob('*.txt'):
            try:
                for ln in open(t):
                    p=ln.strip().split()
                    if len(p)>=5:
                        cid=int(p[0]); cls_all.append(cid)
                        if len(p)>=6:
                            v=float(p[5])
                            if v>=0: scores_all.append(v); _cls_scores_cache[cid].append(v)
            except: pass
    _cls_scores_cache['_nc']=nc
    f1,a1=plt.subplots(figsize=(6,4),dpi=100)
    if cls_all:
        cnt=Counter(cls_all); labs=[f"class_{i}" for i in range(nc)]; vals=[cnt.get(i,0) for i in range(nc)]
        bars=a1.bar(labs,vals,color=plt.cm.Set3(np.linspace(0,1,max(nc,1))),edgecolor='gray')
        a1.set_title('类别分布',fontweight='bold',fontsize=13); a1.set_ylabel('数量')
        for b,v in zip(bars,vals): a1.text(b.get_x()+b.get_width()/2,b.get_height()+.5,str(v),ha='center',fontsize=9)
    else: a1.text(.5,.5,'无标注',ha='center',va='center',transform=a1.transAxes)
    f1.tight_layout()
    f2,a2=plt.subplots(figsize=(6,4),dpi=100)
    if scores_all:
        a2.hist(scores_all,bins=20,range=(0,10),color='steelblue',edgecolor='black',alpha=.8)
        a2.axvline(np.mean(scores_all),color='red',ls='--',label=f'均值={np.mean(scores_all):.1f}')
        a2.set_title(f'全部Severity分布 (n={len(scores_all)})',fontweight='bold',fontsize=13)
        a2.set_xlabel('Score [0-10]'); a2.set_ylabel('数量'); a2.legend()
    else: a2.text(.5,.5,'无Severity标注',ha='center',va='center',transform=a2.transAxes)
    f2.tight_layout()
    cls_choices=["全部"]+[f"class_{i}" for i in range(nc)]
    return f1,f2,gr.update(choices=cls_choices,value="全部")

from collections import defaultdict
_cls_scores_cache = defaultdict(list)

def show_class_scores(cls_name):
    f,a=plt.subplots(figsize=(6,4),dpi=100)
    nc=_cls_scores_cache.get('_nc',0)
    if cls_name=="全部":
        all_s=[]
        for i in range(nc): all_s.extend(_cls_scores_cache.get(i,[]))
        if all_s:
            a.hist(all_s,bins=20,range=(0,10),color='steelblue',edgecolor='black',alpha=.8)
            a.axvline(np.mean(all_s),color='red',ls='--',label=f'均值={np.mean(all_s):.1f}')
            a.set_title(f'全部类别 Severity (n={len(all_s)})',fontweight='bold'); a.legend()
        else: a.text(.5,.5,'无数据',ha='center',va='center',transform=a.transAxes)
    else:
        try: cid=int(cls_name.replace('class_',''))
        except: cid=-1
        scores=_cls_scores_cache.get(cid,[])
        if scores:
            a.hist(scores,bins=20,range=(0,10),color=plt.cm.Set2(cid/max(nc,1)),edgecolor='black',alpha=.8)
            a.axvline(np.mean(scores),color='red',ls='--',label=f'均值={np.mean(scores):.1f}')
            a.set_title(f'{cls_name} Severity (n={len(scores)})',fontweight='bold'); a.legend()
        else: a.text(.5,.5,f'{cls_name} 无Severity数据',ha='center',va='center',transform=a.transAxes)
    a.set_xlabel('Score [0-10]'); a.set_ylabel('数量'); a.set_xlim(-0.5,10.5)
    f.tight_layout(); return f

# ═══════ Tab2: 训练 ═══════
def val_yaml(p):
    if not p or not p.strip(): return "⏳ 请输入data.yaml"
    try:
        from utils.data_validator import validate_data_yaml; return validate_data_yaml(p.strip()).message
    except Exception as e: return f"❌ {e}"
def model_info(s):
    if not s: return ""
    try:
        from config.model_registry import get_model_display_info; return get_model_display_info(s)
    except Exception as e: return f"❌ {e}"
def pt_status(s):
    if not s: return ""
    try:
        from models.model_factory import check_pretrained_status; return check_pretrained_status(s)
    except Exception as e: return f"❌ {e}"

def start_train(dy,ms,ep,bs,lr,opt,isz,pat,mos,flr,fud,hh,hs,hv,tr,sc,clr,sp,gpu):
    if cs.running: return "⚠️ 转换中",""
    if ts.running: return "⚠️ 训练中","\n".join(ts.logs[-50:])
    if not dy or not ms: return "❌ 填写data.yaml和模型",""
    ts.reset();ts.running=True;ts.total=int(ep);ts.t0=time.time()
    threading.Thread(target=_train,args=(dy.strip(),ms,int(ep),int(bs),float(lr),opt,int(isz),int(pat),float(mos),float(flr),float(fud),float(hh),float(hs),float(hv),float(tr),float(sc),clr,int(sp),gpu),daemon=True).start()
    return "🚀 已启动",""

def _train(dy,ms,ep,bs,lr,opt,isz,pat,mos,flr,fud,hh,hs,hv,tr,sc,clr,sp,gpu):
    try:
        from config.model_registry import get_model_config
        from engine.trainer import SevSegTrainer,TrainingCallback
        from utils.env_validator import parse_gpu_choice
        from models.model_factory import get_model_yaml_path
        ts.log("━"*50);ts.log("🚀 SevSeg-YOLO训练");ts.log("━"*50)
        dev,_=parse_gpu_choice(gpu);ts.log(f"🖥️ {'cuda:'+dev if dev!='cpu' else 'cpu'}")
        cfg=get_model_config(ms)
        if not cfg: ts.log(f"❌ 配置不存在:{ms}");ts.running=False;return
        yp=get_model_yaml_path(ms)
        if not yp: ts.log("❌ YAML不存在");ts.running=False;return
        ts.log(f"🧠 {cfg['name']} ({cfg['params']}M)")
        od=get_output_dir()/f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        od.mkdir(parents=True,exist_ok=True);ts.out=str(od)
        if dlhub_params:
            dlhub_params.clear_history(auto_save=False);dlhub_params.clear_logs('training',auto_save=False)
            cp=dlhub_params.get_all()
            cp['data']={'data_yaml':dy}
            cp['model']={'scale':ms}
            cp['training']={'epochs':ep,'batch_size':bs,'learning_rate':lr,'optimizer':opt,'img_size':isz,'patience':pat,'save_period':sp}
            cp['augmentation']={'mosaic':mos,'fliplr':flr,'flipud':fud,'hsv_h':hh,'hsv_s':hs,'hsv_v':hv,'translate':tr,'scale':sc,'cos_lr':clr}
            cp['device']=gpu
            dlhub_params.save(cp)
            dlhub_params.save_history({'total_epochs':ep,'current_epoch':0,'output_dir':str(od),'completed':False},auto_save=True)
        def on_ep(epoch,m):
            ts.epoch=epoch;ts.box_l.append(float(m.get('box_loss',0)));ts.cls_l.append(float(m.get('cls_loss',0)))
            ts.score_l.append(float(m.get('score_loss',0)));ts.map50.append(float(m.get('map50',0)))
            ts.map95.append(float(m.get('map50_95',0)))
            sm=m.get('score_mae',float('nan'))
            if not np.isnan(sm): ts.mae.append(sm)
            ts.best_map=float(m.get('best_map50',0));ts.best_ep=int(m.get('best_epoch',0))
            if dlhub_params:
                eh={'train_loss':float(m.get('box_loss',0)),'map50':float(m.get('map50',0)),'map50_95':float(m.get('map50_95',0)),'current_epoch':epoch,'best_map50':ts.best_map,'best_epoch':ts.best_ep}
                if m.get('score_loss',0)>0: eh['score_loss']=float(m['score_loss'])
                if not np.isnan(sm): eh['score_mae']=float(sm)
                dlhub_params.update_history_epoch(eh,auto_save=(epoch%5==0))
        def on_log(m):
            ts.log(m)
            if dlhub_params: dlhub_params.append_log(m,'training',auto_save=False)
        cb=TrainingCallback(on_epoch_end=on_ep,on_log=on_log,should_stop=lambda:ts.stop)
        trainer=SevSegTrainer(model_yaml=str(yp),pretrained_weights=cfg['pretrained_weights'],data_yaml=dy,output_dir=str(od),epochs=ep,batch_size=bs,img_size=isz,learning_rate=lr,optimizer=opt,patience=pat,device=dev,workers=4,cos_lr=clr,mosaic=mos,fliplr=flr,save_period=sp,callback=cb)
        trainer.extra_aug={'flipud':fud,'hsv_h':hh,'hsv_s':hs,'hsv_v':hv,'translate':tr,'scale':sc}
        ts.trainer=trainer;trainer.train()
        el=time.time()-ts.t0;ts.log("━"*50);ts.log(f"✅ {el/60:.1f}分 mAP@50={ts.best_map:.4f} Ep{ts.best_ep}");ts.log("━"*50)
        if dlhub_params: dlhub_params.mark_training_complete(best_metric=ts.best_map,best_epoch=ts.best_ep)
    except Exception as e:
        import traceback;ts.log(f"❌ {e}");ts.log(traceback.format_exc())
    finally:
        ts.running=False
        if dlhub_params: dlhub_params.save_logs(ts.logs,'training',auto_save=True)

def stop_train():
    if not ts.running: return "⚠️ 无训练"
    ts.stop=True; return "⏹️ 停止中..."

def progress_html():
    pct=(ts.epoch/ts.total)*100 if ts.total>0 else 0
    el=time.time()-ts.t0 if ts.t0 else 0
    sc,st=("#3b82f6","训练中") if ts.running else (("#10b981","已完成") if ts.epoch>0 else ("#6b7280","等待中"))
    mae=f"{ts.mae[-1]:.3f}" if ts.mae else "--"
    return f'<div style="background:linear-gradient(135deg,#f5f7fa,#e4e8ec);padding:15px;border-radius:12px;"><div style="display:flex;justify-content:space-between;margin-bottom:10px;"><b>训练进度</b><span style="background:{sc};color:white;padding:3px 10px;border-radius:12px;font-size:.85em;">{st}</span></div><div style="background:#e5e7eb;border-radius:8px;height:24px;overflow:hidden;margin-bottom:12px;"><div style="background:linear-gradient(90deg,#667eea,#764ba2);height:100%;width:{pct:.1f}%;display:flex;align-items:center;justify-content:center;"><span style="color:white;font-size:.85em;font-weight:bold;">{pct:.1f}%</span></div></div><div style="display:flex;background:white;padding:10px;border-radius:8px;text-align:center;"><div style="flex:1;"><div style="font-size:.75em;color:gray;">轮次</div><b>{ts.epoch}/{ts.total}</b></div><div style="flex:1;border-left:1px solid #ddd;"><div style="font-size:.75em;color:gray;">mAP@50</div><b style="color:#10b981;">{ts.best_map:.4f}</b></div><div style="flex:1;border-left:1px solid #ddd;"><div style="font-size:.75em;color:gray;">MAE</div><b style="color:#f59e0b;">{mae}</b></div><div style="flex:1;border-left:1px solid #ddd;"><div style="font-size:.75em;color:gray;">⏱️</div><b>{fmt_t(el)}</b></div></div></div>'

def train_status():
    for k in list(_figs.keys()):
        if _figs[k]: plt.close(_figs[k])
    html=progress_html()
    st=f"🏃 Ep{ts.epoch}/{ts.total}" if ts.running else ("✅" if ts.epoch>0 else "⏳")
    logs="\n".join(ts.logs[-80:]) or "暂无"
    f1,a=plt.subplots(figsize=(5,3.5),dpi=100);a.set_title('Detection Loss',fontweight='bold');a.grid(True,alpha=.3)
    if ts.box_l:
        a.plot(range(1,len(ts.box_l)+1),ts.box_l,'b-o',label='Box',lw=2,ms=3)
        if ts.cls_l: a.plot(range(1,len(ts.cls_l)+1),ts.cls_l,'r-s',label='Cls',lw=2,ms=3)
        a.legend(fontsize=9)
    else: a.text(.5,.5,'Waiting...',ha='center',va='center',transform=a.transAxes,color='gray')
    f1.tight_layout();_figs['fl']=f1
    f2,a=plt.subplots(figsize=(5,3.5),dpi=100);a.set_title('mAP',fontweight='bold');a.grid(True,alpha=.3)
    if ts.map50:
        a.plot(range(1,len(ts.map50)+1),ts.map50,'g-o',label='@50',lw=2,ms=3)
        if ts.map95: a.plot(range(1,len(ts.map95)+1),ts.map95,'m-s',label='@50:95',lw=2,ms=3)
        a.legend(fontsize=9)
        if ts.best_ep>0 and ts.best_ep<=len(ts.map50): a.scatter([ts.best_ep],[ts.best_map],color='red',s=100,zorder=5,marker='*')
    else: a.text(.5,.5,'Waiting...',ha='center',va='center',transform=a.transAxes,color='gray')
    f2.tight_layout();_figs['fm']=f2
    f3,a=plt.subplots(figsize=(5,3.5),dpi=100);a.set_title('Score',fontweight='bold');a.grid(True,alpha=.3)
    if ts.score_l:
        a.plot(range(1,len(ts.score_l)+1),ts.score_l,color='orange',label='Loss',lw=2,marker='o',ms=3);a.set_ylabel('Loss',color='orange')
        if ts.mae:
            a2=a.twinx();a2.plot(range(1,len(ts.mae)+1),ts.mae,'c-^',label='MAE',lw=2,ms=3);a2.set_ylabel('MAE',color='c')
            h1,l1=a.get_legend_handles_labels();h2,l2=a2.get_legend_handles_labels();a.legend(h1+h2,l1+l2,fontsize=8)
        else: a.legend(fontsize=9)
    else: a.text(.5,.5,'Waiting...',ha='center',va='center',transform=a.transAxes,color='gray')
    f3.tight_layout();_figs['fs']=f3
    return html,st,logs,f1,f2,f3
_figs={}

# ═══════ Tab3: 转换 [Fix-4] ═══════
def start_conv(mp, fmt, dev, half, min_batch, max_batch):
    if cs.running: return "⚠️",""
    if not mp: return "❌ 选模型",""
    cs.reset();cs.running=True
    min_b = int(min_batch) if min_batch else 1
    max_b = int(max_batch) if max_batch else 8
    _save_section('conversion', {'model_path':mp,'format':fmt,'device':dev,'half':half,'min_batch':min_b,'max_batch':max_b})
    threading.Thread(target=_conv, args=(mp, fmt, dev, half, min_b, max_b), daemon=True).start()
    return "🚀",""

def _conv(mp, fmt, dev, half, min_batch, max_batch):
    exported_path=None
    try:
        from ultralytics import YOLO;from utils.env_validator import parse_gpu_choice
        cs.log(f"🔧 加载模型: {mp}");m=YOLO(mp,task='score_detect')
        d,_=parse_gpu_choice(dev)
        if fmt=='TensorRT':
            cs.log("📦 Step 1/2: 导出ONNX (sevseg_yolo.export)...")
            try:
                from sevseg_yolo.export import export_scoreyolo_onnx
                onnx_path=str(Path(mp).with_suffix('.onnx'))
                export_scoreyolo_onnx(m.model, onnx_path, imgsz=640)
                cs.log(f"   ONNX: {onnx_path}")
            except Exception as oe:
                cs.log(f"   sevseg_yolo export失败({oe}), 回退ultralytics...")
                onnx_path=m.export(format='onnx',half=False,imgsz=640,device=d)
                cs.log(f"   ONNX(ultralytics): {onnx_path}")
            cs.log(f"📦 Step 2/2: ONNX→TensorRT (FP16={half}, batch={min_batch}~{max_batch})...")
            from sevseg_yolo.tensorrt_deploy import build_trt_engine
            engine_path=str(Path(onnx_path).with_suffix('.engine'))
            build_trt_engine(str(onnx_path), engine_path, fp16=half, min_batch_size=min_batch, max_batch_size=max_batch)
            cs.log(f"✅ TRT完成: {engine_path}");exported_path=engine_path
        else:
            cs.log("📦 导出ONNX (sevseg_yolo.export)...")
            try:
                from sevseg_yolo.export import export_scoreyolo_onnx
                onnx_path=str(Path(mp).with_suffix('.onnx'))
                export_scoreyolo_onnx(m.model, onnx_path, imgsz=640)
                cs.log(f"✅ ONNX完成: {onnx_path}");exported_path=onnx_path
            except Exception as oe:
                cs.log(f"   sevseg_yolo export失败({oe}), 回退ultralytics...")
                r=m.export(format='onnx',half=False,imgsz=640,device=d)
                cs.log(f"✅ ONNX(ultralytics): {r}");exported_path=str(r)
        if exported_path: _save_deploy_meta(m,mp,exported_path,fmt,half)
        # 打包为 .dlhub
        if exported_path:
            try:
                _conv_tool=str(Path(__file__).parent.parent/"model_conversion")
                if _conv_tool not in sys.path: sys.path.insert(0,_conv_tool)
                from dlhub_packager import DLHubPackager
                _pkg=DLHubPackager().pack(str(Path(exported_path).parent),task_type='sevseg',precision='fp16' if half else 'fp32')
                if _pkg: cs.log(f"📦 已打包: {Path(_pkg).name}")
            except Exception as _pe: cs.log(f"⚠️ .dlhub打包跳过: {_pe}")
    except Exception as e:
        import traceback;cs.log(f"❌ {e}\n{traceback.format_exc()}")
    finally:
        cs.running=False
        if dlhub_params: dlhub_params.save_logs(cs.logs,'conversion',auto_save=True)

def _save_deploy_meta(yolo_m,wt_path,exp_path,fmt,half):
    try:
        names=yolo_m.names or {};nc=len(names)
        my={}
        if hasattr(yolo_m,'model') and hasattr(yolo_m.model,'yaml'): my=yolo_m.model.yaml or {}
        info={"model":{"task":"score_detect","framework":"SevSeg-YOLO","weights":str(wt_path),"exported":str(exp_path),"format":fmt,"fp16":half},"classes":{"nc":nc,"names":{str(k):v for k,v in names.items()}},"preprocessing":{"input_size":[640,640],"input_format":"BCHW","color_space":"RGB","normalize":"divide_255","mean":[0,0,0],"std":[1,1,1],"resize":"letterbox","pad_value":114},"output":{"det_columns":"x1,y1,x2,y2,conf,cls_id,severity","severity_range":[0,10]},"mask_generator":{"method":"MaskGeneratorV2","upsample":"canny_edge_guided_blend"},"exported_at":datetime.now().isoformat()}
        cp=Path(exp_path).parent/"deploy_config.json"
        with open(cp,'w',encoding='utf-8') as f: json.dump(info,f,ensure_ascii=False,indent=2,default=str)
        cs.log(f"📋 部署配置: {cp}")
    except Exception as e: cs.log(f"⚠️ 保存部署配置失败: {e}")

def conv_status(): return ("🔄" if cs.running else "✅" if cs.logs else "⏳"),"\n".join(cs.logs[-50:]) or "暂无"

# ═══════ Tab4: 推理 ═══════
def start_inf(mp,inp,out,conf,dev,mask_on):
    if inf.running: return "⚠️"
    if not mp or not inp: return "❌ 填写路径"
    inf.reset();inf.running=True
    _save_section('inference', {'model_path':mp,'input_path':inp.strip(),'output_dir':(out or'').strip(),'conf':float(conf),'device':dev,'mask_on':mask_on})
    threading.Thread(target=_inf,args=(mp,inp.strip(),(out or'./inference_output').strip(),float(conf),dev,mask_on),daemon=True).start()
    return "🚀 推理中..."

def _inf(mp,inp,out,conf,dev,mask_on):
    try:
        from sevseg_yolo import SevSegYOLO;from utils.env_validator import parse_gpu_choice
        d,_=parse_gpu_choice(dev)
        model=SevSegYOLO(mp,device=d,conf=conf,mask_enabled=bool(mask_on))
        ip=Path(inp);exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
        imgs=[str(ip)] if ip.is_file() else sorted([str(f) for f in ip.rglob('*') if f.suffix.lower() in exts])
        inf.total=len(imgs);op=Path(out);op.mkdir(parents=True,exist_ok=True)
        for i,im in enumerate(imgs):
            r=model.predict(im,conf=conf);inf.done=i+1
            orig=r.image.copy();oh,ow=orig.shape[:2]
            vis=r.image.copy();binary_full=np.zeros((oh,ow),dtype=np.uint8);contour_img=r.image.copy()
            for det in r.detections:
                bx1,by1,bx2,by2=det.bbox;color=det.color
                cv2.rectangle(vis,(bx1,by1),(bx2,by2),color,2)
                label=f"{det.class_name} Sev={det.severity:.1f} Conf={det.confidence:.2f}"
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
                cv2.rectangle(vis,(bx1,by1-th-6),(bx1+tw,by1),color,-1)
                cv2.putText(vis,label,(bx1,by1-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                if mask_on and det.mask is not None and det.mask.sum()>0:
                    m=det.mask;rh,rw=by2-by1,bx2-bx1
                    if m.shape!=(rh,rw): m=cv2.resize(m,(rw,rh),interpolation=cv2.INTER_NEAREST)
                    binary_full[by1:by2,bx1:bx2][m>0]=255
                    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts: c+=np.array([[[bx1,by1]]])
                    cv2.drawContours(contour_img,cnts,-1,(0,255,0),2);cv2.rectangle(contour_img,(bx1,by1),(bx2,by2),color,2)
                else:
                    cv2.rectangle(contour_img,(bx1,by1),(bx2,by2),(0,255,0),2)
            feat_vis=orig.copy()
            try:
                p3=model._hooked.get("p3")
                if p3 is not None:
                    var=np.var(p3,axis=(1,2));topk=min(16,len(var));top_idx=np.argsort(var)[-topk:]
                    feat_map=np.mean(p3[top_idx],axis=0);fmin,fmax=feat_map.min(),feat_map.max()
                    if fmax>fmin: feat_norm=((feat_map-fmin)/(fmax-fmin)*255).astype(np.uint8)
                    else: feat_norm=np.zeros_like(feat_map,dtype=np.uint8)
                    feat_resized=cv2.resize(feat_norm,(ow,oh),interpolation=cv2.INTER_LINEAR)
                    feat_color=cv2.applyColorMap(feat_resized,cv2.COLORMAP_JET)
                    feat_vis=cv2.addWeighted(orig,0.5,feat_color,0.5,0)
                    for det in r.detections: bx1,by1,bx2,by2=det.bbox;cv2.rectangle(feat_vis,(bx1,by1),(bx2,by2),(255,255,255),2)
            except Exception as fe: cv2.putText(feat_vis,f"Feature N/A: {fe}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            cv2.imwrite(str(op/f"det_{Path(im).name}"),vis);cv2.imwrite(str(op/f"feat_{Path(im).name}"),feat_vis)
            cv2.imwrite(str(op/f"bin_{Path(im).name}"),binary_full);cv2.imwrite(str(op/f"cnt_{Path(im).name}"),contour_img)
            inf.results.append({'filename':Path(im).name,'n_det':r.num_detections,'path_det':str(op/f"det_{Path(im).name}"),'path_feat':str(op/f"feat_{Path(im).name}"),'path_bin':str(op/f"bin_{Path(im).name}"),'path_cnt':str(op/f"cnt_{Path(im).name}"),'detections':[{'cls':d.class_name,'sev':d.severity,'conf':d.confidence,'fill':d.fill_ratio} for d in r.detections]})
    except Exception as e:
        import traceback;print(f"❌ {e}\n{traceback.format_exc()}")
    finally: inf.running=False

_inf_was_running = False
def inf_progress():
    """[Fix-3] 推理完成后自动刷新第1张样本"""
    global _inf_was_running
    pct=(inf.done/max(inf.total,1))*100
    st=f"🔍 {inf.done}/{inf.total}" if inf.running else (f"✅ {inf.done}张" if inf.done>0 else "⏳")
    mx=max(len(inf.results),1)
    if _inf_was_running and not inf.running and inf.done > 0:
        _inf_was_running = False
        imgs = show_sample(1)
        return (st, pct, gr.update(maximum=mx, value=1)) + imgs
    _inf_was_running = inf.running
    return (st, pct, gr.update(maximum=mx)) + (gr.update(),)*4 + (gr.update(),)

def show_sample(idx):
    if not inf.results: return None,None,None,None,"<div style='text-align:center;color:gray;padding:20px;'>推理后显示结果</div>"
    idx=max(1,min(int(idx),len(inf.results)));r=inf.results[idx-1]
    def load(p):
        try: img=cv2.imread(p);return cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img is not None else None
        except: return None
    det_img=load(r['path_det']);feat_img=load(r['path_feat']);bin_img=load(r['path_bin']);cnt_img=load(r['path_cnt'])
    dets_html="".join([f"<tr><td>{d['cls']}</td><td style='color:{'red' if d['sev']>7 else 'orange' if d['sev']>4 else 'green'}'>{d['sev']:.1f}</td><td>{d['conf']:.2f}</td><td>{d['fill']:.3f}</td></tr>" for d in r['detections']])
    card=f"""<div style="background:#f8f9fa;border-radius:10px;padding:15px;"><h4>📄 {r['filename']} — {r['n_det']}个缺陷</h4><table style="width:100%;border-collapse:collapse;"><tr style="background:#e9ecef;"><th>类别</th><th>Severity</th><th>置信度</th><th>Mask填充</th></tr>{dets_html}</table></div>"""
    return det_img,feat_img,bin_img,cnt_img,card

def nav_sample(direction,cur):
    mx=max(len(inf.results),1);cur=int(cur)
    new=max(1,cur-1) if direction=='prev' else min(mx,cur+1)
    return (new,)+show_sample(new)

# ═══════ UI ═══════
def create_ui():
    gpu=get_gpus()
    # [Fix-2] 加载所有保存的参数
    sd = _get_section('data'); sm = _get_section('model'); st = _get_section('training')
    sa = _get_section('augmentation'); saved_dev = get_saved_param('device', gpu[0] if gpu else "CPU")
    sc_conv = _get_section('conversion'); si = _get_section('inference'); s_convert = _get_section('convert')

    # [Fix-2] 恢复训练历史 → 曲线图
    saved_history = {} if not dlhub_params else dlhub_params.get_history()
    if saved_history and saved_history.get('train_losses'):
        ts.box_l = saved_history.get('train_losses', [])
        ts.map50 = saved_history.get('map50_list', [])
        ts.map95 = saved_history.get('map50_95_list', [])
        ts.score_l = saved_history.get('score_losses', [])
        ts.mae = saved_history.get('score_mae_list', [])
        ts.best_map = saved_history.get('best_map50', 0.0)
        ts.best_ep = saved_history.get('best_epoch', 0)
        ts.epoch = saved_history.get('current_epoch', 0)
        ts.total = saved_history.get('total_epochs', 0)
        ts.out = saved_history.get('output_dir', '')
        print(f"[DL-Hub] ✓ 已恢复训练历史: {len(ts.box_l)} epochs, mAP@50={ts.best_map:.4f}")

    # [Fix-2] 恢复日志
    saved_train_logs = ""; saved_conv_logs = ""
    if dlhub_params:
        try:
            tl = dlhub_params.get_logs('training') or []
            if tl: ts.logs = tl; saved_train_logs = "\n".join(tl); print(f"[DL-Hub] ✓ 已恢复训练日志: {len(tl)} 行")
        except: pass
        try:
            cl = dlhub_params.get_logs('conversion') or []
            if cl: cs.logs = cl; saved_conv_logs = "\n".join(cl); print(f"[DL-Hub] ✓ 已恢复转换日志: {len(cl)} 行")
        except: pass

    env_text=check_env()
    try:
        from dlhub_user_widget import get_widget_head,get_widget_html;hh,wh=get_widget_head(),get_widget_html()
    except: hh,wh="",""
    with gr.Blocks(title="🏭 SevSeg-YOLO",css=CUSTOM_CSS,head=hh) as app:
        if wh: gr.HTML(wh)
        gr.HTML("<h1 style='text-align:center;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2em;margin-bottom:5px;'>🏭 SevSeg-YOLO 工业缺陷检测</h1><p style='text-align:center;color:#666;'>检测 · 严重程度评分 · 零标注近似分割</p>")
        with gr.Accordion("📋 环境信息",open=False):
            gr.Textbox(value=env_text,lines=8,interactive=False,show_label=False)
        with gr.Tabs():
            with gr.TabItem("📂 数据转换"):
                gr.Markdown("### LabelMe JSON → SevSeg-YOLO 6列标注")
                with gr.Row():
                    with gr.Column(scale=1):
                        ci=gr.Textbox(label="图像文件夹",value=s_convert.get('images_dir',''))
                        cj=gr.Textbox(label="JSON文件夹",value=s_convert.get('jsons_dir',''))
                        co=gr.Textbox(label="输出目录",value=s_convert.get('output_dir',''))
                        with gr.Row(): cv=gr.Slider(.05,.5,value=s_convert.get('val_ratio',.2),step=.05,label="验证比例");csd=gr.Number(value=s_convert.get('seed',42),label="种子",precision=0)
                        ckb=gr.Button("🔍 验证路径");ckr=gr.Textbox(label="验证",lines=5,interactive=False)
                        cvb=gr.Button("🚀 开始转换",variant="primary",elem_classes="primary-btn")
                        cvs=gr.Textbox(label="结果",lines=4,interactive=False);cyp=gr.Textbox(label="data.yaml路径",interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 数据集分析");cf1=gr.Plot(label="类别分布")
                        gr.Markdown("### 📏 Severity分数分析")
                        cls_sel=gr.Dropdown(choices=["全部"],value="全部",label="选择类别查看分数分布")
                        cf2=gr.Plot(label="Severity分数分布")
                ckb.click(fn=validate_paths,inputs=[ci,cj],outputs=ckr)
                cvb.click(fn=do_convert,inputs=[ci,cj,co,cv,csd],outputs=[cvs,cyp,cf1,cf2,cls_sel])
                cls_sel.change(fn=show_class_scores,inputs=cls_sel,outputs=cf2)
            with gr.TabItem("🎯 模型训练"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1,min_width=350):
                        gr.Markdown("### ⚙️ 配置")
                        with gr.Group():
                            gr.Markdown("**📂 数据**");dy=gr.Textbox(label="data.yaml",value=sd.get('data_yaml',''));dst=gr.Textbox(label="验证",interactive=False,lines=3)
                        with gr.Group():
                            gr.Markdown("**🧠 模型**");msc=gr.Dropdown(["超小","小","中","大","超大"],value=sm.get('scale','中'),label="规模")
                            mi=gr.Textbox(label="信息",interactive=False,lines=5);ps=gr.Textbox(label="权重",interactive=False,lines=2)
                        with gr.Group():
                            gr.Markdown("**⚙️ 参数**")
                            with gr.Row(): ep=gr.Slider(10,300,value=st.get('epochs',105),step=5,label="轮数");bs=gr.Slider(1,128,value=st.get('batch_size',32),step=1,label="Batch")
                            with gr.Row(): lr=gr.Number(value=st.get('learning_rate',.01),label="学习率",precision=6);opt=gr.Dropdown(['SGD','Adam','AdamW'],value=st.get('optimizer','SGD'),label="优化器")
                            with gr.Row(): isz=gr.Dropdown([320,416,512,640,768,1024],value=st.get('img_size',640),label="尺寸");pat=gr.Slider(5,100,value=st.get('patience',50),step=5,label="早停")
                        with gr.Group():
                            gr.Markdown("**🎨 增强**")
                            with gr.Row(): mos=gr.Slider(0,1,value=sa.get('mosaic',1.),step=.1,label="Mosaic");flr=gr.Slider(0,1,value=sa.get('fliplr',.5),step=.1,label="水平翻转")
                            with gr.Row(): fud=gr.Slider(0,1,value=sa.get('flipud',0.),step=.1,label="垂直翻转");hh_=gr.Slider(0,.1,value=sa.get('hsv_h',.015),step=.005,label="色调")
                            with gr.Row(): hs_=gr.Slider(0,1,value=sa.get('hsv_s',.7),step=.1,label="饱和度");hv_=gr.Slider(0,1,value=sa.get('hsv_v',.4),step=.1,label="明度")
                            with gr.Row(): tr_=gr.Slider(0,.5,value=sa.get('translate',.1),step=.05,label="平移");sc_=gr.Slider(0,1,value=sa.get('scale',.5),step=.1,label="缩放")
                            clr=gr.Checkbox(value=sa.get('cos_lr',True),label="余弦LR");gr.Markdown("⚠️ **MixUp强制关闭**")
                        with gr.Group():
                            sp=gr.Slider(0,50,value=st.get('save_period',0),step=5,label="保存间隔")
                            gp=gr.Dropdown(choices=gpu,value=saved_dev if saved_dev in gpu else (gpu[0] if gpu else "CPU"),label="设备")
                        with gr.Row():
                            tb=gr.Button("🚀 训练",variant="primary",elem_classes="primary-btn");sb=gr.Button("⏹️ 停止",elem_classes="stop-btn")
                        tst=gr.Textbox(label="状态",interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 监控");ph=gr.HTML(progress_html())
                        with gr.Row(): fl=gr.Plot(label="Loss");fm=gr.Plot(label="mAP");fs=gr.Plot(label="Score")
                        tl=gr.Textbox(label="日志",lines=16,interactive=False,elem_classes="log-box",value=saved_train_logs)
                tt=gr.Timer(value=3);tt.tick(fn=train_status,outputs=[ph,tst,tl,fl,fm,fs])
                dy.change(fn=val_yaml,inputs=dy,outputs=dst);dy.submit(fn=val_yaml,inputs=dy,outputs=dst)
                msc.change(fn=model_info,inputs=msc,outputs=mi);msc.change(fn=pt_status,inputs=msc,outputs=ps)
                tb.click(fn=start_train,inputs=[dy,msc,ep,bs,lr,opt,isz,pat,mos,flr,fud,hh_,hs_,hv_,tr_,sc_,clr,sp,gp],outputs=[tst,tl])
                sb.click(fn=stop_train,outputs=tst)
            with gr.TabItem("🔄 模型转换"):
                gr.Markdown("### 导出 ONNX / TensorRT")
                with gr.Row():
                    with gr.Column(scale=1):
                        ef=gr.Dropdown(choices=[],label="训练文件夹");em=gr.Dropdown(choices=[],label="模型",allow_custom_value=True)
                        efmt=gr.Dropdown(['ONNX','TensorRT'],value=sc_conv.get('format','ONNX'),label="格式")
                        edev=gr.Dropdown(choices=gpu,value=sc_conv.get('device',gpu[0] if gpu else 'CPU'),label="设备")
                        ehf=gr.Checkbox(value=sc_conv.get('half',True),label="FP16")
                        with gr.Row():
                            e_minb=gr.Number(value=sc_conv.get('min_batch',1),label="Min Batch",precision=0,info="TensorRT最小batch size")
                            e_maxb=gr.Number(value=sc_conv.get('max_batch',8),label="Max Batch",precision=0,info="TensorRT最大batch size")
                        ebt=gr.Button("🚀 转换",variant="primary",elem_classes="primary-btn");es=gr.Textbox(label="状态",interactive=False)
                    with gr.Column(scale=2):
                        el=gr.Textbox(label="日志",lines=25,interactive=False,elem_classes="log-box",value=saved_conv_logs)
                ct=gr.Timer(value=2);ct.tick(fn=conv_status,outputs=[es,el])
                ef.focus(fn=scan_folders,outputs=ef);ef.change(fn=scan_models,inputs=ef,outputs=em)
                ebt.click(fn=start_conv,inputs=[em,efmt,edev,ehf,e_minb,e_maxb],outputs=[es,el])
            with gr.TabItem("🔍 批量推理"):
                gr.Markdown("### 推理 + 样本预览\n检测 · Severity评分 · Mask热力图/二值图/轮廓")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1,min_width=300):
                        ifd=gr.Dropdown(choices=[],label="训练文件夹");ifm=gr.Dropdown(choices=[],label="模型",allow_custom_value=True)
                        iip=gr.Textbox(label="输入路径(文件夹/图片)",value=si.get('input_path',''))
                        iop=gr.Textbox(label="输出目录",value=si.get('output_dir',str(get_output_dir()/'inference_output')))
                        icf=gr.Slider(.01,.95,value=si.get('conf',.15),step=.01,label="置信度")
                        idv=gr.Dropdown(choices=gpu,value=si.get('device',gpu[0] if gpu else 'CPU'),label="设备")
                        imk=gr.Checkbox(value=si.get('mask_on',True),label="MaskGenerator(缺陷分割)",info="零标注近似分割: 双峰通道选择+Canny引导上采样")
                        ibt=gr.Button("🚀 推理",variant="primary",elem_classes="primary-btn")
                        ist_=gr.Textbox(label="状态",interactive=False);ipg=gr.Slider(0,100,value=0,label="进度",interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ 样本预览")
                        with gr.Row():
                            prev_btn=gr.Button("◀ 上一个",size="lg");idx_slider=gr.Slider(label="样本编号",minimum=1,maximum=1,value=1,step=1);next_btn=gr.Button("下一个 ▶",size="lg")
                        with gr.Row(): img_det=gr.Image(label="检测+Severity",height=250);img_feat=gr.Image(label="P3特征图",height=250)
                        with gr.Row(): img_bin=gr.Image(label="Mask二值图",height=250);img_cnt=gr.Image(label="Mask轮廓图",height=250)
                        result_card=gr.HTML("<div style='text-align:center;color:gray;padding:20px;'>推理后显示结果</div>")
                it=gr.Timer(value=2);it.tick(fn=inf_progress,outputs=[ist_,ipg,idx_slider,img_det,img_feat,img_bin,img_cnt,result_card])
                ifd.focus(fn=scan_folders,outputs=ifd);ifd.change(fn=scan_models,inputs=ifd,outputs=ifm)
                ibt.click(fn=start_inf,inputs=[ifm,iip,iop,icf,idv,imk],outputs=ist_)
                idx_slider.change(fn=show_sample,inputs=idx_slider,outputs=[img_det,img_feat,img_bin,img_cnt,result_card])
                prev_btn.click(fn=lambda c:nav_sample('prev',c),inputs=idx_slider,outputs=[idx_slider,img_det,img_feat,img_bin,img_cnt,result_card])
                next_btn.click(fn=lambda c:nav_sample('next',c),inputs=idx_slider,outputs=[idx_slider,img_det,img_feat,img_bin,img_cnt,result_card])
    return app

if __name__=="__main__":
    os.environ['NO_PROXY']='localhost,127.0.0.1,0.0.0.0'
    parser=argparse.ArgumentParser();parser.add_argument('--task-dir',type=str,default=None);parser.add_argument('--port',type=int,default=7860)
    args,_=parser.parse_known_args();app=create_ui()
    port=dlhub_adapter.port if(dlhub_adapter and dlhub_adapter.is_dlhub_mode) else args.port
    app.launch(server_name="127.0.0.1",server_port=port,share=False,inbrowser=not(dlhub_adapter and dlhub_adapter.is_dlhub_mode))
