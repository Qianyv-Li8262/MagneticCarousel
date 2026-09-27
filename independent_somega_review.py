# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Read-only model review; no trajectory integration. Writes figures and text."""
from pathlib import Path
from decimal import Decimal, getcontext
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import RESULTS_DIR, SOMEGA_DD_LOG_DIR, SOMEGA_FP64_LOG_DIR, image_output_path

getcontext().prec = 60
HERE = Path(__file__).resolve().parent
LOG_DIR = SOMEGA_FP64_LOG_DIR
DD_LOG_DIR = SOMEGA_DD_LOG_DIR
OUTPUT_DIR = RESULTS_DIR / 'critical-slowing'
BASE = Decimal('0.55413752928877091')
UNIT = Decimal('1e-10')
OUT = OUTPUT_DIR / 'somega_independent_review.txt'
report=[]
def say(s):
    print(s, flush=True); report.append(str(s))
def read(path):
    b=path.read_bytes(); txt=b.decode('utf-16') if b[:2] in (b'\xff\xfe',b'\xfe\xff') else b.decode('utf-8-sig')
    rows=[]
    for line in txt.splitlines():
        q=line.split()
        if len(q)!=3: continue
        try: rows.append((Decimal(q[0]),int(q[1]),float(q[2])))
        except (ValueError,ArithmeticError): pass
    return rows
blocks={p.name:read(p) for p in LOG_DIR.glob('slow_*.log')}
dd=read(DD_LOG_DIR/'slow_fine8dd.log')
unique={}
for name,rows in sorted(blocks.items()):
    say(f'{name}: n={len(rows)} locked={sum(r[1]==1 for r in rows)}')
    for s,c,t in rows:
        # FP64 inputs keyed by their actual binary value, not printed decimals.
        key=Decimal.from_float(float(s))
        unique.setdefault(key,set()).add((c,t))
say(f'FP64 unique inputs={len(unique)}, inconsistent duplicates={sum(len(v)>1 for v in unique.values())}')
fp=[(s,c,t) for s,v in sorted(unique.items()) for c,t in sorted(v) if c==1]
def arrays(rows):
    return np.array([float((s-BASE)/UNIT) for s,c,t in rows]),np.array([t for s,c,t in rows])
xf,yf=arrays(fp); xd,yd=arrays(dd)
# Conditional boundary used in earlier FP64 analysis; it is not a valid monotone bracket.
scfp=float((Decimal.from_float(float(BASE))-BASE)/UNIT)+np.spacing(float(BASE))/2e-10

def fit(x,y,kind,sc=None,width=None):
    # Linear intercept/amplitude solved exactly; only nonlinear shape is optimized.
    def calc(q,ret=False):
        j=0
        if sc is None: center=x.max()+np.exp(q[j]); j+=1
        else: center=sc
        d=center-x
        if kind=='log': phi=-np.log(d)
        elif kind=='sqrt': phi=-np.sqrt(d)
        elif kind=='power':
            beta=q[j]; phi=-d**beta
        elif kind=='cross':
            w=np.exp(q[j]) if width is None else width
            phi=-2*np.arcsinh(np.sqrt(d/w))
        mat=np.column_stack((np.ones(len(x)),phi))
        coef=np.linalg.lstsq(mat,y,rcond=None)[0]
        res=mat@coef-y
        return (center,coef,res) if ret else res
    starts=[[]] if sc is not None else [[np.log(g)] for g in (.001,.1,1,10)]
    if kind=='power': starts=[q+[b] for q in starts for b in (.1,.5,.9)]
    if kind=='cross' and width is None: starts=[q+[np.log(w)] for q in starts for w in (1e-6,.1,1,10)]
    lo=([-25] if sc is None else [])+([.001] if kind=='power' else [-25] if kind=='cross' and width is None else [])
    hi=([25] if sc is None else [])+([1.5] if kind=='power' else [25] if kind=='cross' and width is None else [])
    if starts==[[]]: q=np.array([])
    else:
        sols=[least_squares(calc,q,bounds=(lo,hi),max_nfev=1800,ftol=1e-10,xtol=1e-10,gtol=1e-9) for q in starts]
        q=min(sols,key=lambda a:np.dot(a.fun,a.fun)).x
    center,coef,res=calc(q,True); rss=res@res; k=2+len(q)
    return dict(sc=center,coef=coef,q=q,rmse=np.sqrt(rss/len(x)),maxabs=np.max(abs(res)),aic=len(x)*np.log(rss/len(x))+2*k,res=res)

def show(label,x,y,kind,**kw):
    f=fit(x,y,kind,**kw)
    say(f'{label} {kind}: n={len(x)} rmse={f["rmse"]:.8g} max={f["maxabs"]:.8g} AIC={f["aic"]:.5f} sc={BASE+Decimal(str(f["sc"]))*UNIT} coef={f["coef"]} shape={f["q"]}')
    return f

for cut in (np.inf,1e-3,1e-6,1e-9,3e-11,1e-11,1e-12):
    m=(scfp-xf)*1e-10<cut
    for model in ('log','cross'):
        show(f'FP64 cutoff={cut} fixed_sc',xf[m],yf[m],model,sc=scfp)
        show(f'FP64 cutoff={cut} free_sc',xf[m],yf[m],model)
    if cut<=3e-11:
        show(f'FP64 cutoff={cut} fixed_sc',xf[m],yf[m],'power',sc=scfp)

# Reproduce annulus without selecting machine-limit points.
m=((scfp-xf)*1e-10>=1e-12)&((scfp-xf)*1e-10<=3e-11)
for model in ('log','sqrt','power','cross'):
    show('FP64 annulus fixed_sc',xf[m],yf[m],model,sc=scfp)
    show('FP64 annulus free_sc',xf[m],yf[m],model)

ddfits={}
for n in (20,40,60,100):
    for model in ('log','sqrt','cross'):
        ddfits[n,model]=show(f'DD tail={n}',xd[-n:],yd[-n:],model)
for w in (1e-6,1e-4,.001,.01,.1,.3,1):
    show(f'DD crossover physical_width={w*1e-10}',xd,yd,'cross',width=w)

# Combine DD with successively less near-boundary FP64 data; do not assume identical backends.
for cutoff in (1e-7,1e-8,1e-9):
    m=(scfp-xf)*1e-10>=cutoff
    x=np.r_[xf[m],xd]; y=np.r_[yf[m],yd]
    # Restrict distant non-asymptotic points separately.
    for outer in (np.inf,1e-3,1e-4):
        keep=(scfp-x)*1e-10<outer
        show(f'mixed fp_distance>={cutoff} outer={outer}',x[keep],y[keep],'log')

fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
fl=ddfits[100,'log']; center=fl['sc']
axes[0].semilogx((center-xf)*1e-10,yf,'.',ms=3,label='FP64 all successful')
axes[0].semilogx((center-xd)*1e-10,yd,'.',label='DD')
axes[0].set(xlabel='Distance to DD log extrapolated critical point',ylabel='Transition time')
axes[0].legend()
for model in ('log','sqrt','cross'):
    f=ddfits[100,model]
    axes[1].plot(xd,f['res'],label=model)
axes[1].set(xlabel='(somega - 0.55413752928877091) / 1e-10',ylabel='DD fitted minus observed time')
axes[1].legend()
ff=blocks['slow_fine8.log']; yy=np.array([r[2] for r in ff]); cc=np.array([r[1] for r in ff])
axes[2].plot(xd[cc==1],(yd-yy)[cc==1],'.')
axes[2].set(xlabel='(somega - FP64 apparent boundary) / 1e-10',ylabel='DD minus FP64 transition time')
for ax in axes: ax.grid(alpha=.25)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure_path=image_output_path(OUTPUT_DIR/'somega_independent_review.png')
figure_path.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(figure_path,dpi=180)
OUT.write_text('\n'.join(report),encoding='utf-8')
