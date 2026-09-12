from pathlib import Path
from decimal import Decimal, getcontext
import ast
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
getcontext().prec=60
root=Path(__file__).resolve().parent
tree=ast.parse((root/'independent_somega_review.py').read_text())
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('read','fit'):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<fit helpers>','exec'))
base=Decimal('0.5541375293204'); unit=Decimal('1e-12')
f7=read(root/'slow-somega-scan/slow_fine7.log')
f8=read(root/'slow-somega-scan/slow_fine8.log')
dd8=read(root/'slow_fine8dd.log')
new=read(root/'slow_somega_dd_discriminate50.log')
def arrays(rows):
    rows=[r for r in rows if r[1]==1]
    return np.array([float((r[0]-base)/unit) for r in rows]),np.array([r[2] for r in rows])
x,y=arrays(dd8+new); fitted=fit(x,y,'log')
sc=base+Decimal(str(fitted['sc']))*unit
print('DD combined sc',sc,'B',fitted['coef'][1],'RMSE',fitted['rmse'])
fig,axes=plt.subplots(1,2,figsize=(14,5.7),layout='constrained',sharey=True)
scfp=Decimal('0.55413752928877096914916933201311')
panels=[(axes[0],scfp,[(f7,'FP64 fine7','#808080'),(f8,'FP64 fine8','#2878b5')]),
        (axes[1],sc,[(f7,'FP64 fine7','#808080'),(dd8,'DD fine8','#ed9b21'),(new,'DD new scan (46 locked)','#c43b45')])]
for ax,center,blocks in panels:
    for rows,label,color in blocks:
        d=np.array([float(center-r[0]) for r in rows if r[1]==1]); t=np.array([r[2] for r in rows if r[1]==1])
        ax.semilogx(d,t,'o',ms=3.8,color=color,label=label)
    ax.set_xlim(1.3e-8,7e-15); ax.set_ylim(15500,28500)
    ax.set_xlabel(r'$s_c-s_\omega$ (log scale; closer to boundary $\rightarrow$)')
    ax.grid(alpha=.2,which='both'); ax.legend(loc='upper left',fontsize=9)
axes[0].set_title('FP64 fine7 + fine8\nDistance to FP64 apparent boundary')
axes[1].set_title('FP64 fine7 + DD fine8 + new DD scan\nDistance to combined DD log-fit boundary')
axes[0].set_ylabel('Transition time (linear scale)')
d=np.geomspace(7e-15,1.3e-8,600)
a,b=fitted['coef']; axes[1].plot(d,a-b*np.log(d/1e-12),'k-',lw=1,zorder=0)
axes[1].text(.04,.07,f'DD log fit: B = {b:.3f}\n146 points, RMSE = {fitted["rmse"]:.3f}\n4 unconfirmed points omitted',transform=axes[1].transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
fig.suptitle('Critical slowing: the new DD scan extends the logarithmic trend',fontsize=14)
fig.savefig(root/'somega_dd_discriminate50_linear_log.png',dpi=180)
plt.close(fig)
lines=[]
for label,rows in [('new46',new),('combined146',dd8+new)]:
    xx,yy=arrays(rows)
    for model in ('log','sqrt','cross'):
        f=fit(xx,yy,model)
        lines.append(f'{label} {model}: RMSE={f["rmse"]} max={f["maxabs"]} sc={base+Decimal(str(f["sc"]))*unit} coef={f["coef"]} shape={f["q"]} AIC={f["aic"]}')
for w in (1e-4,.001,.01,.1,1):
    f=fit(x,y,'cross',width=w)
    lines.append(f'cross width={w*1e-12} RMSE={f["rmse"]} sc={base+Decimal(str(f["sc"]))*unit}')
(root/'somega_dd_discriminate50_fit.txt').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
