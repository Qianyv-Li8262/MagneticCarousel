exec(open('audits/eddy_integration_20261016/analyze_height39_transition.py',encoding='utf-8-sig').read().split('report=[]')[0])
for lo,hi in [(1500,1550),(1600,1625),(1650,1700),(4500,5000)]:
 w=(t>=lo)&(t<hi);oz=s[:,11]*20.943951
 print(lo,hi,'JA axial/other power',np.mean(a['JA_torque_z_uNm'][w]*oz[w]),np.mean(a['JA_mechanical_uW'][w]-a['JA_torque_z_uNm'][w]*oz[w]),'eddy axial/other',np.mean(a['eddy_torque_z_uNm'][w]*oz[w]),np.mean(a['eddy_mechanical_uW'][w]-a['eddy_torque_z_uNm'][w]*oz[w]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10})
fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
w=(t>=1400)&(t<=1800);tx=t[w]
axs[0].plot(tx,s[w,11]*20.943951,label='竖直自旋');axs[0].plot(tx,4*(1-phidot[w])*20.943951,'--',label='轨道坐标系基频');axs[0].set_ylabel('角速度 (rad/s)');axs[0].legend()
n=201;sm=lambda v:np.convolve(v,np.ones(n)/n,mode='valid');tt=t[n//2:-(n//2)];z=(tt>=1400)&(tt<=1800)
for k,label in [('JA_torque_z_uNm','磁滞轴向力矩'),('eddy_torque_z_uNm','涡流轴向力矩')]:axs[1].plot(tt[z],sm(a[k])[z],label=label)
axs[1].set_ylabel('短窗平均力矩 (μN·m)');axs[1].legend()
axs[2].plot(tx,np.linalg.norm(s[w,3:5],axis=1)*366.519,label='质心速率',lw=.7)
axs[2].set_ylabel('质心速率 (mm/s)');axs[2].legend()
axs[3].plot(tx,r[w]*17.5);axs[3].set_ylabel('半径 (mm)');axs[3].set_xlabel('无量纲时间 t2D')
for ax in axs:ax.grid(alpha=.25);ax.axvspan(1600,1650,color='orange',alpha=.1)
fig.suptitle('t≈1600：自旋响应转变，但没有产生持续平动加速',fontsize=15)
fig.savefig(out/'transition1600.png',dpi=150)
