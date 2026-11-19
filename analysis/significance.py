"""Paired change of Ours over each baseline per setting, with t-tests per setting and Wilcoxon,
sign and bootstrap tests over settings, overall and per group. Usage: python analysis/significance.py"""
import json,numpy as np
from scipy import stats
D=json.load(open('results/all.json')); seeds=[0,42,123,999,1337]
rng=np.random.default_rng(0)
rows=[]
for sat in ['esri','s2']:
  for c in ['lahore','karachi','mumbai']:
    for m in ['fixmatch','unimatch','unimatchv2']:
      for b in ['10','20','30']:
        for i,met in enumerate(['mIoU','slum']):
          d=np.array([D[f'{sat}|{c}|{m}_freematchc|{b}|{s}'][i]-D[f'{sat}|{c}|{m}|{b}|{s}'][i] for s in seeds])
          rows.append(dict(sat=sat,c=c,m=m,b=b,met=met,d=d,mean=d.mean(),p=stats.ttest_1samp(d,0).pvalue))
def summ(sel,label):
    for met in ['slum','mIoU']:
        r=[x for x in rows if sel(x) and x['met']==met]
        mu=np.array([x['mean'] for x in r]); n=len(mu)
        w=stats.wilcoxon(mu); sg=stats.binomtest(int((mu>0).sum()),n)
        bs=[rng.choice(mu,n).mean() for _ in range(10000)]; lo,hi=np.percentile(bs,[2.5,97.5])
        sigp=sum(1 for x in r if x['p']<0.05 and x['mean']>0); sign=sum(1 for x in r if x['p']<0.05 and x['mean']<0)
        print(f'{label:22s} {met:4s} n={n:2d} up={int((mu>0).sum()):2d} mean={mu.mean():+.2f} [{lo:+.2f},{hi:+.2f}] median={np.median(mu):+.2f} wilcoxon p={w.pvalue:.1e} sign p={sg.pvalue:.1e} | per-setting t-test p<.05: {sigp} up, {sign} down')
summ(lambda x:True,'all')
for s in ['esri','s2']: summ(lambda x,s=s:x['sat']==s,s)
for m in ['fixmatch','unimatch','unimatchv2']: summ(lambda x,m=m:x['m']==m,m)
for b in ['10','20','30']: summ(lambda x,b=b:x['b']==b,'budget '+b)
for c in ['lahore','karachi','mumbai']: summ(lambda x,c=c:x['c']==c,c)
print('significant losses:',[(x['sat'],x['c'],x['m'],x['b'],x['met'],round(x['mean'],2),round(x['p'],3)) for x in rows if x['p']<0.05 and x['mean']<0])
