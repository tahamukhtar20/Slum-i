"""Table II: test mIoU and slum IoU (mean and standard deviation over five seeds) of every method.
Reads results/all.json and writes results/table2.tex (a standalone LaTeX page with the table);
prints the paired change of Ours over each baseline per setting. Usage: python analysis/table2.py"""
import json,statistics as st
D=json.load(open('results/all.json'))
seeds=[0,42,123,999,1337]; cities=['lahore','karachi','mumbai']; sats=['esri','s2']
R101=[('Supervised (R101)','sup_r101'),('FixMatch','fixmatch'),('Ours (FixMatch)','fixmatch_freematchc'),('UniMatch','unimatch'),('Ours (UniMatch)','unimatch_freematchc')]
DINO=[('Supervised (DINOv2)','sup_dino'),('UniMatch V2','unimatchv2'),('Ours (UniMatch V2)','unimatchv2_freematchc')]
def vals(sat,c,m,b,i): return [D[f'{sat}|{c}|{m}|{b}|{s}'][i] for s in seeds]
def ms(sat,c,m,b,i): v=vals(sat,c,m,b,i); return st.mean(v),st.stdev(v)
B=['10','20','30','100']
best={(sat,c,i):max(ms(sat,c,m,b,i)[0] for b in B for _,m in R101+DINO if f'{sat}|{c}|{m}|{b}|0' in D) for sat in sats for c in cities for i in (0,1)}
def row(name,m,b,block):
    cells=[name]
    for c in cities:
        for sat in sats:
            for i in (0,1):
                mu,sd=ms(sat,c,m,b,i); num=f'{mu:.1f}'
                if len(block)>1:
                    r=sorted((ms(sat,c,x,b,i)[0] for x in block),reverse=True)
                    if mu==r[0]: num=rf'\mathbf{{{num}}}'
                    elif mu==r[1]: num=rf'\underline{{{num}}}'
                t=f'${num}_{{\\pm{sd:.1f}}}$'
                cells.append(t)
    return ' & '.join(cells)+r' \\'
L=[]
for b in B:
    L+= [rf'\multicolumn{{13}}{{c}}{{{b}\% labels}} \\', r'\cmidrule{1-13}']
    r1=[x for x in R101 if b!='100' or x[1].startswith('sup')]; d1=[x for x in DINO if b!='100' or x[1].startswith('sup')]
    L+=[row(n,m,b,[x[1] for x in r1]) for n,m in r1]
    L.append(r'\arrayrulecolor[gray]{0.65}\cmidrule[0.3pt]{1-13}\arrayrulecolor{black}')
    L+=[row(n,m,b,[x[1] for x in d1]) for n,m in d1]
    L.append(r'\midrule' if b!='100' else r'\bottomrule\addlinespace[2pt]')
doc=r'''\documentclass{article}\usepackage[a4paper,landscape,margin=1cm]{geometry}
\usepackage{amsmath,amsfonts}\usepackage{array}\usepackage{stfloats}\usepackage{tabularx}\usepackage{booktabs}\usepackage{colortbl}
\pagestyle{empty}
\begin{document}
\begin{table*}[!t]
\caption{Test Results (mIoU and Slum IoU, \%, Mean and Standard Deviation over Five Seeds)}
\label{tab:results}
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabularx}{\textwidth}{@{}l*{12}{>{\centering\arraybackslash}X}@{}}
\toprule
 & \multicolumn{4}{c}{\textbf{Lahore}} & \multicolumn{4}{c}{\textbf{Karachi}} & \multicolumn{4}{c}{\textbf{Mumbai}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}
 & \multicolumn{2}{c}{Esri} & \multicolumn{2}{c}{Sentinel-2} & \multicolumn{2}{c}{Esri} & \multicolumn{2}{c}{Sentinel-2} & \multicolumn{2}{c}{Esri} & \multicolumn{2}{c}{Sentinel-2} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(l){12-13}
Method & mIoU & Slum & mIoU & Slum & mIoU & Slum & mIoU & Slum & mIoU & Slum & mIoU & Slum \\
\midrule
'''+'\n'.join(L)+r'''
\multicolumn{13}{@{}l}{Bold marks the best value in each backbone block and underline the second best.} \\
\end{tabularx}
\end{table*}
\null
\end{document}
'''
open('results/table2.tex','w').write(doc)
# summary
print('PAIRED change (Ours - base): sat city method budget dmIoU wins dslum wins')
agg={}
for sat in sats:
  for c in cities:
    for m in ['fixmatch','unimatch','unimatchv2']:
      for b in ['10','20','30']:
        out=[]
        for i in (0,1):
            d=[a-bb for a,bb in zip(vals(sat,c,m+'_freematchc',b,i),vals(sat,c,m,b,i))]
            out.append((st.mean(d),sum(x>0 for x in d)))
            agg.setdefault((sat,i),[]).append(st.mean(d))
        print(f'{sat:4s} {c:7s} {m:10s} {b:>2s} {out[0][0]:+6.2f} {out[0][1]}/5 {out[1][0]:+6.2f} {out[1][1]}/5')
for (sat,i),v in sorted(agg.items()):
    print(sat,['mIoU','slum'][i],'settings up',sum(x>0 for x in v),'/',len(v),'mean change',round(st.mean(v),2))
wins={}
for sat in sats:
  for c in cities:
    for b in ['10','20','30']:
      for bk in (R101,DINO):
        for i in (0,1):
          w=max(bk,key=lambda x:ms(sat,c,x[1],b,i)[0])[1]
          k=(sat,i); wins.setdefault(k,[0,0]); wins[k][0]+=w.endswith('_freematchc'); wins[k][1]+=1
          if not w.endswith('_freematchc'): print('block lost:',sat,c,b,'R101' if bk is R101 else 'DINO',['mIoU','slum'][i],'winner',w)
print('block wins',wins)
