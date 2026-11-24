from pathlib import Path
import numpy as np, pandas as pd, json
D=Path(__file__).parent/'data'; D.mkdir(exist_ok=True)
r=np.random.default_rng(42); n=10000; ids=np.arange(100001,100001+n)
p=pd.DataFrame({'patient_id':ids,'program':r.choice(['MAP','MAP Basic','Other safety-net'],n),'region':r.choice(['North','South','East','Central','West'],n),'age_band':r.choice(['18–39','40–64','65+'],n),'chronic_conditions':r.poisson(1.7,n)})
c=pd.DataFrame({'patient_id':ids,'transport_barrier':r.binomial(1,.25,n),'food_insecurity':r.binomial(1,.22,n),'housing_insecurity':r.binomial(1,.15,n),'medication_barrier':r.binomial(1,.19,n),'primary_care_gap':r.binomial(1,.35,n),'followup_gap':r.binomial(1,.3,n)})
e=pd.DataFrame({'patient_id':ids,'ed_visits_180d':r.poisson(1.2,n),'primary_care_visits_180d':r.poisson(2,n)})
a=pd.DataFrame({'patient_id':ids,'admissions_180d':r.poisson(.45,n),'recent_discharge':r.binomial(1,.35,n)})
a.loc[a.recent_discharge==1,'admissions_180d']+=1
claims=pd.DataFrame({'patient_id':ids,'paid_180d':np.round(r.gamma(2,900,n)+e.ed_visits_180d*400+a.admissions_180d*1500,2)})
x=p.merge(c).merge(e).merge(a); sigmoid=lambda z:1/(1+np.exp(-z))
y=pd.DataFrame({'patient_id':ids})
y['hospitalization']=r.binomial(1,sigmoid(-3+.35*x.chronic_conditions+.55*x.admissions_180d+.25*x.ed_visits_180d+.45*x.medication_barrier+.4*x.primary_care_gap))
y['repeat_ed']=r.binomial(1,sigmoid(-2.8+.65*x.ed_visits_180d+.55*x.transport_barrier+.5*x.primary_care_gap+.35*x.housing_insecurity))
y['readmission']=np.where(x.recent_discharge==1,r.binomial(1,sigmoid(-3+.5*x.admissions_180d+.45*x.chronic_conditions+.7*x.followup_gap+.5*x.medication_barrier)),np.nan)
for name,df in [('patients',p),('care_coordination',c),('encounters',e),('admissions',a),('claims',claims),('outcomes',y)]: df.to_csv(D/f'{name}.csv',index=False)
resources=[('transport_barrier','Transportation navigation','Review transport options and confirm eligibility with coordinator.'),('food_insecurity','Food support navigation','Offer referral review for food assistance.'),('housing_insecurity','Housing support navigation','Review housing support referral options.'),('medication_barrier','Medication access support','Refer to pharmacy assistance team for review.'),('primary_care_gap','Primary care navigation','Offer scheduling assistance and confirm patient preferences.'),('followup_gap','Post-discharge outreach','Review discharge plan and follow-up scheduling with care team.')]
pd.DataFrame(resources,columns=['need','resource','action']).to_csv(D/'community_resources.csv',index=False)
(D/'policies.json').write_text(json.dumps([{'id':f'DEMO-{i+1:02}','title':title,'text':action+' Fictional workflow; verify consent, eligibility and availability before referral. Never discourage emergency care.'} for i,(_,title,action) in enumerate(resources)],indent=2))
