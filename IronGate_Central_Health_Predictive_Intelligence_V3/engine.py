from pathlib import Path
import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.feature_extraction.text import TfidfVectorizer
D=Path(__file__).parent/'data'
FEATURES=['chronic_conditions','transport_barrier','food_insecurity','housing_insecurity','medication_barrier','primary_care_gap','followup_gap','ed_visits_180d','primary_care_visits_180d','admissions_180d','recent_discharge']
TARGETS={'hospitalization':'Hospitalization','repeat_ed':'Repeat ED utilization','readmission':'Readmission'}
def load():
    x=pd.read_csv(D/'patients.csv')
    for f in ['care_coordination','encounters','admissions','claims','outcomes']: x=x.merge(pd.read_csv(D/f'{f}.csv'),on='patient_id',validate='one_to_one')
    return x

def build():
    x=load(); train,test=train_test_split(x,test_size=.3,random_state=42); test=test.copy(); metrics=[]; fairness=[]; models={}
    for t,label in TARGETS.items():
        tr=train.dropna(subset=[t]); te=test.dropna(subset=[t]); m=LogisticRegression(max_iter=1000).fit(tr[FEATURES],tr[t]); models[t]=m
        prob=m.predict_proba(te[FEATURES])[:,1]; test.loc[te.index,t+'_risk']=prob
        metrics.append({'Model':label,'Train':len(tr),'Test':len(te),'ROC AUC':roc_auc_score(te[t],prob),'Brier':brier_score_loss(te[t],prob)})
        for field in ['program','age_band','region']:
            for group,g in test.dropna(subset=[t+'_risk']).groupby(field):
                fairness.append({'Model':label,'Dimension':field,'Group':group,'N':len(g),'Observed':g[t].mean(),'Predicted':g[t+'_risk'].mean(),'Brier':brier_score_loss(g[t],g[t+'_risk']),'AUC':roc_auc_score(g[t],g[t+'_risk']) if g[t].nunique()>1 else np.nan})
    return test,pd.DataFrame(metrics),pd.DataFrame(fairness),models

def search(q):
    docs=json.loads((D/'policies.json').read_text()); v=TfidfVectorizer(stop_words='english'); mat=v.fit_transform([d['title']+' '+d['text'] for d in docs]); scores=(mat@v.transform([q]).T).toarray().ravel(); return [(docs[i],scores[i]) for i in np.argsort(-scores)[:3] if scores[i]>0]

def referrals(cohort):
    resources=pd.read_csv(D/'community_resources.csv'); resources['Patients with documented need']=[int(cohort[n].sum()) for n in resources.need]; return resources.sort_values('Patients with documented need',ascending=False)

BARRIERS=['transport_barrier','food_insecurity','housing_insecurity','medication_barrier','primary_care_gap','followup_gap']
def explain(row, model):
    # Exact additive contributions relative to zero-valued features, in log odds.
    return pd.DataFrame({'Factor':FEATURES,'Value':[row[f] for f in FEATURES],'Log-odds contribution':model.coef_[0]*row[FEATURES].astype(float).to_numpy()}).sort_values('Log-odds contribution',ascending=False)

def interpret(question, cohort, default_target='hospitalization'):
    q=question.lower(); t=default_target
    if 'readmission' in q: t='readmission'
    elif any(w in q for w in ['repeat ed','emergency department','ed utilization']): t='repeat_ed'
    elif 'hospitalization' in q or 'hospital admission' in q: t='hospitalization'
    if 'map basic' in q: cohort=cohort[cohort.program=='MAP Basic']
    elif 'map only' in q: cohort=cohort[cohort.program=='MAP']
    elif 'other safety-net' in q: cohort=cohort[cohort.program=='Other safety-net']
    for region in cohort.region.unique():
        if str(region).lower()+' region' in q: cohort=cohort[cohort.region==region]
    return cohort,t
