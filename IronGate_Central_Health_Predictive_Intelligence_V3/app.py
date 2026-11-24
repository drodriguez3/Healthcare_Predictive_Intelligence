import streamlit as st
import pandas as pd
from engine import build,search,referrals,TARGETS,BARRIERS,explain,interpret
st.set_page_config(page_title='IronGate | Healthcare Predictive Intelligence',layout='wide')
@st.cache_resource
def initialize(): return build()
x,metrics,fairness,models=initialize()
st.sidebar.markdown('### IRONGATE DATA SOLUTIONS')
page=st.sidebar.radio('Business solutions',['Coordinate Care & Interventions','Predict Patient Risk','Understand Barriers to Care','Technology & Validation'])
st.sidebar.caption('Independent concept demonstration · V3')
st.title('Healthcare Predictive Intelligence')
st.markdown('**Predict risk. Understand barriers. Coordinate care.**')
st.caption('Improving access and care for safety-net populations | Prepared for Central Health | IronGate Data Solutions LLC')
st.warning('Synthetic patients and fictional resources. Independent demonstration, not clinical software or affiliated with Central Health. Actual local ML; simulated Cortex services. Never use risk scores to deny care or discourage emergency visits.')
program=st.sidebar.multiselect('Program',sorted(x.program.unique()),default=sorted(x.program.unique()))
region=st.sidebar.multiselect('Region',sorted(x.region.unique()),default=sorted(x.region.unique()))
cohort=x[x.program.isin(program)&x.region.isin(region)]
threshold=st.sidebar.slider('Illustrative review threshold',.05,.8,.25,.05)
def label(s): return s.replace('_',' ').capitalize()
def documents(q):
    hits=search(q)
    if not hits: st.info('No matching fictional documents. Try transportation, food, housing, medication, primary care or discharge.')
    for d,s in hits:
        st.markdown(f"**{d['id']} · {d['title']}**"); st.write(d['text'])
def risk_view(c,t,detail=True):
    eligible=c.dropna(subset=[t+'_risk']); high=eligible[eligible[t+'_risk']>=threshold].sort_values(t+'_risk',ascending=False)
    a,b,d=st.columns(3); a.metric('Eligible demo patients',len(eligible)); b.metric('Flagged for review',len(high)); d.metric('Mean estimated risk',f'{eligible[t+"_risk"].mean():.1%}' if len(eligible) else 'N/A')
    st.caption(f'{TARGETS[t]} · synthetic 30-day outcome · held-out patients only. Readmission applies to recent discharges. Repeat ED does not mean avoidable ED.')
    if detail and len(high):
        st.subheader('Patient review queue'); st.dataframe(high[['patient_id','program','region',t+'_risk']].head(100),hide_index=True)
        pid=st.selectbox('Explore a synthetic patient',high.patient_id.head(100).tolist(),key='patient_'+page+t)
        row=high.loc[high.patient_id==pid].iloc[0]
        st.write(f'**Patient {pid}** · {row.program} · {row.region} · age band {row.age_band}')
        st.write({TARGETS[k]:f'{row[k+"_risk"]:.1%}' if pd.notna(row[k+'_risk']) else 'Not eligible' for k in TARGETS})
        st.write(f'Historical ED visits: {row.ed_visits_180d}; admissions: {row.admissions_180d}; primary care visits: {row.primary_care_visits_180d}.')
        st.write('Documented barriers: '+(', '.join(label(f) for f in BARRIERS if row[f]) or 'None recorded'))
        with st.expander('Why this model produced the score',expanded=True):
            e=explain(row,models[t]); st.dataframe(e,hide_index=True); st.caption('Exact logistic regression feature contributions relative to zero-valued inputs, in log odds. Positive values raise model log odds; negative values lower them. Not causal explanations.')
        st.dataframe(referrals(pd.DataFrame([row])).query('`Patients with documented need` > 0')[['resource','action']],hide_index=True)
    return high
if page=='Technology & Validation':
    st.header('Technology & Validation'); st.dataframe(metrics,hide_index=True)
    with st.expander('Subgroup performance'): st.dataframe(fairness,hide_index=True); st.caption('Descriptive synthetic subgroup metrics; not proof of fairness or clinical validity.')
    st.subheader('Proposed Snowflake architecture'); st.code('Governed healthcare data → Snowpark ML → risk scores\nCortex Analyst + Cortex Search → Cortex Agents → Streamlit → care team review',language=None)
    st.write('Local equivalents: scikit-learn, deterministic question routing and pandas, TF-IDF retrieval, rule-based orchestration. No live LLM or Snowflake connection.')
    with st.expander('Illustrative business value'):
        n=st.number_input('Annual eligible encounters',0,100000,1000); r=st.slider('Assumed incremental reduction (%)',0,30,5); cost=st.number_input('Avoided cost per encounter ($)',0,100000,1500); expense=st.number_input('Annual program cost ($)',0,10000000,30000); st.metric('Illustrative net value',f'${n*r/100*cost-expense:,.0f}'); st.caption('Scenario only; not a forecast or demonstrated savings.')
elif cohort.empty: st.info('Select at least one program and region.')
elif page=='Predict Patient Risk':
    st.header(page); t=st.selectbox('Outcome',list(TARGETS),format_func=TARGETS.get); risk_view(cohort,t)
elif page=='Understand Barriers to Care':
    st.header(page); st.bar_chart(cohort.groupby('region')[BARRIERS].mean()); st.caption('Fraction with documented barriers; not causal evidence.'); st.dataframe(referrals(cohort),hide_index=True); documents(st.text_input('Search fictional resource guidance','transportation'))
else:
    st.header('AI Care Coordination Copilot'); mode=st.radio('Explore', ['AI Care Coordination Copilot','Ask Your Data','Ask Your Knowledge'],horizontal=True)
    if mode=='Ask Your Knowledge': documents(st.text_input('Ask about fictional care resources','What transportation support is available?'))
    else:
        q=st.text_input('Ask a healthcare question','Which MAP Basic patients have elevated hospitalization risk and what outreach opportunities should we review?' if mode.startswith('AI') else 'Which regions have the greatest transportation barriers?',key=mode)
        st.caption('Supported question patterns: hospitalization, readmission, repeat ED, MAP Basic, MAP only, other safety-net, named region followed by “region”, and documented barriers. Local pattern matching, not unrestricted natural-language AI.')
        c,t=interpret(q,cohort); st.caption(f'Applied scope: {len(c)} patients · {TARGETS[t]} · selected sidebar filters plus recognized question terms.')
        if not len(c): st.info('No patients match the applied filters.')
        elif mode=='Ask Your Data':
            matches=[f for f in BARRIERS if any(w in q.lower() for w in {'transport_barrier':['transport'],'food_insecurity':['food'],'housing_insecurity':['housing'],'medication_barrier':['medication'],'primary_care_gap':['primary care'],'followup_gap':['follow-up','followup']}[f])]
            if matches: st.dataframe(c.groupby('region')[matches].agg(['sum','mean']),use_container_width=True); st.caption('Sum = patients with documented need; mean = fraction of cohort.')
            elif any(w in q.lower() for w in ['risk','hospitalization','readmission','repeat ed','emergency department']): risk_view(c,t,False)
            else: st.info('Question not supported by the local analytics router. Try a listed example.')
        else:
            if not any(w in q.lower() for w in ['risk','hospitalization','readmission','repeat ed','emergency department']): st.info('Choose a supported predictive objective in your question.')
            else:
                high=risk_view(c,t,False); st.subheader('Understand · documented needs'); rr=referrals(high); st.dataframe(rr[['resource','Patients with documented need','action']],hide_index=True)
                st.subheader('Act · outreach opportunities for review')
                for _,r in rr[rr['Patients with documented need']>0].head(3).iterrows():
                    st.markdown(f"**{r.resource}** — {r.action} ({r['Patients with documented need']} patients)")
                    hits=search(r.resource)
                    if hits: st.caption(f"Fictional evidence: {hits[0][0]['id']} · {hits[0][0]['title']}")
                st.caption('Counts overlap. Confirm consent, eligibility, availability and clinical appropriateness. No autonomous decisions.')
                with st.expander('Patient-level review and explanations'): risk_view(c,t)
                st.write('Pilot success measures: primary care access, completed follow-ups, successful referrals, patient experience and utilization. No improvement demonstrated.')
