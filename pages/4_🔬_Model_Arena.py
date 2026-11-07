"""Legacy Streamlit view of the real model registry and measured benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.services.model_registry import describe_models,benchmark_model
from backend.services.classifier import dataset,predict_window

st.set_page_config(page_title='Model Arena · SatModClass',page_icon='🔬',layout='wide')
st.title('🔬 Model Arena — actual artifacts and measurements')
st.caption('Legacy reference. Only independently loaded models produce predictions. Historical evaluation is not a newly reproduced experiment.')
try:
    models=describe_models()
    st.dataframe(pd.DataFrame([{key:model.get(key) for key in ['id','name','status','version','parameters','accuracy','reason','error']} for model in models]),use_container_width=True)
    available=[m for m in models if m['status']=='available']
    if not available:
        st.error('No model artifact is available.');st.stop()
    sample_id=st.number_input('Demo sample ID',0,len(dataset()['signals'])-1,42)
    result=predict_window(dataset()['signals'][sample_id])
    st.metric('Actual 1D-CNN prediction',result['predictedModulation'],f"{100*result['confidence']:.2f}% uncalibrated confidence")
    st.bar_chart(pd.Series(result['probabilities'],name='Actual CNN probabilities'))
    st.info('Original-domain alternative models remain Not Trained. Separately labeled synthetic experiments have genuine checkpoints and held-out results; these do not establish performance on independent satellite captures. Optional neural inference requires compatible TensorFlow.')
    model_id=st.selectbox('Available model',[m['id'] for m in available])
    runs=st.number_input('Measured repetitions',5,100,30)
    if st.button('Run measured benchmark'):
        report=benchmark_model(model_id,int(sample_id),int(runs))
        a,b,c=st.columns(3)
        a.metric('Median inference',f"{report['medianMs']:.3f} ms")
        b.metric('P95 inference',f"{report['p95Ms']:.3f} ms")
        c.metric('Median preprocessing',f"{report['preprocessing']['medianMs']:.3f} ms")
        st.line_chart(report['latenciesMs'])
        st.json(report['hardware'])
        st.caption(report['scope'])
        import json
        st.download_button('Export measured result',json.dumps(report,indent=2),'satmod-benchmark.json','application/json')
except Exception as exc:
    st.error(f'Model service unavailable: {exc}')
