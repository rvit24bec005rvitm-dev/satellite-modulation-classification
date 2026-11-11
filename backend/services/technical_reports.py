"""Versioned analysis exports. PDF graphics use stored computed samples only."""
import csv
import io
import json
import textwrap
from backend.services import workspace_store as store


def load(record_id):
    with store.LOCK:
        store.get('analysis',record_id)
        return json.loads(store.file_path(record_id,'json').read_text())


def export(record_id,format_name):
    report=load(record_id)
    if format_name=='json':return json.dumps(report,indent=2,allow_nan=False).encode(),'application/json'
    if format_name=='csv':
        stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['field','value'])
        for key,value in report.items():
            if key in ['analysis','samples']:continue
            text=json.dumps(value) if isinstance(value,(dict,list)) else str(value)
            if text.startswith(('=','+','-','@')):text="'"+text
            writer.writerow([key,text])
        for key,value in report.get('analysis',{}).get('features',{}).items():writer.writerow([key,value])
        return stream.getvalue().encode(),'text/csv'
    if format_name!='pdf':raise ValueError('Use json, csv or pdf.')
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    output=io.BytesIO();c=canvas.Canvas(output,pagesize=(595,842));c.setTitle('SatModClass technical analysis')
    y=800
    def line(text):
        nonlocal y
        for part in textwrap.wrap(str(text),95) or ['']:
            if y<55:c.showPage();y=800
            c.setFillColor(HexColor('#10283c'));c.setFont('Helvetica',9);c.drawString(42,y,part);y-=14
    c.setFillColor(HexColor('#0b1728'));c.rect(0,755,595,87,fill=1,stroke=0)
    c.setFillColor(HexColor('#00d9ff'));c.setFont('Helvetica-Bold',18);c.drawString(42,804,'SATMODCLASS / ANALYSIS REPORT');y=775
    c.setFillColor(HexColor('#ffffff'));c.setFont('Helvetica',9);c.drawString(42,y,'Computed signal intelligence | Version satmod-analysis-v2');y=730
    for key in ['id','createdAt','source','captureId','modelId','modelVersion','modelHash','preprocessing','predictedModulation','confidence','entropyBits','processingMs','probabilities']:
        line(f'{key}: {report.get(key,"Unavailable")}')
    line('Confidence and probabilities are raw, uncalibrated softmax outputs.')
    for name,value in report.get('captureMetadata',{}).items():
        if name!='sigmf':line(f'Capture {name}: {value}')
    for name,value in report.get('analysis',{}).get('features',{}).items():line(f'{name}: {value}')
    waveform=report.get('analysis',{}).get('waveform',[])
    if waveform:
        if y<200:c.showPage();y=790
        line('I/Q waveform - relative sample units; capture sample index')
        top=y;bottom=y-110;maximum=max(abs(p[k]) for p in waveform for k in ['i','q']) or 1
        c.setStrokeColor(HexColor('#c7d8e3'));c.line(42,(top+bottom)/2,550,(top+bottom)/2)
        for channel,color in [('i','#008fa8'),('q','#7546b8')]:
            c.setStrokeColor(HexColor(color));path=c.beginPath()
            for i,p in enumerate(waveform):
                x=42+508*i/max(1,len(waveform)-1);yy=(top+bottom)/2+p[channel]/maximum*50
                if i==0:path.moveTo(x,yy)
                else:path.lineTo(x,yy)
            c.drawPath(path)
        y=bottom-24
    for limitation in report.get('limitations',[]):line('Limitation: '+limitation)
    line('SNR unavailable unless independently supplied. No validated production OOD decision.')
    c.save();return output.getvalue(),'application/pdf'
