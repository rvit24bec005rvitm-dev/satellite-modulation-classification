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
    for key in ['reportVersion','id','createdAt','source','captureId','capturePosition','inputHash','inputRetention','sampleRate','modelId','modelVersion','modelHash','preprocessing','predictedModulation','confidence','entropyBits','processingMs','probabilities','quantityTypes']:
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


def batch_pdf(report):
    """Bounded vector report from completed results; never reruns inference."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    output=io.BytesIO()
    c=canvas.Canvas(output,pagesize=(595,842),pageCompression=0)
    c.setTitle('SatModClass batch signal analysis')
    page=0;y=0
    def new_page():
        nonlocal page,y
        if page:c.showPage()
        page+=1
        c.setFillColor(HexColor('#0b1728'));c.rect(0,770,595,72,fill=1,stroke=0)
        c.setFillColor(HexColor('#00d9ff'));c.setFont('Helvetica-Bold',16)
        c.drawString(40,807,'SATMODCLASS / BATCH REPORT')
        c.setFont('Helvetica',9);c.setFillColor(HexColor('#ffffff'))
        c.drawString(40,787,'Computed results | satmod-technical-pdf-v1')
        c.setFillColor(HexColor('#52687b'));c.drawString(40,30,f'Page {page} | Raw softmax scores are uncalibrated')
        y=746
    def line(text,bold=False):
        nonlocal y
        for part in textwrap.wrap(str(text),100,break_long_words=True) or ['']:
            if y<60:new_page()
            c.setFillColor(HexColor('#10283c'));c.setFont('Helvetica-Bold' if bold else 'Helvetica',9)
            c.drawString(40,y,part);y-=14
    new_page()
    for key in ['reportVersion','jobId','createdAt','modelId','modelVersion','modelHash','parameters','classOrder','classCounts']:
        if key in report:line(f'{key}: {report[key]}')
    line('Preprocessing: each prediction records the saved scalar training mean/std; no unit-power normalization.')
    line('Quantity types: model outputs and entropy are computed; UTC times require capture metadata. SNR/labels are importer supplied.')
    capture_items=list(report.get('captures',{}).items())
    line(f'Capture metadata preview: {min(20,len(capture_items))} of {len(capture_items)} sources; full raw metadata is in JSON.')
    for record_id,meta in capture_items[:20]:
        line('CAPTURE '+record_id,True)
        for key in ['source','hash','sourceHash','sessionId','sampleCount','sampleRate','sampleRateKnown','centerFrequency','sourceOffset','datatype','scale','layout','iqOrder','license']:
            line(f'{key}: {meta.get(key)}')
    dataset=report.get('dataset')
    if dataset:
        line('LABELED DATASET',True)
        for key in ['id','hash','source','license','preprocessing','note']:line(f'{key}: {dataset.get(key)}')
        line(report.get('labelSource',''))
    metrics=report.get('metrics')
    if metrics:
        line('EVALUATION / IMPORTER LABELS',True)
        for key in ['samples','accuracy','precision','recall','macroF1','perClass','confusionMatrix','snrAccuracy','calibration']:
            if key in metrics:line(f'{key}: {metrics[key]}')
        line(report.get('calibration',''))
    predictions=report['predictions']
    if predictions:
        if y<240:new_page()
        line('WINDOW TIMELINE / RAW SOFTMAX',True)
        left,right=45,550;bottom,top=y-125,y-5
        c.setStrokeColor(HexColor('#9ab0c0'));c.rect(left,bottom,right-left,top-bottom,fill=0,stroke=1)
        c.setFont('Helvetica',8);c.drawString(20,top-2,'1.0');c.drawString(20,bottom,'0.0')
        # Evenly spaced preview, bounded independently of batch length. Actual
        # window indices are retained, rather than inventing uniform timestamps.
        count=min(512,len(predictions));indices=sorted({round(i*(len(predictions)-1)/max(1,count-1)) for i in range(count)})
        colors={'BPSK':'#008fa8','QPSK':'#7546b8','16-QAM':'#c37400'}
        for index in indices:
            p=predictions[index];x=left+(right-left)*index/max(1,len(predictions)-1)
            yy=bottom+(top-bottom)*p['confidence']
            c.setFillColor(HexColor(colors[p['predictedModulation']]))
            c.circle(x,yy,1.7,fill=1,stroke=0)
        c.setFillColor(HexColor('#10283c'));c.drawString(left,bottom-14,'Window index 0')
        c.drawRightString(right,bottom-14,str(len(predictions)-1));y=bottom-35
        line('BPSK: teal | QPSK: purple | 16-QAM: amber. Plot previews at most 512 actual windows.')
    line(f'WINDOW TABLE / first {min(200,len(predictions))} of {len(predictions)} observations',True)
    line('Index | sample offset | class | raw softmax | entropy bits | inference ms')
    for p in predictions[:200]:
        line(f"{p['index']} | {p['start']} | {p['predictedModulation']} | {p['confidence']:.6f} | {p['entropyBits']:.6f} | {p['processingMs']:.4f}")
    line('JSON/CSV exports retain every exact prediction, probability vector, capture identifier and available timestamp.')
    line('LIMITATIONS',True)
    line(report['note'])
    line('Closed-set three-class predictions do not establish a true satellite modulation label. No validated production OOD decision.')
    c.save()
    return output.getvalue()
