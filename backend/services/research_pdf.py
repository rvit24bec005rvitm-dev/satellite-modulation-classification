"""Deterministic vector PDF rendering of already computed research reports."""
import io
import json
import math
import textwrap
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


class TechnicalPDF:
    def __init__(self,title,identifier):
        self.output=io.BytesIO();self.title=title;self.identifier=identifier;self.page=0;self.y=0
        self.canvas=canvas.Canvas(self.output,pagesize=(595,842),pageCompression=0,invariant=1)
        self.canvas.setTitle('SatModClass '+title)
        self.new_page()

    def new_page(self):
        if self.page:self.canvas.showPage()
        self.page+=1;c=self.canvas
        c.setFillColor(HexColor('#ffffff'));c.rect(0,0,595,842,fill=1,stroke=0)
        c.setFillColor(HexColor('#0b1728'));c.rect(0,766,595,76,fill=1,stroke=0)
        c.setFillColor(HexColor('#00d9ff'));c.setFont('Helvetica-Bold',16)
        c.drawString(40,807,'SATMODCLASS / '+self.title.upper())
        c.setFillColor(HexColor('#d8e9f2'));c.setFont('Helvetica',8)
        c.drawString(40,785,'Technical research report | satmod-research-pdf-v1')
        c.setFillColor(HexColor('#52687b'));c.drawString(40,27,f'Page {self.page} | {self.identifier}')
        self.y=744

    def line(self,value,bold=False):
        for line in textwrap.wrap(str(value),100,break_long_words=True) or ['']:
            if self.y<55:self.new_page()
            self.canvas.setFillColor(HexColor('#10283c'))
            self.canvas.setFont('Helvetica-Bold' if bold else 'Helvetica',9)
            self.canvas.drawString(40,self.y,line);self.y-=14

    def section(self,title):
        if self.y<95:self.new_page()
        self.y-=6;self.line(title,True)
        self.canvas.setStrokeColor(HexColor('#afc7d5'));self.canvas.line(40,self.y+5,552,self.y+5);self.y-=6

    def values(self,report,keys):
        for key in keys:
            value=report.get(key,'Unavailable')
            if isinstance(value,(dict,list)):value=json.dumps(value,sort_keys=True,ensure_ascii=True)
            self.line(f'{key}: {value}')

    def chart(self,title,series,x_label,y_label,height=145):
        if self.y<height+105:self.new_page()
        self.line(title,True)
        c=self.canvas;left,right=66,550;top=self.y-5;bottom=top-height
        selected=[]
        for entry in series:
            x,y=list(entry['x']),list(entry['y'])
            if len(x)!=len(y):raise ValueError('PDF plot coordinates have inconsistent lengths.')
            indices=np.unique(np.linspace(0,len(x)-1,min(512,len(x))).astype(int)) if x else []
            points=[(float(x[i]),float(y[i])) if y[i] is not None and math.isfinite(float(y[i])) else None for i in indices]
            selected.append((entry,points))
        points=[point for _,group in selected for point in group if point is not None]
        if not points:self.line('Plot unavailable for this window.');return
        xmin=min(p[0] for p in points);xmax=max(p[0] for p in points)
        ymin=min(p[1] for p in points);ymax=max(p[1] for p in points)
        if xmin==xmax:xmin-=.5;xmax+=.5
        if ymin==ymax:
            margin=max(abs(ymin)*.05,.5);ymin-=margin;ymax+=margin
        else:
            margin=(ymax-ymin)*.04;ymin-=margin;ymax+=margin
        px=lambda x:left+(x-xmin)/(xmax-xmin)*(right-left)
        py=lambda y:bottom+(y-ymin)/(ymax-ymin)*(top-bottom)
        c.setStrokeColor(HexColor('#bfd0db'));c.rect(left,bottom,right-left,top-bottom,fill=0,stroke=1)
        c.setFont('Helvetica',7);c.setFillColor(HexColor('#52687b'))
        c.drawRightString(left-5,top-2,f'{ymax:.4g}');c.drawRightString(left-5,bottom,f'{ymin:.4g}')
        c.drawString(left,bottom-12,f'{xmin:.5g}');c.drawRightString(right,bottom-12,f'{xmax:.5g}')
        if ymin<=0<=ymax:c.line(left,py(0),right,py(0))
        for entry,group in selected:
            color=HexColor(entry.get('color','#008fa8'));c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(.8)
            if entry.get('scatter'):
                for p in group:
                    if p is not None:c.circle(px(p[0]),py(p[1]),1.1,fill=1,stroke=0)
            else:
                path=c.beginPath();drawing=False
                for p in group:
                    if p is None:drawing=False;continue
                    if drawing:path.lineTo(px(p[0]),py(p[1]))
                    else:path.moveTo(px(p[0]),py(p[1]));drawing=True
                c.drawPath(path)
        c.setFont('Helvetica',8);c.setFillColor(HexColor('#10283c'))
        c.drawCentredString((left+right)/2,bottom-25,x_label)
        c.saveState();c.translate(17,(top+bottom)/2);c.rotate(90);c.drawCentredString(0,0,y_label);c.restoreState()
        self.y=bottom-42
        self.line(' | '.join(entry['name'] for entry,_ in selected))

    def finish(self):
        self.canvas.save()
        return self.output.getvalue()


def analysis_pdf(report):
    explanations=report.get('explanations',[])
    pdf=TechnicalPDF('explanation report' if explanations else 'analysis report',report['id'])
    pdf.values(report,['reportVersion','id','createdAt','source','captureId','capturePosition','inputHash','inputRetention',
                       'sampleRate','modelId','modelVersion','modelHash','preprocessing','predictedModulation','confidence',
                       'entropyBits','processingMs','probabilities','quantityTypes'])
    pdf.line('Confidence and class scores are raw, uncalibrated softmax outputs; they are not calibrated correctness probabilities.')
    meta=report.get('captureMetadata',{})
    if meta:
        pdf.section('CAPTURE PROVENANCE')
        pdf.values(meta,['id','source','sessionId','hash','sourceHash','sampleCount','sampleRate','sampleRateKnown','sourceOffset',
                         'centerFrequency','datatype','layout','iqOrder','scale','license'])
        if meta.get('sigmf'):pdf.line('Complete original SigMF metadata is retained in the machine-readable analysis export.')
    analysis=report.get('analysis',{})
    pdf.section('COMPUTED SIGNAL METRICS')
    pdf.values(analysis,['sampleCount','sampleRate','signalPower','powerDb','rmsAmplitude','bandwidthHz','noisePower','estimatedSnrDb','qualityNote'])
    for key,value in analysis.get('features',{}).items():pdf.line(f'{key}: {value}')
    waveform=analysis.get('waveform',[])
    if waveform:
        pdf.chart('SELECTED I/Q WAVEFORM',[{'name':channel.upper(),'x':[p['sample'] for p in waveform],
                  'y':[p[channel] for p in waveform],'color':color} for channel,color in [('i','#008fa8'),('q','#7546b8')]],
                  'Relative capture sample index','Relative sample units')
    for explanation in explanations:
        pdf.new_page();pdf.section(explanation['method'].upper()+' / '+explanation['targetClass'])
        pdf.values(explanation,['createdAt','method','targetClass','targetQuantity','modelId','modelVersion','modelHash','inputHash',
                               'windowStart','windowLength','integrationSteps','baseline','inputLogit','baselineLogit','logitDifference',
                               'attributionSum','completenessError','processingMs','quantityUnit'])
        diagnostics=explanation.get('diagnostics',{})
        pdf.line(diagnostics.get('quantityDescription',''))
        values=explanation['samples']
        pdf.chart('SIGNED I/Q CLASS-LOGIT ATTRIBUTION' if explanation['method']=='integrated-gradients' else 'SIGNED I/Q LOCAL LOGIT DERIVATIVES',
                  [{'name':channel.upper(),'x':[p['sample'] for p in values],'y':[p[channel] for p in values],'color':color}
                   for channel,color in [('i','#008fa8'),('q','#7546b8')]],
                  'Model window sample index',explanation.get('quantityUnit','Model attribution'))
        pdf.section('SIGNED CHANNEL SUMMARIES')
        for channel,totals in diagnostics.get('channels',{}).items():pdf.line(channel.upper()+': '+json.dumps(totals,sort_keys=True))
        pdf.section('MODEL-RANKED SAMPLE REGIONS')
        pdf.line(diagnostics.get('regionRule',''))
        for region in diagnostics.get('importantRegions',[]):
            pdf.line(f"Samples [{region['startSample']}, {region['endSample']}) | absolute importance {region['absoluteImportance']:.6g} | share {region['shareOfAbsoluteImportance']:.2%}")
        if not diagnostics.get('importantRegions'):pdf.line('No nonzero attribution region is available.')
        for limitation in explanation['limitations']:pdf.line('Limitation: '+limitation)
    if explanations:
        from backend.services.saved_explanations import method_comparison
        pdf.section('METHOD RANK COMPARISON')
        for target in dict.fromkeys(e['targetClass'] for e in explanations):
            comparison=method_comparison(explanations,target)
            pdf.values(comparison,['status','targetClass','measure','spearmanRho','note'])
        pdf.line('Machine-readable JSON retains all 128 signed I/Q values, raw input values and diagnostics for every stored explanation.')
    pdf.section('SCIENTIFIC LIMITATIONS')
    for note in report.get('limitations',[]):pdf.line(note)
    pdf.line('No SNR is inferred from arbitrary samples. No validated production OOD decision or physical-causality claim is made.')
    return pdf.finish()


def comparison_pdf(report):
    pdf=TechnicalPDF('signal comparison',report['id'])
    pdf.values(report,['reportVersion','id','createdAt','modelId','modelVersion','modelHash','processingMs','sampleRates','ratesEqual',
                       'sampleClocksCompatible','normalizedAxes','lengthsEqual','overlapSamples','sampleMse','powerDifferenceDb','confidenceDifference'])
    for name in ['first','second']:
        side=report[name];pdf.section(name.upper()+' SELECTED RECORDING')
        pdf.values(side,['captureId','start','length','position','inputHash','modelWindowHash','inputRetention'])
        pdf.values(side['metadata'],['source','sessionId','hash','sourceHash','sampleRate','sampleRateKnown','sourceOffset','datatype','scale','license'])
        pdf.section(name.upper()+' PREDICTION')
        pdf.values(side['prediction'],['status','reason','predictionSource','predictedModulation','confidence','entropyBits','probabilities','processingMs','timingScope','preprocessing'])
        pdf.section(name.upper()+' COMPUTED QUALITY')
        pdf.values(side['analysis'],['signalPower','powerDb','rmsAmplitude','bandwidthHz','estimatedSnrDb','qualityNote'])
    plots=report['plots']
    for name in ['first','second']:
        waveform=plots[name]['waveform']
        pdf.chart(name.upper()+' I/Q WAVEFORM',[
            {'name':channel.upper(),'x':[p['x'] for p in waveform],'y':[p[channel] for p in waveform],'color':color}
            for channel,color in [('i','#008fa8'),('q','#7546b8')]],plots['timeUnit'],'Relative sample units')
    pdf.chart('RECEIVED CONSTELLATIONS',[
        {'name':name.title(),'x':[p['i'] for p in plots[name]['constellation']],
         'y':[p['q'] for p in plots[name]['constellation']],'scatter':True,'color':color}
        for name,color in [('first','#008fa8'),('second','#7546b8')]],'I / relative sample units','Q / relative sample units')
    pdf.chart('SPECTRAL DENSITY COMPARISON',[
        {'name':name.title(),'x':[p['frequency'] for p in plots[name]['spectrum']],
         'y':[p['psd'] for p in plots[name]['spectrum']],'color':color}
        for name,color in [('first','#008fa8'),('second','#7546b8')]],'Frequency / '+plots['frequencyUnit'],plots['psdUnit'])
    pdf.chart('PHASE TRAJECTORIES',[
        {'name':name.title(),'x':[p['x'] for p in plots[name]['phase']],
         'y':[p['phaseRad'] for p in plots[name]['phase']],'color':color}
        for name,color in [('first','#008fa8'),('second','#7546b8')]],plots['timeUnit'],'Independently unwrapped phase / rad')
    distribution=plots['amplitudeDistribution']
    pdf.chart('COMMON-BIN AMPLITUDE DISTRIBUTIONS',[
        {'name':name.title(),'x':[p['value'] for p in distribution[name]],
         'y':[p['fraction'] for p in distribution[name]],'color':color}
        for name,color in [('first','#008fa8'),('second','#7546b8')]],'Amplitude / relative sample units','Fraction of samples / bin')
    pdf.line(distribution['note'])
    differences=report['attributionDifferences'];pdf.section('ATTRIBUTION DIFFERENCES')
    pdf.values(differences,['status','reason','note'])
    for comparison in differences['comparisons']:
        pdf.values(comparison,['method','targetClass','quantityUnit','firstAnalysisId','secondAnalysisId','meanAbsoluteDifference'])
        pdf.chart('FIRST MINUS SECOND / '+comparison['method'],[
            {'name':channel.upper(),'x':[p['sample'] for p in comparison['samples']],
             'y':[p[channel] for p in comparison['samples']],'color':color}
            for channel,color in [('i','#008fa8'),('q','#7546b8')]],'Relative model-window sample index',comparison['quantityUnit'])
    pdf.section('SCIENTIFIC LIMITATIONS')
    for note in report['limitations']:pdf.line(note)
    pdf.line('Plots draw at most 512 actual report samples per series. JSON retains the complete bounded plot arrays and exact input/capture/model identities.')
    return pdf.finish()
