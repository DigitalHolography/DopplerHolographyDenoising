"""Run selected one-factor-at-a-time experiments, then compare curves and reports.

Training uses existing preparations once, without copying the video caches.
An optional, separate evaluation dataset is prepared and scored only after training.
"""
import argparse
from dataclasses import asdict
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import noise2time as api


def variants(base, validation_video, experiments=None):
    """Every alternative changes one conceptual factor relative to baseline."""
    if base.input_mode != 'patched' or not base.convlstm or not base.brightness_correction:
        raise ValueError('Baseline must use patched input, ConvLSTM and brightness correction')
    if base.split_mode != 'mixed' or base.validation_records:
        raise ValueError('Baseline must use mixed validation with no validation_records')
    changes = dict(baseline={}, no_patch={'input_mode':'no_patch'},
                   temporal_split={'split_mode':'temporal'},
                   video_validation={'split_mode':'record','validation_records':[validation_video]},
                   no_convlstm={'convlstm':False}, no_brightness={'brightness_correction':False})
    defaults=list(changes)
    changes.update(l1={'objective':'l1'},l2={'objective':'l2'},
                   l1_grad_hessian={'objective':'l1_grad_hessian'},
                   l2_grad_hessian={'objective':'l2_grad_hessian'})
    selected=defaults if experiments is None else experiments
    if not isinstance(selected,list) or not selected or any(not isinstance(n,str) for n in selected):
        raise ValueError('experiments must be a nonempty list of experiment names')
    if len(set(selected))!=len(selected): raise ValueError('Duplicate experiment names')
    unknown=set(selected)-changes.keys()
    if unknown: raise ValueError(f'Unknown experiments: {sorted(unknown)}. Choices: {list(changes)}')
    return {name: dict(asdict(base), **changes[name]) for name in selected}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def log_event(path, message):
    """Append a timestamped line and echo it so console and file stay aligned."""
    line = f'[{now()}] {message}'
    print(line, flush=True)
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def write_status(path, **values):
    target = Path(path)
    current = read_json(target) if target.exists() else {}
    if values.get('status') in ('running', 'complete'):
        current.pop('error', None)
    current.update(values, updated_at=now())
    api.write_json(target, current)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict): result.update(flatten(item,name))
        elif isinstance(item, (int,float)) or item is None: result[name] = item
    return result


def epoch_evaluation(record_path, folder, group, device_name):
    """Score retained weights on one measure; cache results with input provenance.

    Old runs provide only best/last. Missing epochs stay missing: interpolating
    them would suggest measurements that were never performed.
    """
    support=api.load_sibling('training_monitor')
    record=api.Record(record_path)
    destination=folder/'reports'/group/record.name
    destination.mkdir(parents=True,exist_ok=True)
    cache_path=destination/'epoch_metrics.json'
    cached=read_json(cache_path) if cache_path.exists() else {'rows':[]}
    input_signature=dict(frames=api.sha256(record.path/'frames.npy'),
                         metadata=api.sha256(record.path/'metadata.json'),
                         monitor_code=api.sha256(Path(support.__file__)),
                         inference_code=api.sha256(Path(api.__file__)))
    rows=[]
    device=api.get_device(device_name)
    seen=set()
    paths=sorted((folder/'runs/checkpoints').glob('epoch_*.pt'))
    paths += [folder/'runs/best.pt',folder/'runs/last.pt']
    for path in paths:
        if not path.exists(): continue
        saved=api.torch.load(path,map_location='cpu',weights_only=True)
        epoch=int(saved['epoch'])
        if epoch in seen: continue
        seen.add(epoch)
        cfg=api.Config(**saved['config'])
        monitor=support.Monitor(record,cfg.history,api,2)
        signature=dict(input_signature,checkpoint=api.sha256(path),masks=monitor.provenance)
        previous=next((row for row in cached['rows'] if row['epoch']==epoch and row['sources']==signature),None)
        if previous:
            rows.append(previous)
            del saved
            continue
        log_event(folder/'evaluation.log',f'{group}/{record.name}: scoring epoch {epoch} ({path.name})')
        array=folder/'denoised'/group/record.name/'denoised.npy'
        provenance_path=array.with_suffix('.json')
        provenance=read_json(provenance_path) if provenance_path.exists() else {}
        reuse=(array.exists() and provenance.get('checkpoint_sha256')==signature['checkpoint']
               and provenance.get('record_sha256')==input_signature['frames'])
        if reuse:
            # Best-checkpoint inference already exists for the regional report.
            predictions=np.load(array,mmap_mode='r')
            if predictions.shape!=record.frames.shape: raise ValueError('Cached denoised shape mismatch')
            for index in range(cfg.history,len(predictions)): monitor.update(index,predictions[index])
            del predictions
        else:
            model=api.Noise2Time(cfg.base_channels,cfg.convlstm).to(device)
            model.load_state_dict(saved['model'])
            api.export_denoised(model,record,cfg,device,frame_callback=monitor.update)
            del model
        rows.append(dict(epoch=epoch,diagnostics=monitor.result(),sources=signature))
        # Save after each checkpoint so an interrupted evaluation can resume.
        api.write_json(cache_path,dict(rows=sorted(rows,key=lambda row:row['epoch'])))
        del saved
    api.write_json(cache_path,dict(rows=sorted(rows,key=lambda row:row['epoch'])))


def epoch_comparison(output, destination, group, record, strategies):
    """Write per-measure epoch tables and curves, never filling absent epochs."""
    rows=[]
    for name in strategies:
        path=output/name/'reports'/group/record/'epoch_metrics.json'
        if path.exists():
            for row in read_json(path)['rows']:
                rows.append(dict(strategy=name,epoch=row['epoch'],**flatten(row['diagnostics'])))
    if not rows: return '<p>No checkpoint diagnostics have been computed for this measurement yet.</p>'
    keys=sorted({key for row in rows for key in row}-{'strategy','epoch'})
    with (destination/'epoch_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['strategy','epoch']+keys)
        writer.writeheader();writer.writerows(rows)
    plots=destination/'epochs';plots.mkdir(exist_ok=True)
    figures=[]
    for index,key in enumerate(keys):
        if key=='frames_scored': continue
        fig=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        for name in strategies:
            selected=sorted((r for r in rows if r['strategy']==name and r.get(key) is not None),key=lambda r:r['epoch'])
            if not selected: continue
            # NaNs break lines over missing epochs instead of interpolating gaps.
            values={r['epoch']:r[key] for r in selected}
            x=list(range(min(values),max(values)+1))
            ax.plot(x,[values.get(epoch,np.nan) for epoch in x],'.-',label=name)
        ax.set(title=key,xlabel='Checkpoint epoch');ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
        filename=f'metric_{index:03d}.png'
        temporary=plots/(filename+'.tmp')
        fig.savefig(temporary,format='png',dpi=120);fig.clear();temporary.replace(plots/filename)
        figures.append(f'<details><summary>{html.escape(key)}</summary><img style="max-width:100%" src="epochs/{filename}"></details>')
    return ('<p>Diagnostics computed on this measurement with retained epoch weights. '
            'Missing checkpoints are not interpolated. Older runs may have only best and last epochs. '
            'These are inference diagnostics, not new training or validation losses.</p>'
            '<p><a href="epoch_metrics.csv">Epoch metrics CSV</a></p>'+''.join(figures))


def comparison(output, plan):
    """Rebuild an index and one comparison plot per logged numerical metric."""
    plots = output/'comparison'; plots.mkdir(exist_ok=True)
    all_rows, status_rows, final_rows = [], [], []
    for name in plan['variants']:
        folder = output/name
        history = folder/'runs/metrics.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                all_rows.append(dict(strategy=name, **flatten(row)))
        status = read_json(folder/'status.json') if (folder/'status.json').exists() else {'status':'pending'}
        links = []
        for report in sorted(folder.glob('reports/*/*/report.html')):
            relative = report.relative_to(output).as_posix()
            links.append(f'<a href="{html.escape(relative)}">{html.escape(str(report.parent.relative_to(folder/"reports")))}</a>')
            metrics=read_json(report.parent/'metrics.json')
            final_rows.append(dict(strategy=name,group=report.parent.parent.name,record=report.parent.name,
                                   **flatten(dict(background=metrics['background'],regions=metrics['regions']))))
        if (folder/'runs/metrics.png').exists():
            links.insert(0,f'<a href="{name}/runs/metrics.png">Training curves</a>')
        status_rows.append(f'<tr><td>{name}</td><td>{html.escape(status["status"])}</td><td>{" | ".join(links)}</td><td>{html.escape(status.get("error",""))}</td></tr>')
    keys = sorted({key for row in all_rows for key in row} - {'strategy','epoch'})
    final_keys=sorted({key for row in final_rows for key in row}-{'strategy','group','record'})
    with (output/'final_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['strategy','group','record']+final_keys)
        writer.writeheader();writer.writerows(final_rows)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['strategy','epoch']+keys);writer.writeheader();writer.writerows(all_rows)
    figures=[]
    for index,key in enumerate(keys):
        if key.endswith('frames_scored'): continue
        fig=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        for name in plan['variants']:
            rows=[row for row in all_rows if row['strategy']==name and row.get(key) is not None]
            if rows: ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],'.-',label=name)
        ax.set(title=key, xlabel='Epoch');ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,'Metric undefined for these data (see CSV null values)',
                    ha='center',va='center',transform=ax.transAxes)
        filename=f'metric_{index:03d}.png'
        temporary_plot=plots/(filename+'.tmp')
        fig.savefig(temporary_plot,format='png',dpi=120);fig.clear()
        temporary_plot.replace(plots/filename)
        figures.append(f'<details><summary>{html.escape(key)}</summary><img loading="lazy" src="comparison/{filename}"></details>')
    report_tables=[]
    for group,record in sorted({(row['group'],row['record']) for row in final_rows}):
        rows=[row for row in final_rows if row['group']==group and row['record']==record]
        headings=''.join(f'<th>{html.escape(row["strategy"])}</th>' for row in rows)
        body=[]
        for key in final_keys:
            cells=''.join('<td>'+('—' if row.get(key) is None else f'{row[key]:.6g}')+'</td>' for row in rows)
            body.append(f'<tr><th>{html.escape(key)}</th>{cells}</tr>')
        # Keep development and unseen records separate even when names match.
        destination=output/'comparison'/group/record
        destination.mkdir(parents=True,exist_ok=True)
        with (destination/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['strategy','group','record']+final_keys)
            writer.writeheader();writer.writerows(rows)
        report_links=''.join(
            f'<li><a href="../../../{html.escape(row["strategy"])}/reports/'
            f'{html.escape(group)}/{html.escape(record)}/report.html">'
            f'{html.escape(row["strategy"])}</a></li>' for row in rows)
        title=html.escape(group+' / '+record)
        epoch_plots=epoch_comparison(output,destination,group,record,plan['variants'])
        page=('<!doctype html><meta charset="utf-8"><title>'+title+' comparison</title>'
              '<style>body{font:16px system-ui;margin:40px}td,th{padding:8px;text-align:left;'
              'border-bottom:1px solid #ddd}table{font-size:14px}</style>'
              f'<h1>{title}</h1><p>Comparison of saved best checkpoints for this measurement.</p>'
              '<p>Background variability reduction alone does not establish denoising accuracy. '
              'Inspect regional signal preservation and the individual reports.</p>'
              '<p><a href="../../../index.html">Benchmark overview</a> | '
              '<a href="metrics.csv">Measurement metrics (CSV)</a></p>'
              '<h2>Strategy reports</h2><ul>'+report_links+'</ul>'
              f'<h2>Metrics</h2><div style="overflow-x:auto"><table><tr><th>Metric</th>{headings}</tr>'
              +''.join(body)+'</table></div><h2>Measurement diagnostics by epoch</h2>'+epoch_plots)
        temporary=destination/'report.html.tmp'
        temporary.write_text(page,encoding='utf-8');temporary.replace(destination/'report.html')
        relative=(destination/'report.html').relative_to(output).as_posix()
        report_tables.append(f'<p><a href="{html.escape(relative)}">{title}</a></p>')
    content='''<!doctype html><meta charset="utf-8"><title>Noise2Time benchmark</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%}summary{padding:12px;cursor:pointer}code{background:#eee}</style>
<h1>Noise2Time: one-factor-at-a-time benchmark</h1>
<p>Each alternative changes one factor from baseline. Epoch diagnostics use only the first development video;
they are not independent test scores. Separate evaluation videos never select checkpoints. Reports use best.pt.</p>
<p>Lower background variability alone does not establish accuracy. Inspect vessel means, waveform preservation,
and regional reports. Validation losses across different splits use different data and are not directly comparable.</p>
<p>Temporal splitting isolates training and validation target/history frames. Validation donors come from training
cycles; peak timing, prepared brightness, intensity scale and spatial masks are shared calibration.</p>
<p><a href="comparison.csv">All epoch metrics (CSV)</a> | <a href="final_comparison.csv">Final report metrics (CSV)</a> | <a href="plan.json">Exact experiment plan</a></p>
<table><tr><th>Strategy</th><th>Status</th><th>Reports</th><th>Error</th></tr>'''+''.join(status_rows)+'</table><h2>Best-checkpoint comparisons by measurement</h2>'+''.join(report_tables)+'<h2>Compare every logged training metric</h2>'+''.join(figures)
    temporary=output/'index.html.tmp';temporary.write_text(content,encoding='utf-8');temporary.replace(output/'index.html')


def run_child(command, log, status_path=None, phase='training'):
    """Stream a child process to console and its per-strategy log."""
    log = Path(log)
    log_event(log, 'Running: ' + ' '.join(map(str, command)))
    with log.open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, *map(str, command)], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        if status_path:
            write_status(status_path, status='running', phase=phase, pid=process.pid)
        for line in process.stdout:
            # tqdm uses carriage returns to redraw one terminal line. A pipe has
            # no cursor, so expose each redraw as an ordinary progress line.
            updates = [part.strip() for part in line.replace('\r', '\n').splitlines() if part.strip()]
            for update in updates:
                stream.write(update + '\n'); stream.flush()
                print(update, flush=True)
                if status_path:
                    write_status(status_path, status='running', phase=phase, last_output=update[-500:])
        returncode = process.wait()  # Popen.wait() returns an int, not CompletedProcess.
        process.stdout.close()
    if returncode:
        raise RuntimeError(f'Command exited {returncode}; see {log}')


def prepare_evaluation(dataset, output, reference, api, names=None):
    workflow=api.load_sibling('dataset_workflow')
    target=output/'evaluation_data'
    args=['prepare','--input',str(dataset),'--output',str(target),'--skip-existing',
          '--fps',str(reference['fps'])]
    if names: args += ['--measures', *names]
    for key,value in reference['circle'].items(): args += ['--'+key,str(value)]
    for key in ('peak_min_hz','peak_max_hz'): args += ['--'+key.replace('_','-'),str(reference[key])]
    if reference['input_mode']=='avi': args.append('--avi')
    if api.main(args): raise ValueError('Evaluation preparation failed; inspect evaluation_data/preparation_summary.json')
    return [target/'prepared'/folder.name for folder in workflow.measurements(dataset, names)]


def final_report(record, folder, group, device, force=False):
    workflow=api.load_sibling('dataset_workflow')
    metadata=read_json(record/'metadata.json');source=Path(metadata['dataset_measure'])
    destination=folder/'reports'/group/record.name
    checkpoint=folder/'runs/best.pt'
    masks=dict(artery=workflow.manual_mask(source,'artery'),vein=workflow.manual_mask(source,'vein'),
               choroid=workflow.choroidal_masks(source)[0][0])
    signature=dict(checkpoint_sha256=api.sha256(checkpoint),frames_sha256=api.sha256(record/'frames.npy'),
                   masks={key:api.sha256(path) for key,path in masks.items()})
    if (destination/'report.html').exists() and force:
        # Preserve the old metric definition before regenerating in place.
        backup = folder/'legacy_reports'/group/record.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(f'Legacy report backup already exists: {backup}')
        destination.rename(backup)
    if (destination/'report.html').exists():
        if not (destination/'benchmark_sources.json').exists() or read_json(destination/'benchmark_sources.json')!=signature:
            raise ValueError(f'Existing report sources changed: {destination}')
        return
    denoised_root=folder/'denoised'/group
    array=denoised_root/record.name/'denoised.npy'
    if not array.exists():
        api.main(['denoise','--record',str(record),'--checkpoint',str(checkpoint),
                  '--output',str(denoised_root),'--device',device])
    provenance=read_json(array.with_suffix('.json'))
    if provenance['checkpoint_sha256']!=api.sha256(checkpoint) or provenance['record_sha256']!=api.sha256(record/'frames.npy'):
        raise ValueError('Existing denoised output does not match checkpoint or prepared input')
    api.main(['evaluate','--record',str(record),'--denoised',str(array),'--output',str(destination),
              '--retinal-artery-mask',str(masks['artery']),
              '--retinal-vein-mask',str(masks['vein']),
              '--choroidal-masks',str(masks['choroid'])])
    api.write_json(destination/'benchmark_sources.json',signature)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',help='Existing workflow prepared/ directory (or a single prepared record)')
    parser.add_argument('--output',required=True,help='New benchmark directory; resume/report commands reuse it')
    parser.add_argument('--config',help='Baseline JSON configuration; all variants inherit it')
    parser.add_argument('--experiments',help='JSON with an experiments list; selects one-factor variants for a new benchmark')
    parser.add_argument('--validation-video',help='Video excluded from training in the video-validation variant only')
    parser.add_argument('--measures',nargs='+',help='Select development measurements from the prepared folder')
    parser.add_argument('--evaluation-input',help='Separate dataset folder, now or later with --evaluate-only')
    parser.add_argument('--evaluation-measures',nargs='+',
                        help='Measurement names to evaluate; without --evaluation-input, select existing development records')
    parser.add_argument('--development-measures',nargs='+',
                        help='Development measurements to report when --evaluation-input is also supplied')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--force-evaluation',action='store_true',
                        help='Regenerate existing reports, moving them to legacy_reports/ first')
    parser.add_argument('--epoch-metrics', action=argparse.BooleanOptionalAction, default=True,
                        help='Compute per-epoch inference diagnostics (default: enabled; use --no-epoch-metrics to disable)')
    parser.add_argument('--dry-run',action='store_true',help='Validate all configurations and write the plan without training')
    args=parser.parse_args(argv);output=Path(args.output).resolve()
    if (output/'plan.json').exists():
        if not any((args.resume,args.evaluate_only,args.report_only)):
            raise ValueError('Benchmark exists; use --resume, --evaluate-only or --report-only')
        plan=read_json(output/'plan.json')
        if args.config or args.prepared or args.validation_video or args.measures or args.experiments:
            raise ValueError('Existing plan supplies config, records and validation video; omit those flags')
    else:
        if args.resume or args.evaluate_only or args.report_only:
            raise ValueError('No benchmark plan exists')
        if not args.prepared: parser.error('--prepared is required for a new benchmark')
        paths=api.resolve_training_records([args.prepared])
        if args.measures:
            missing=set(args.measures)-{path.name for path in paths}
            if missing: raise ValueError(f'Unknown prepared measurements: {sorted(missing)}')
            paths=[path for path in paths if path.name in args.measures]
        records=[api.Record(path) for path in paths]
        if len({r.name for r in records})!=len(records) or len({r.frames.shape[1:] for r in records})!=1:
            raise ValueError('Development records need unique names and identical image dimensions')
        if any(r.metadata.get('diaphragm_mask_applied') is not True for r in records):
            raise ValueError('Benchmark requires diaphragm-masked dataset preparations')
        if len({r.metadata['input_mode'] for r in records})!=1:
            raise ValueError('Do not mix raw and AVI preparations')
        base=api.Config(**read_json(Path(args.config))) if args.config else api.Config()
        base.validate()
        selected=args.validation_video or records[-1].name
        experiment_spec=read_json(Path(args.experiments)) if args.experiments else None
        if experiment_spec is not None and (not isinstance(experiment_spec,dict) or set(experiment_spec)!={'experiments'}):
            raise ValueError('Experiment JSON must contain exactly one key: experiments')
        configs=variants(base,selected,experiment_spec['experiments'] if experiment_spec is not None else None)
        if 'video_validation' in configs:
            if len(paths)<2: raise ValueError('At least two development videos are required for video-disjoint validation')
            if selected not in {r.name for r in records} or selected==records[0].name:
                raise ValueError('Validation video must be present and differ from the first preview video')
        for name,values in configs.items():
            cfg=api.Config(**values);cfg.validate()
            train,valid=api.split_samples(records,cfg)
            if cfg.samples_per_epoch<len({i for i,t in train}): raise ValueError('Increase samples_per_epoch')
            for stage,pool in (('train',train),('valid',valid)):
                for i,t in pool[:1]: api.replacement(records[i],t,cfg,np.random.default_rng(cfg.seed),stage)
            print(f'Preflight {name}: {len(train)} training targets, {len(valid)} validation targets',flush=True)
        api.load_sibling('training_monitor').Monitor(records[0],base.history,api,2)
        plan=dict(records=[str(p.resolve()) for p in paths],variants=configs,
                  preview_record=records[0].name,validation_video=selected,
                  reference=records[0].metadata,
                  prepared_hashes={r.name:api.sha256(r.path/'metadata.json') for r in records})
        output.mkdir(parents=True,exist_ok=False);api.write_json(output/'plan.json',plan)
    if args.report_only:
        comparison(output,plan);return 0
    if args.dry_run:
        comparison(output,plan);return 0
    for path in map(Path,plan['records']):
        if api.sha256(path/'metadata.json')!=plan['prepared_hashes'][path.name]:
            raise ValueError('Prepared metadata changed since benchmark planning')
    evaluation=[]
    development=[Path(plan['records'][0])]
    if args.development_measures:
        available={Path(p).name:Path(p) for p in plan['records']}
        missing=set(args.development_measures)-available.keys()
        if missing: raise ValueError(f'Unknown development measurements: {sorted(missing)}')
        development=[available[name] for name in dict.fromkeys(args.development_measures)]
    elif args.evaluation_measures and not args.evaluation_input:
        available={Path(p).name:Path(p) for p in plan['records']}
        missing=set(args.evaluation_measures)-available.keys()
        if missing: raise ValueError(f'Unknown development measurements: {sorted(missing)}; use --evaluation-input for another dataset')
        development=[available[name] for name in dict.fromkeys(args.evaluation_measures)]
    if args.evaluation_input:
        # Check source identities before preparing or evaluating any external data.
        workflow=api.load_sibling('dataset_workflow')
        train_hashes={read_json(Path(p)/'metadata.json')['source_sha256'] for p in plan['records']}
        for source in workflow.measurements(args.evaluation_input,args.evaluation_measures):
            if api.sha256(workflow.single_h5(source)) in train_hashes:
                raise ValueError(f'Evaluation source also occurs in development data: {source}')
        evaluation=prepare_evaluation(args.evaluation_input,output,plan['reference'],api,args.evaluation_measures)
        if args.evaluate_only and not args.development_measures: development=[]
    if args.evaluate_only and not evaluation and not args.evaluation_measures:
        parser.error('--evaluate-only requires --evaluation-input or --evaluation-measures')
    root_log = output/'benchmark.log'
    log_event(root_log, f'Benchmark started; {len(plan["variants"])} strategies')
    api.write_json(output/'benchmark_status.json', dict(status='running', strategies=list(plan['variants']),
                                                        started_at=now()))
    failed=False
    for name,config in plan['variants'].items():
        folder=output/name;folder.mkdir(exist_ok=True)
        started=time.monotonic()
        try:
            write_status(folder/'status.json', status='running', phase='starting', strategy=name)
            log_event(root_log, f'{name}: started')
            if not args.evaluate_only and not (folder/'trained.json').exists():
                write_status(folder/'status.json', phase='training')
                api.write_json(folder/'config.json',config)
                command=[Path(api.__file__),'train','--records',*plan['records'],
                         '--output',folder/'runs','--config',folder/'config.json','--device',args.device]
                if (folder/'runs/last.pt').exists(): command.append('--resume')
                elif (folder/'runs').exists():
                    # Preserve evidence from an interrupted run before its first checkpoint.
                    (folder/'runs').rename(folder/f'unfinished_runs_{time.time_ns()}')
                run_child(command,folder/'training.log',folder/'status.json','training')
                api.write_json(folder/'trained.json',dict(checkpoint_sha256=api.sha256(folder/'runs/best.pt')))
                write_status(folder/'status.json', phase='training_complete')
            if not (folder/'trained.json').exists(): raise ValueError('Strategy has not completed training')
            if read_json(folder/'trained.json')['checkpoint_sha256']!=api.sha256(folder/'runs/best.pt'):
                raise ValueError('Trained checkpoint changed since benchmark completion')
            write_status(folder/'status.json', phase='development_evaluation')
            for record in development:
                log_event(root_log, f'{name}: evaluating development measurement {record.name}')
                final_report(record,folder,'development',args.device,args.force_evaluation)
                if args.epoch_metrics:
                    epoch_evaluation(record,folder,'development',args.device)
            write_status(folder/'status.json', phase='external_evaluation' if evaluation else 'complete')
            for record in evaluation:
                log_event(root_log, f'{name}: evaluating unseen measurement {record.name}')
                final_report(record,folder,'unseen',args.device,args.force_evaluation)
                if args.epoch_metrics:
                    epoch_evaluation(record,folder,'unseen',args.device)
            write_status(folder/'status.json', status='complete', phase='complete',
                         elapsed_seconds=time.monotonic()-started, external_evaluation=bool(evaluation))
            log_event(root_log, f'{name}: complete in {time.monotonic()-started:.1f}s')
        except Exception as exc:
            failed=True;print(f'{name}: {exc}',flush=True)
            write_status(folder/'status.json', status='failed', phase='failed', error=str(exc))
            log_event(root_log, f'{name}: FAILED: {exc}')
        finally:
            comparison(output,plan)
    write_status(output/'benchmark_status.json', status='failed' if failed else 'complete',
                 phase='complete', failed=failed, finished_at=now())
    log_event(root_log, 'Benchmark finished: ' + ('failures occurred' if failed else 'all strategies complete'))
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
