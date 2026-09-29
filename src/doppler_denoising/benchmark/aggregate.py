"""Aggregate a completed benchmark across measurements and data groups."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


GROUPS=("development","unseen")
METRICS=(
    ("background.NRR","Background NRR","higher is better"),
    ("background.background_std_denoised","Denoised background temporal SD","lower is better"),
    ("regions.retinal_artery.temporal_correlation","Arterial temporal correlation","higher is better"),
    ("regions.retinal_artery.amplitude_ratio","Arterial cardiac amplitude ratio","closer to 1 is better"),
    ("regions.retinal_artery.residual_pulsatility_ratio","Arterial residual pulsatility ratio","lower is better"),
    ("regions.retinal_vein.temporal_correlation","Venous temporal correlation","higher is better"),
    ("regions.retinal_vein.amplitude_ratio","Venous cardiac amplitude ratio","closer to 1 is better"),
    ("regions.retinal_vein.residual_pulsatility_ratio","Venous residual pulsatility ratio","lower is better"),
    ("regions.choroidal.temporal_correlation","Choroidal temporal correlation","higher is better"),
    ("regions.choroidal.amplitude_ratio","Choroidal cardiac amplitude ratio","closer to 1 is better"),
    ("regions.choroidal.residual_pulsatility_ratio","Choroidal residual pulsatility ratio","lower is better"),
)
EPOCH_METRICS=(
    ("NRR","Background NRR","higher is better"),
    ("retinal_artery.temporal_correlation","Arterial temporal correlation","higher is better"),
    ("retinal_artery.residual_pulsatility_ratio","Arterial residual pulsatility ratio","lower is better"),
    ("retinal_vein.temporal_correlation","Venous temporal correlation","higher is better"),
    ("retinal_vein.residual_pulsatility_ratio","Venous residual pulsatility ratio","lower is better"),
    ("choroidal.temporal_correlation","Choroidal temporal correlation","higher is better"),
    ("choroidal.residual_pulsatility_ratio","Choroidal residual pulsatility ratio","lower is better"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(value, key):
    for part in key.split("."):
        if not isinstance(value,dict) or part not in value: return None
        value=value[part]
    return float(value) if isinstance(value,(int,float)) and np.isfinite(value) else None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def strategies(root):
    plan=root/"plan.json"
    if plan.exists(): return list(read_json(plan).get("variants",{}))
    return sorted(folder.name for folder in root.iterdir()
                  if folder.is_dir() and (folder/"reports").is_dir())


def collect_final(root, names):
    rows=[];links=[]
    for strategy in names:
        for group in GROUPS:
            reports=root/strategy/"reports"/group
            if not reports.is_dir(): continue
            for folder in sorted(p for p in reports.iterdir() if p.is_dir()):
                path=folder/"metrics.json"
                if not path.exists(): continue
                data=read_json(path)
                row=dict(strategy=strategy,group=group,measure=data.get("record",folder.name))
                for key,_,_ in METRICS: row[key]=nested(data,key)
                # Useful audit columns retained even when they are not plotted.
                for region in ("retinal_artery","retinal_vein","choroidal"):
                    for field in ("mean_change_percent","lag_ms"):
                        key=f"regions.{region}.{field}";row[key]=nested(data,key)
                rows.append(row)
                links.append(dict(strategy=strategy,group=group,measure=row["measure"],path=folder/"report.html"))
    return rows,links


def collect_epochs(root, names):
    rows=[]
    for strategy in names:
        for group in GROUPS:
            reports=root/strategy/"reports"/group
            if not reports.is_dir(): continue
            for folder in sorted(p for p in reports.iterdir() if p.is_dir()):
                path=folder/"epoch_metrics.json"
                if not path.exists(): continue
                for saved in read_json(path).get("rows",[]):
                    row=dict(strategy=strategy,group=group,measure=folder.name,
                             epoch=int(saved["epoch"]))
                    for key,_,_ in EPOCH_METRICS: row[key]=nested(saved.get("diagnostics",{}),key)
                    rows.append(row)
    return rows


def collect_losses(root, names):
    rows=[]
    for strategy in names:
        path=root/strategy/"runs"/"metrics.csv"
        if not path.exists(): continue
        with path.open(newline="",encoding="utf-8") as stream:
            for source in csv.DictReader(stream):
                row=dict(strategy=strategy,epoch=int(source["epoch"]))
                for key in ("train.total","valid.total"):
                    try: row[key]=float(source[key])
                    except (KeyError,TypeError,ValueError): row[key]=None
                rows.append(row)
    return rows


def tukey(values):
    """Return lower/upper Tukey fences; four points are required."""
    if len(values)<4: return -np.inf,np.inf
    q1,q3=np.percentile(values,[25,75]);width=q3-q1
    return q1-1.5*width,q3+1.5*width


def summarize_final(rows):
    summary=[];outliers=[]
    for key,label,direction in METRICS:
        for group in GROUPS:
            for strategy in sorted({r["strategy"] for r in rows}):
                selected=[r for r in rows if r["strategy"]==strategy and r["group"]==group
                          and r.get(key) is not None]
                values=np.asarray([r[key] for r in selected],float)
                if not len(values): continue
                low,high=tukey(values)
                flagged=[r for r in selected if r[key]<low or r[key]>high]
                summary.append(dict(metric=key,label=label,direction=direction,group=group,
                    strategy=strategy,n=len(values),mean=float(values.mean()),
                    std=float(values.std(ddof=1)) if len(values)>1 else 0.0,
                    median=float(np.median(values)),q1=float(np.percentile(values,25)),
                    q3=float(np.percentile(values,75)),minimum=float(values.min()),
                    maximum=float(values.max()),outlier_count=len(flagged)))
                outliers.extend(dict(metric=key,label=label,group=group,strategy=strategy,
                                     measure=r["measure"],value=r[key],lower_fence=low,upper_fence=high)
                                for r in flagged)
    return summary,outliers


def summarize_epochs(rows):
    result=[]
    for key,label,direction in EPOCH_METRICS:
        groups={(r["strategy"],r["group"],r["epoch"]) for r in rows if r.get(key) is not None}
        for strategy,group,epoch in sorted(groups):
            values=np.asarray([r[key] for r in rows if r["strategy"]==strategy
                               and r["group"]==group and r["epoch"]==epoch
                               and r.get(key) is not None],float)
            result.append(dict(metric=key,label=label,direction=direction,strategy=strategy,
                               group=group,epoch=epoch,n=len(values),mean=float(values.mean()),
                               std=float(values.std(ddof=1)) if len(values)>1 else 0.0))
    return result


def save_figure(figure,path):
    FigureCanvasAgg(figure).print_figure(path,dpi=145,format="png")


def short_labels(names):
    return {name:f"V{index+1}" for index,name in enumerate(names)}


def boxplots(rows,outliers,names,assets):
    labels=short_labels(names);files=[]
    colors=("#2671b8","#e47b25")
    for key,title,direction in METRICS:
        figure=Figure(figsize=(15,6),layout="constrained")
        axes=figure.subplots(1,2,sharey=True)
        for axis,group,color in zip(axes,GROUPS,colors):
            series=[];positions=[]
            for position,strategy in enumerate(names,1):
                values=[r[key] for r in rows if r["group"]==group and r["strategy"]==strategy
                        and r.get(key) is not None]
                if values: series.append(values);positions.append(position)
            if series:
                box=axis.boxplot(series,positions=positions,widths=.56,patch_artist=True,
                                 showfliers=False,medianprops={"color":"#111","linewidth":1.7})
                for patch in box["boxes"]: patch.set_facecolor(color);patch.set_alpha(.35)
            for position,strategy in enumerate(names,1):
                selected=[r for r in rows if r["group"]==group and r["strategy"]==strategy
                          and r.get(key) is not None]
                for index,row in enumerate(selected):
                    jitter=((index*37)%17-8)/75
                    flagged=any(o["metric"]==key and o["group"]==group and
                                o["strategy"]==strategy and o["measure"]==row["measure"]
                                for o in outliers)
                    axis.scatter(position+jitter,row[key],s=24,color="#c62828" if flagged else color,
                                 alpha=.9 if flagged else .55,zorder=3)
                    if flagged:
                        axis.annotate(row["measure"],(position+jitter,row[key]),xytext=(3,3),
                                      textcoords="offset points",fontsize=6,color="#a21818",rotation=25)
            axis.set(title=f"{group.title()} (n = measurements)",xticks=range(1,len(names)+1),
                     xticklabels=[labels[n] for n in names])
            axis.grid(axis="y",alpha=.2)
        figure.suptitle(f"{title} — {direction}",fontsize=14)
        filename=f"box_{key.replace('.','_')}.png";save_figure(figure,assets/filename);files.append((title,filename))
    return files


def epoch_plots(summary,names,assets):
    labels=short_labels(names);files=[]
    colors=[f"C{i%10}" for i in range(len(names))]
    for key,title,direction in EPOCH_METRICS:
        figure=Figure(figsize=(15,6),layout="constrained");axes=figure.subplots(1,2,sharey=True)
        for axis,group in zip(axes,GROUPS):
            for strategy,color in zip(names,colors):
                selected=sorted((r for r in summary if r["metric"]==key and
                                 r["strategy"]==strategy and r["group"]==group),key=lambda r:r["epoch"])
                if not selected: continue
                x=np.asarray([r["epoch"] for r in selected]);mean=np.asarray([r["mean"] for r in selected])
                std=np.asarray([r["std"] for r in selected])
                axis.plot(x,mean,".-",color=color,label=labels[strategy],linewidth=1.5)
                axis.fill_between(x,mean-std,mean+std,color=color,alpha=.12)
            axis.set(title=group.title(),xlabel="Checkpoint epoch");axis.grid(alpha=.2)
        axes[0].set_ylabel(title)
        figure.suptitle(f"{title}: mean ± SD across measurements — {direction}",fontsize=14)
        handles,legend_labels=axes[1].get_legend_handles_labels()
        if handles: figure.legend(handles,legend_labels,loc="outside lower center",ncol=4)
        filename=f"epoch_{key.replace('.','_')}.png";save_figure(figure,assets/filename);files.append((title,filename))
    return files


def loss_plot(rows,names,assets):
    if not rows: return None
    labels=short_labels(names);figure=Figure(figsize=(15,6),layout="constrained");axes=figure.subplots(1,2)
    for axis,key,title in zip(axes,("train.total","valid.total"),("Training objective","Validation objective")):
        for index,strategy in enumerate(names):
            selected=sorted((r for r in rows if r["strategy"]==strategy and r.get(key) is not None),
                            key=lambda r:r["epoch"])
            if selected: axis.plot([r["epoch"] for r in selected],[r[key] for r in selected],
                                   ".-",label=labels[strategy],color=f"C{index%10}")
        axis.set(title=title,xlabel="Epoch",ylabel="Loss");axis.grid(alpha=.2)
    handles,legend_labels=axes[1].get_legend_handles_labels()
    if handles: figure.legend(handles,legend_labels,loc="outside lower center",ncol=4)
    filename="optimization_loss.png";save_figure(figure,assets/filename);return filename


def table(headers,rows):
    return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(h))}</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
        "<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in row)+"</tr>" for row in rows)+"</tbody></table>"


def generate(benchmark,output=None):
    root=Path(benchmark).resolve();destination=Path(output).resolve() if output else root/"aggregate"
    if not (root/"plan.json").exists(): raise FileNotFoundError(f"No benchmark plan: {root}")
    destination.mkdir(parents=True,exist_ok=True);assets=destination/"assets";assets.mkdir(exist_ok=True)
    names=strategies(root);final,links=collect_final(root,names);epochs=collect_epochs(root,names);losses=collect_losses(root,names)
    if not final: raise ValueError("No current metrics.json reports were found")
    summary,outliers=summarize_final(final);epoch_summary=summarize_epochs(epochs)
    final_fields=["strategy","group","measure"]+[m[0] for m in METRICS]
    final_fields += [f"regions.{r}.{f}" for r in ("retinal_artery","retinal_vein","choroidal")
                     for f in ("mean_change_percent","lag_ms")]
    write_csv(destination/"measure_metrics.csv",final,final_fields)
    write_csv(destination/"summary_metrics.csv",summary,["metric","label","direction","group","strategy","n","mean","std","median","q1","q3","minimum","maximum","outlier_count"])
    write_csv(destination/"outliers.csv",outliers,["metric","label","group","strategy","measure","value","lower_fence","upper_fence"])
    write_csv(destination/"epoch_summary.csv",epoch_summary,["metric","label","direction","strategy","group","epoch","n","mean","std"])
    write_csv(destination/"training_curves.csv",losses,["strategy","epoch","train.total","valid.total"])
    boxes=boxplots(final,outliers,names,assets);curves=epoch_plots(epoch_summary,names,assets);loss=loss_plot(losses,names,assets)
    labels=short_labels(names)
    coverage=[]
    for name in names:
        coverage.append((labels[name],name,
                         len({r["measure"] for r in final if r["strategy"]==name and r["group"]=="development"}),
                         len({r["measure"] for r in final if r["strategy"]==name and r["group"]=="unseen"})))
    metric_rows=[]
    for _,label,direction in METRICS:
        metric_rows.append((label,direction))
    outlier_rows=[(o["label"],o["group"],labels[o["strategy"]],o["measure"],f'{o["value"]:.6g}') for o in outliers]
    link_rows=[]
    for item in links:
        href=Path(os.path.relpath(Path(item["path"]).resolve(),destination)).as_posix()
        link_rows.append((item["group"],labels[item["strategy"]],item["measure"],f'<a href="{html.escape(href)}">report</a>'))
    report=f"""<!doctype html><html><head><meta charset="utf-8"><title>Benchmark aggregate</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#17202a}}h1,h2{{color:#173b57}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{padding:7px;border-bottom:1px solid #d8dee4;text-align:left}}th{{background:#edf4f8;position:sticky;top:0}}img{{width:100%;border:1px solid #d8dee4;border-radius:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:18px}}.note{{background:#fff5d6;padding:12px;border-left:4px solid #d59b18}}code{{background:#eef2f4;padding:2px 4px}}details{{margin:8px 0}} </style></head><body>
<h1>Noise2Time benchmark aggregate</h1>
<p>Generated from <code>{html.escape(str(root))}</code>. Final-checkpoint box plots show every measurement. Red labeled points are Tukey outliers (outside 1.5 × IQR); groups with fewer than four measurements are not assigned formal outliers.</p>
<div class="note"><b>Interpretation:</b> development and unseen measurements are different cohorts, so differences are distribution shifts, not paired changes. Shaded epoch bands are mean ± one SD across measurements; they do not measure uncertainty across training seeds. Validation losses from different split policies may use different data and are not directly comparable.</div>
<h2>Variants and coverage</h2>{table(("ID","Variant","Development n","Unseen n"),coverage)}
<h2>Metric direction</h2>{table(("Metric","Desired result"),metric_rows)}
<p>Downloads: <a href="measure_metrics.csv">per-measure metrics</a> · <a href="summary_metrics.csv">distribution summaries</a> · <a href="outliers.csv">outliers</a> · <a href="epoch_summary.csv">epoch summaries</a> · <a href="training_curves.csv">training curves</a>.</p>
<h2>Final-checkpoint distributions</h2><div class="grid">{"".join(f'<figure><img src="assets/{f}"><figcaption>{html.escape(t)}</figcaption></figure>' for t,f in boxes)}</div>
<h2>Progression across checkpoint epochs</h2><div class="grid">{"".join(f'<figure><img src="assets/{f}"><figcaption>{html.escape(t)}</figcaption></figure>' for t,f in curves)}</div>
<h2>Optimization curves</h2>{f'<img src="assets/{loss}">' if loss else '<p>No training metrics found.</p>'}
<h2>Identified outliers</h2>{table(("Metric","Group","Variant","Measurement","Value"),outlier_rows) if outlier_rows else '<p>No Tukey outliers identified.</p>'}
<h2>Individual reports</h2><table><thead><tr><th>Group</th><th>Variant</th><th>Measurement</th><th>Link</th></tr></thead><tbody>{''.join('<tr>'+''.join(f'<td>{v if i==3 else html.escape(str(v))}</td>' for i,v in enumerate(row))+'</tr>' for row in link_rows)}</tbody></table>
</body></html>"""
    (destination/"report.html").write_text(report,encoding="utf-8")
    return destination/"report.html"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",required=True,help="Completed benchmark directory containing plan.json")
    parser.add_argument("--output",help="Report directory (default: BENCHMARK/aggregate)")
    args=parser.parse_args(argv)
    report=generate(args.benchmark,args.output);print(f"Aggregate report: {report}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
