import csv
import json

from doppler_denoising.benchmark import aggregate


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def test_aggregate_report_summarizes_groups_epochs_and_outliers(tmp_path,monkeypatch):
    root=tmp_path/"benchmark"
    variants={"one":{},"two":{}}
    write_json(root/"plan.json",{"variants":variants})
    monkeypatch.setattr(aggregate,"METRICS",(("background.NRR","Background NRR","higher is better"),))
    monkeypatch.setattr(aggregate,"EPOCH_METRICS",(("NRR","Background NRR","higher is better"),))
    for strategy in variants:
        runs=root/strategy/"runs";runs.mkdir(parents=True)
        (runs/"metrics.csv").write_text("epoch,train.total,valid.total\n1,0.2,0.3\n2,0.1,0.25\n",encoding="utf-8")
        for group in aggregate.GROUPS:
            for index,value in enumerate((1.0,1.0,1.0,10.0) if strategy=="one" else (2.0,2.1,2.2,2.3)):
                folder=root/strategy/"reports"/group/f"measure_{index}"
                write_json(folder/"metrics.json",{"record":f"measure_{index}","background":{"NRR":value}})
                write_json(folder/"epoch_metrics.json",{"rows":[
                    {"epoch":1,"diagnostics":{"NRR":value/2}},
                    {"epoch":2,"diagnostics":{"NRR":value}},
                ]})
                (folder/"report.html").write_text("report",encoding="utf-8")
    report=aggregate.generate(root)
    assert report.exists()
    page=report.read_text(encoding="utf-8")
    assert "mean ± one SD" in page and "measure_3" in page
    with (root/"aggregate/outliers.csv").open(newline="",encoding="utf-8") as stream:
        outliers=list(csv.DictReader(stream))
    assert {(r["strategy"],r["measure"]) for r in outliers}=={("one","measure_3")}
    with (root/"aggregate/epoch_summary.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert {int(r["epoch"]) for r in rows}=={1,2}
    assert (root/"aggregate/assets/box_background_NRR.png").exists()
    assert (root/"aggregate/assets/epoch_NRR.png").exists()
    assert (root/"aggregate/assets/optimization_loss.png").exists()
