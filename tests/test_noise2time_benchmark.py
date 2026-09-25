"""Ablation semantics, temporal leakage boundaries and a tiny six-run benchmark."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import h5py

from test_noise2time_dataset import dataset, prepare_command, n2t
from doppler_denoising.benchmark import runner as benchmark


@pytest.mark.parametrize('exit_code', [0, 7])
def test_real_child_exit_code_and_progress(tmp_path, capsys, exit_code):
    log = tmp_path/'training.log'
    status = tmp_path/'status.json'
    command = ['-u', '-c',
               f"import sys; print('progress 1\\rprogress 2', flush=True); sys.exit({exit_code})"]
    if exit_code:
        with pytest.raises(RuntimeError, match='Command exited 7'):
            benchmark.run_child(command, log, status)
    else:
        benchmark.run_child(command, log, status)
    assert 'progress 1\nprogress 2\n' in log.read_text()
    assert 'progress 2' in capsys.readouterr().out
    assert json.loads(status.read_text())['last_output'] == 'progress 2'


def test_recovered_status_clears_old_error(tmp_path):
    status = tmp_path/'status.json'
    benchmark.write_status(status, status='failed', error='old failure')
    benchmark.write_status(status, status='running', phase='training')
    assert 'error' not in json.loads(status.read_text())


def test_comparison_folders_separate_measurements_and_groups(tmp_path):
    import csv
    from urllib.parse import unquote
    import re
    for group,record in [('development','one'),('unseen','one'),('unseen','two')]:
        for strategy in ['baseline','no_convlstm']:
            report=tmp_path/strategy/'reports'/group/record
            report.mkdir(parents=True)
            (report/'report.html').write_text('Individual report')
            n2t.write_json(report/'metrics.json',dict(background={'NRR':.8},regions={}))
    benchmark.comparison(tmp_path,{'variants':{'baseline':{},'no_convlstm':{}}})
    for group,record in [('development','one'),('unseen','one'),('unseen','two')]:
        folder=tmp_path/'comparison'/group/record
        page=(folder/'report.html').read_text(encoding='utf-8')
        for href in re.findall(r'href="([^"]+)"',page):
            assert (folder/unquote(href)).resolve().exists()
        with (folder/'metrics.csv').open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        assert {r['strategy'] for r in rows}=={'baseline','no_convlstm'}
        assert {r['record'] for r in rows}=={record}
        assert {r['group'] for r in rows}=={group}


def prepared(tmp_path):
    root,folder,raw=dataset(tmp_path)
    output=tmp_path/'prepared_output'
    assert n2t.main(prepare_command(root,output))==0
    return n2t.Record(output/'prepared/measure_1')


def test_history_only_cannot_see_target_and_brightness_toggle(tmp_path):
    record=prepared(tmp_path)
    cfg=n2t.Config(history=2,block_size=8,blocks=1,objective='l2')
    t=record.eligible(2)[20]
    cfg.input_mode='history_only'
    sequence,target,mask=n2t.replacement(record,t,cfg,np.random.default_rng(1))
    np.testing.assert_array_equal(sequence,record.frames[t-2:t])
    np.testing.assert_array_equal(target[0],record.frames[t])
    cfg.input_mode='patched'; record.brightness[t]*=2
    corrected,_,_=n2t.replacement(record,t,cfg,np.random.default_rng(1))
    cfg.brightness_correction=False
    uncorrected,_,_=n2t.replacement(record,t,cfg,np.random.default_rng(1))
    assert np.any(corrected[-1][mask[0]>0]!=uncorrected[-1][mask[0]>0])
    np.testing.assert_array_equal(corrected[:-1],uncorrected[:-1])


def test_temporal_split_separates_all_frame_support(tmp_path):
    record=prepared(tmp_path)
    cfg=n2t.Config(history=2,block_size=8,blocks=1,validation_samples=3,split_mode='temporal')
    training,validation=n2t.split_samples([record],cfg)
    support={}
    for stage,pool in [('train',training),('valid',validation)]:
        used=set();lo,hi=record.split_ranges[stage]
        for i,t in pool:
            used.update(range(t-cfg.history,t+1))
            view=record.stage_views[stage]
            donors=view.matching_donors(t,view.donor_bounds)
            used.update(endpoint for donor in donors for endpoint in (donor.lower,donor.upper))
            sequence,_,_=n2t.replacement(record,t,cfg,np.random.default_rng(1),stage)
            assert sequence.shape[0]==cfg.history+1
        support[stage]=used
    train_range=set(range(*record.split_ranges['train']))
    valid_range=set(range(*record.split_ranges['valid']))
    assert train_range.isdisjoint(valid_range)
    assert all(d.lower in train_range and d.upper in train_range
               for _,t in validation for d in record.stage_views['valid'].matching_donors(t,record.stage_views['valid'].donor_bounds))
    # Changing validation intensities cannot change training timing or brightness.
    train_phase=record.stage_views['train'].phase.copy()
    train_brightness=record.stage_views['train'].brightness.copy()
    record.frames=np.array(record.frames)
    record.frames[record.split_ranges['valid'][0]:]*=.9
    n2t.split_samples([record],cfg)
    np.testing.assert_array_equal(record.stage_views['train'].phase,train_phase)
    np.testing.assert_array_equal(record.stage_views['train'].brightness,train_brightness)


def test_variants_change_one_factor():
    variants=benchmark.variants(n2t.Config(),'last_video')
    assert len(variants)==6
    for name,config in variants.items():
        changed={key for key in config if config[key]!=variants['baseline'][key]}
        assert len(changed)==(0 if name=='baseline' else 2 if name=='video_validation' else 1)


def test_explicit_combination_schema_expands_all_categories():
    base=n2t.Config(objective="l2")
    specification={
        "schema":benchmark.COMBINATION_SCHEMA,
        "defaults":{
            "objective":"l2","split":{"strategy":"random"},
            "frame_pairing":"cycle_phase","patch":"vessel_patches",
            "brightness_correction":"on","model":"unet_convlstm",
        },
        "experiments":[
            {"name":"baseline"},
            {"name":"combined","objective":"l1_grad_hessian",
             "split":{"strategy":"external_video","validation_records":["held_out"]},
             "frame_pairing":"next","patch":"none",
             "brightness_correction":"none","model":"unet"},
        ],
    }
    expanded=benchmark.combination_variants(base,specification)
    assert expanded["baseline"]["patch_mode"]=="vessel_patches"
    combined=expanded["combined"]
    assert combined["objective"]=="l1_grad_hessian"
    assert combined["split_mode"]=="record" and combined["validation_records"]==["held_out"]
    assert combined["frame_pairing"]=="next" and combined["patch_mode"]=="none"
    assert combined["brightness_correction"] is False and combined["convlstm"] is False
    portable=dict(specification)
    portable["experiments"]=[{"name":"portable_external",
                               "split":{"strategy":"external_video"}}]
    resolved=benchmark.combination_variants(base,portable,"last_video")
    assert resolved["portable_external"]["validation_records"]==["last_video"]
    duplicate=dict(specification)
    duplicate["experiments"]=[{"name":"one"},{"name":"two"}]
    with pytest.raises(ValueError,match="identical"):
        benchmark.combination_variants(base,duplicate)


def test_combination_benchmark_dry_run_builds_vessel_plan(tmp_path):
    record=prepared(tmp_path)
    config=tmp_path/'config.json'
    n2t.write_json(config,dict(base_channels=8,history=2,block_size=8,blocks=1,
                              epochs=1,samples_per_epoch=2,batch_size=2,
                              validation_samples=2,objective='l2'))
    specification={
        "schema":benchmark.COMBINATION_SCHEMA,
        "defaults":{
            "objective":"l2","split":{"strategy":"random"},
            "frame_pairing":"cycle_phase","patch":"vessel_patches",
            "brightness_correction":"on","model":"unet_convlstm",
        },
        "experiments":[{"name":"baseline"}],
    }
    experiments=tmp_path/'combinations.json';n2t.write_json(experiments,specification)
    output=tmp_path/'combination_benchmark'
    assert benchmark.main(['--prepared',str(record.path),'--output',str(output),
                           '--config',str(config),'--experiments',str(experiments),
                           '--device','cpu','--dry-run'])==0
    plan=json.loads((output/'plan.json').read_text())
    assert plan['variants']['baseline']['patch_mode']=='vessel_patches'
    assert plan['experiment_spec']==specification


@pytest.mark.parametrize('correction',[False,True])
def test_no_patch_inputs_and_whole_donor_target(tmp_path,correction):
    record=prepared(tmp_path)
    cfg=n2t.Config(history=9,input_mode='no_patch',brightness_correction=correction,
                   block_size=1000,blocks=1000,objective='l2')
    t=record.eligible(cfg.history)[30]
    donors=record.matching_donors(t,exclude=(t-cfg.history,t+1))
    chosen=donors[int(np.random.default_rng(3).integers(len(donors)))]
    donor_frame=n2t.interpolate_donor(record.frames,chosen)
    donor_brightness=float(n2t.interpolate_donor(record.brightness,chosen))
    record.brightness[t]=donor_brightness*1.2
    sequence,target,mask=n2t.replacement(record,t,cfg,np.random.default_rng(3))
    np.testing.assert_array_equal(sequence,record.frames[t-9:t+1])
    np.testing.assert_allclose(target[0],np.clip(donor_frame*(1.2 if correction else 1),0,1),rtol=1e-6)
    np.testing.assert_array_equal(mask[0],record.roi)
    assert sequence.shape[0]==10
    record.donor_bounds=(t-cfg.history,t+1)
    with pytest.raises(ValueError,match='outside the input window'):
        n2t.replacement(record,t,cfg,np.random.default_rng(3))


def test_selected_experiments_and_single_record_training(tmp_path,monkeypatch):
    base=n2t.Config(objective='l2')
    chosen=benchmark.variants(base,'unused',['no_patch','l1','l2','l1_grad_hessian','l2_grad_hessian'])
    assert list(chosen)==['no_patch','l1','l2','l1_grad_hessian','l2_grad_hessian']
    assert chosen['no_patch']['input_mode']=='no_patch'
    for name in ('l1','l2','l1_grad_hessian','l2_grad_hessian'):
        assert chosen[name]['objective']==name
    for bad in ([],['missing'],['l1','l1']):
        with pytest.raises(ValueError): benchmark.variants(base,'unused',bad)
    record=prepared(tmp_path)
    config=tmp_path/'config.json';selection=tmp_path/'experiments.json'
    n2t.write_json(config,dict(base_channels=8,history=9,block_size=8,blocks=1,objective='l2_grad_hessian',
                              epochs=1,samples_per_epoch=2,batch_size=2,validation_samples=2))
    n2t.write_json(selection,{'experiments':['no_patch']})
    def child(command,log,*args): assert n2t.main(list(map(str,command[2:])))==0
    monkeypatch.setattr(benchmark,'run_child',child)
    output=tmp_path/'selected_benchmark'
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert benchmark.main(['--prepared',str(record.path),'--output',str(output),'--config',str(config),
                               '--experiments',str(selection),'--device','cpu'])==0
    finally: torch.set_num_threads(previous)
    plan=json.loads((output/'plan.json').read_text())
    assert list(plan['variants'])==['no_patch']
    assert (output/'no_patch/reports/development/measure_1/report.html').exists()
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert benchmark.main(['--output',str(output),'--resume','--epochs','2',
                               '--device','cpu','--no-epoch-metrics'])==0
    finally: torch.set_num_threads(previous)
    assert torch.load(output/'no_patch/runs/last.pt',weights_only=True)['epoch']==2
    resumed=json.loads((output/'plan.json').read_text())
    assert resumed['variants']['no_patch']['epochs']==2
    assert list((output/'no_patch/legacy_reports/development').iterdir())


def test_plan_records_are_relocated_by_metadata_hash(tmp_path):
    output=tmp_path/'benchmark'
    record=tmp_path/'prepared'/'measure_1';record.mkdir(parents=True)
    n2t.write_json(record/'metadata.json',{'identity':'unchanged'})
    plan=dict(records=[r'C:\old_machine\prepared\measure_1'],
              prepared_hashes={'measure_1':n2t.sha256(record/'metadata.json')})
    assert benchmark.relocate_plan_records(plan,output)
    assert Path(plan['records'][0])==record.resolve()


def test_record_relinks_a_moved_dataset_by_folder_name(tmp_path):
    record=prepared(tmp_path)
    metadata=json.loads((record.path/'metadata.json').read_text())
    metadata['dataset_measure']=r'C:\old_machine\dataset\measure_1'
    n2t.write_json(record.path/'metadata.json',metadata)
    moved=n2t.Record(record.path)
    assert moved.dataset_measure==(tmp_path/'dataset/measure_1').resolve()
    assert moved.training_vessel_mask().any()


def test_history_only_inference_excludes_current_frame(tmp_path):
    record=prepared(tmp_path)
    class LastInput(torch.nn.Module):
        def forward(self, sequence): return sequence[:,-1:]
    cfg=n2t.Config(history=2,input_mode='history_only')
    restored=tmp_path/'denoised.npy'
    n2t.export_denoised(LastInput(),record,cfg,torch.device('cpu'),npy_path=restored)
    np.testing.assert_array_equal(np.load(restored)[2:],record.frames[1:-1])


def test_next_frame_inference_is_aligned_to_predicted_frame(tmp_path):
    record=prepared(tmp_path)
    class LastInput(torch.nn.Module):
        def forward(self, sequence): return sequence[:,-1:]
    cfg=n2t.Config(history=2,patch_mode="none",frame_pairing="next")
    restored=tmp_path/'next.npy'
    n2t.export_denoised(LastInput(),record,cfg,torch.device('cpu'),npy_path=restored)
    result=np.load(restored)
    np.testing.assert_array_equal(result[:3],record.frames[:3])
    np.testing.assert_array_equal(result[3:],record.frames[2:-1])


def test_six_strategy_orchestration(tmp_path,monkeypatch):
    record=prepared(tmp_path)
    # Give the second recording its own source and mask paths.
    root2,folder2,raw=dataset(tmp_path/'second')
    parent=record.path.parent
    assert n2t.main(prepare_command(root2,tmp_path/'second_output'))==0
    import shutil
    shutil.copytree(tmp_path/'second_output/prepared/measure_1',parent/'measure_2')
    config=tmp_path/'config.json'
    n2t.write_json(config,dict(base_channels=8,history=2,block_size=8,blocks=1,objective='l2',
                              epochs=1,samples_per_epoch=2,batch_size=2,validation_samples=2))
    # Run training in-process so tiny CPU tests do not pay six Python startup costs.
    def child(command,log,*args):
        assert n2t.main(list(map(str,command[2:])))==0
    monkeypatch.setattr(benchmark,'run_child',child)
    output=tmp_path/'benchmark'
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert benchmark.main(['--prepared',str(parent),'--output',str(output),'--config',str(config),'--device','cpu'])==0
    finally:
        torch.set_num_threads(previous)
    plan=json.loads((output/'plan.json').read_text())
    for name in plan['variants']:
        assert (output/name/'trained.json').exists()
        assert (output/name/'reports/development/measure_1/report.html').exists()
        previews=list((output/name/'runs/previews').glob('**/*.avi'))
        assert len(previews)==1 and previews[0].stem=='measure_1'
    assert (output/'index.html').exists()
    assert 'background_std_denoised' in (output/'comparison.csv').read_text()
    assert 'amplitude_ratio' in (output/'final_comparison.csv').read_text()
    assert (output/'comparison/development/measure_1/metrics_overview.png').is_file()
    assert benchmark.main(['--output',str(output),'--report-only'])==0


def test_external_evaluation_without_retraining(tmp_path,monkeypatch):
    record=prepared(tmp_path)
    output=tmp_path/'benchmark';folder=output/'baseline';folder.mkdir(parents=True)
    config=n2t.Config(base_channels=8,history=2,block_size=8,blocks=1,objective='l2',
                      epochs=1,samples_per_epoch=2,batch_size=2,validation_samples=2)
    config_path=folder/'config.json';n2t.write_json(config_path,benchmark.asdict(config))
    plan=dict(records=[str(record.path)],variants={'baseline':benchmark.asdict(config)},
              reference=record.metadata,prepared_hashes={record.name:n2t.sha256(record.path/'metadata.json')})
    n2t.write_json(output/'plan.json',plan)
    def forbidden(*args,**kwargs): raise AssertionError('Evaluation must not launch training')
    monkeypatch.setattr(benchmark,'run_child',forbidden)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        assert n2t.main(['train','--records',str(record.path),'--output',str(folder/'runs'),
                         '--config',str(config_path),'--device','cpu','--no-epoch-previews','--no-epoch-metrics'])==0
        checkpoint_hash=n2t.sha256(folder/'runs/best.pt')
        n2t.write_json(folder/'trained.json',dict(checkpoint_sha256=checkpoint_hash))
        external,source,_=dataset(tmp_path/'external')
        with h5py.File(source/'measure_1_DV.h5','r+') as h5:
            h5['doppler_signal/M0_ff'][...]*=.8
        # An unselected development-source copy must not block external evaluation
        # or be prepared/scored accidentally.
        import shutil
        shutil.copytree(Path(record.metadata['dataset_measure']),external/'unselected')
        assert benchmark.main(['--output',str(output),'--evaluate-only','--evaluation-input',str(external),
                               '--evaluation-measures','measure_1','--device','cpu'])==0
        assert (folder/'reports/unseen/measure_1/report.html').exists()
        epoch_path=folder/'reports/unseen/measure_1/epoch_metrics.json'
        assert [row['epoch'] for row in json.loads(epoch_path.read_text())['rows']]==[1]
        assert (output/'comparison/unseen/measure_1/epoch_metrics.csv').exists()
        assert list((output/'comparison/unseen/measure_1/epochs').glob('*.png'))
        # Rerunning with matching weights, masks and inputs must reuse diagnostics.
        def no_inference(*args,**kwargs): raise AssertionError('Cached epochs must not be recomputed')
        with monkeypatch.context() as patch:
            patch.setattr(n2t,'export_denoised',no_inference)
            benchmark.epoch_evaluation(output/'evaluation_data/prepared/measure_1',folder,'unseen','cpu')
        assert (folder/'runs/checkpoints/epoch_001.pt').exists()
        assert not (output/'evaluation_data/prepared/unselected').exists()
        assert 'unseen / measure_1' in (output/'index.html').read_text(encoding='utf-8')
        assert not (folder/'reports/development/measure_1/report.html').exists()
        assert benchmark.main(['--output',str(output),'--evaluate-only',
                               '--evaluation-measures','measure_1','--device','cpu'])==0
        assert (folder/'reports/development/measure_1/report.html').exists()
        with pytest.raises(ValueError,match='Unknown development measurements'):
            benchmark.main(['--output',str(output),'--evaluate-only','--evaluation-measures','missing'])
        assert n2t.sha256(folder/'runs/best.pt')==checkpoint_hash
        with pytest.raises(ValueError,match='also occurs in development'):
            benchmark.main(['--output',str(output),'--evaluate-only','--evaluation-input',str(Path(record.metadata['dataset_measure']).parent)])
    finally:
        torch.set_num_threads(previous)
