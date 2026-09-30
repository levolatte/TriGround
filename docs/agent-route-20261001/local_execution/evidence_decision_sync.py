"""Run inside the authenticated local cloud session; no credentials stored here."""
import json
from pathlib import Path
import shlex
import tarfile
import threading

TASK_TOOL_LOCK = threading.Lock()
TASK_ACTIVE_GPU_PHASE = None

TASK_ROOT = '/root/autodl-tmp/rematch_20260922'
TASK_OLD = TASK_ROOT + '/results/visual_agent/capability_rebuild_20260929'
TASK_OUT = TASK_ROOT + '/results/visual_agent/evidence_decision_20260930'
TASK_CODE = TASK_OUT + '/code'
TASK_PY = TASK_ROOT + '/.venvs/aux_selection/bin/python'


def remote_task(command, timeout=180):
    _, stdout, stderr = cloud.exec_command(command, timeout=timeout)
    output = stdout.read().decode()
    error = stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f'Remote command exited {status}: {error or output}')
    return output


def sync_task_code():
    package = Path('F:/AIC/code/experiments/evidence_decision')
    archive = Path('F:/AIC/results/visual_agent/evidence_decision_20260930/code.tar')
    files = [p for p in package.rglob('*') if p.is_file() and p.suffix in {'.py', '.md'}]
    with tarfile.open(archive, 'w') as handle:
        for path in files:
            handle.add(path, arcname='experiments/evidence_decision/' + path.relative_to(package).as_posix())
    remote_task('mkdir -p ' + shlex.quote(TASK_CODE + '/experiments'))
    sftp = cloud.open_sftp()
    sftp.put(str(archive), TASK_OUT + '/code.tar')
    sftp.close()
    command = ('cp -a ' + shlex.quote(TASK_OLD + '/training_code/tools') + ' ' + shlex.quote(TASK_CODE) +
               ' && cp -a ' + shlex.quote(TASK_OLD + '/training_code/src') + ' ' + shlex.quote(TASK_CODE) +
               ' && tar -xf ' + shlex.quote(TASK_OUT + '/code.tar') + ' -C ' + shlex.quote(TASK_CODE))
    remote_task(command)
    return {'remote_code': TASK_CODE, 'files': len(files)}


def run_task_module(module, *args, timeout=180):
    module = module if module.startswith('experiments.') else 'experiments.evidence_decision.' + module
    command = ('cd ' + shlex.quote(TASK_CODE) + ' && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ' +
               shlex.join([TASK_PY, '-m', module, *map(str, args)]))
    return remote_task(command, timeout)


def sync_task_data():
    data = Path('F:/AIC/results/visual_agent/evidence_decision_20260930/data')
    archive = data.parent / 'data.tar'
    files = [path for path in data.rglob('*') if path.is_file()]
    with tarfile.open(archive, 'w') as handle:
        for path in files:
            handle.add(path, arcname='data/' + path.relative_to(data).as_posix())
    sftp = cloud.open_sftp()
    sftp.put(str(archive), TASK_OUT + '/data.tar')
    sftp.close()
    remote_task('tar -xf ' + shlex.quote(TASK_OUT + '/data.tar') + ' -C ' + shlex.quote(TASK_OUT))
    return {'files': len(files), 'bytes': archive.stat().st_size}


def fetch_visible_state(remote_dir, local_dir, step_index=None):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    arguments = ['--episode-dir', remote_dir]
    if step_index is not None:
        arguments += ['--step-index', str(step_index)]
    details = json.loads(run_task_module('prepare_teacher_views', *arguments))
    remote_visible = details['directory']
    visible_dir = local_dir / 'visible' / f"step_{details['step']:04d}"
    visible_dir.mkdir(parents=True, exist_ok=True)
    sftp = cloud.open_sftp()
    messages = json.loads(sftp.open(remote_visible + '/current_messages.json').read().decode())
    cache_dir = Path('F:/AIC/results/visual_agent/evidence_decision_20260930/teachers/image_cache')
    if threading.current_thread().name != 'MainThread':
        cache_dir = cache_dir / threading.current_thread().name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index = cache_dir / 'index.json'
    cache = json.loads(cache_index.read_text(encoding='utf-8')) if cache_index.exists() else {}
    for message in messages:
        for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
            if part.get('type') != 'image':
                continue
            cache_key = json.dumps([part['remote_image'], part['max_pixels'], part['prepared_size']])
            if cache_key not in cache:
                destination = cache_dir / f'image_{len(cache):06d}.png'
                sftp.get(part['image'], str(destination))
                cache[cache_key] = str(destination.resolve())
                cache_index.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
            destination = Path(cache[cache_key])
            part['image'] = str(destination.resolve())
    encoded = json.dumps(messages, ensure_ascii=False, indent=2)
    (local_dir / 'current_messages.json').write_text(encoded, encoding='utf-8')
    (visible_dir / 'current_messages.json').write_text(encoded, encoding='utf-8')
    for name in ['current_tools.json', 'image_preparation.json']:
        sftp.get(remote_visible + '/' + name, str(local_dir / name))
        (visible_dir / name).write_bytes((local_dir / name).read_bytes())
    sftp.get(remote_dir + '/episode.json', str(local_dir / 'episode.json'))
    sftp.close()
    return {'id': local_dir.name, 'directory': str(local_dir), 'visible_state': str(visible_dir),
            'images': details['images'], 'step': details['step']}


def fetch_teacher_views(sample_id):
    return fetch_visible_state(TASK_OUT + '/teachers/fresh/' + sample_id,
                               Path('F:/AIC/results/visual_agent/evidence_decision_20260930/teachers/fresh') / sample_id)


def fetch_episode(remote_episode_dir, local_episode_dir):
    local_episode_dir = Path(local_episode_dir)
    sftp = cloud.open_sftp()
    try:
        sftp.stat(remote_episode_dir + '/episode.json')
    except FileNotFoundError:
        sftp.close()
        return False
    local_episode_dir.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_episode_dir + '/episode.json', str(local_episode_dir / 'episode.json'))
    sftp.close()
    return True


def init_fresh_teacher(sample_id, prefix=None, prefix_events=None):
    arguments = ['init', '--manifest', TASK_OLD + '/data/train_manifest.jsonl',
                 '--candidate-cache', TASK_OLD + '/data/train_candidates.jsonl',
                 '--sample-id', sample_id, '--output-dir', TASK_OUT + '/teachers/fresh/' + sample_id,
                 '--dino-model', TASK_ROOT + '/models/grounding-dino-tiny',
                 '--sam-model', TASK_ROOT + '/models/sam2.1-hiera-tiny']
    if prefix:
        arguments += ['--resume-prefix', prefix, '--migrate-legacy-prefix']
        if prefix_events is not None:
            arguments += ['--prefix-events', str(prefix_events)]
    run_task_module('object_teacher', *arguments)
    return fetch_teacher_views(sample_id)


def step_fresh_teacher(sample_id, action_path):
    action = json.loads(Path(action_path).read_text(encoding='utf-8-sig'))
    result = run_collection_module('object_teacher', 'step', '--episode',
                            TASK_OUT + '/teachers/fresh/' + sample_id + '/episode.json',
                            '--note', action['note'], '--name', action['name'],
                            '--arguments', json.dumps(action['arguments'], ensure_ascii=False))
    return {'step': json.loads(result), 'views': fetch_teacher_views(sample_id)}


def start_cpu_preflight(training_name='train_full.jsonl', output_name='cpu_preflight_full_8192'):
    output = TASK_OUT + '/' + output_name
    command = [TASK_PY, '-m', 'experiments.evidence_decision.train',
               '--train', TASK_OUT + '/data/' + training_name,
               '--model', '/root/rematch_models/Qwen3-VL-8B-Instruct',
               '--init-adapter', TASK_ROOT + '/results/next_stage_20260925/c_phase2/checkpoint-500',
               '--output-dir', output, '--max-length', '8192', '--preflight-only']
    code = ('import os,subprocess,json; '
            'log=open(' + repr(TASK_OUT + '/cpu_preflight_8192.log') + ',"w"); '
            'p=subprocess.Popen(' + repr(command) + ',cwd=' + repr(TASK_CODE) + ',stdout=log,stderr=subprocess.STDOUT,'
            'env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"},start_new_session=True); '
            'print(json.dumps({"pid":p.pid,"log":log.name}))')
    return json.loads(remote_task(shlex.join([TASK_PY, '-c', code])))


def start_gpu_phase(phase, context_tokens=8192):
    global TASK_ACTIVE_GPU_PHASE
    command = [TASK_PY, '-u', '-m', 'experiments.evidence_decision.run_gpu_phase',
               '--phase', phase, '--context-tokens', str(context_tokens), '--execute']
    log_path = TASK_OUT + '/' + phase + '_launch.log'
    code = ('import os,subprocess,json; log=open(' + repr(log_path) + ',"x"); '
            'p=subprocess.Popen(' + repr(command) + ',cwd=' + repr(TASK_CODE) + ',stdout=log,stderr=subprocess.STDOUT,'
            'env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"},start_new_session=True); '
            'print(json.dumps({"pid":p.pid,"log":log.name}))')
    TASK_TOOL_LOCK.acquire()
    try:
        launched = json.loads(remote_task(shlex.join([TASK_PY, '-c', code])))
    except Exception:
        TASK_TOOL_LOCK.release()
        raise
    TASK_ACTIVE_GPU_PHASE = (phase, context_tokens)
    return launched


def read_gpu_phase():
    global TASK_ACTIVE_GPU_PHASE
    phase, context_tokens = TASK_ACTIVE_GPU_PHASE
    path = TASK_OUT + '/stages/' + phase + '_' + str(context_tokens) + '/state.json'
    result = json.loads(remote_task('cat ' + shlex.quote(path)))
    if result['status'] != 'running':
        TASK_ACTIVE_GPU_PHASE = None
        TASK_TOOL_LOCK.release()
    return result


def start_gpu_segment(phase, segment, context_tokens=8192, run_seconds=500):
    """Resume the same evaluation between finite, separately charged stages."""
    global TASK_ACTIVE_GPU_PHASE
    plan = json.loads(run_task_module('run_gpu_phase', '--phase', phase,
                                     '--context-tokens', context_tokens))
    command = plan['command']
    label = phase + '_seg' + str(segment).zfill(2)
    command[command.index('--name') + 1] = label + '_' + str(context_tokens)
    command[command.index('--seconds') + 1] = str(run_seconds + 100)
    command[command.index('--max-run-seconds') + 1] = str(run_seconds)
    if segment > 1:
        command.append('--resume')
    log_path = TASK_OUT + '/' + label + '_launch.log'
    launch = ('import os,subprocess,json; log=open(' + repr(log_path) + ',"x"); '
              'p=subprocess.Popen(' + repr(command) + ',cwd=' + repr(TASK_CODE) +
              ',stdout=log,stderr=subprocess.STDOUT,'
              'env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"},start_new_session=True); '
              'print(json.dumps({"pid":p.pid,"log":log.name}))')
    TASK_TOOL_LOCK.acquire()
    try:
        launched = json.loads(remote_task(shlex.join([TASK_PY, '-c', launch])))
    except Exception:
        TASK_TOOL_LOCK.release()
        raise
    TASK_ACTIVE_GPU_PHASE = (label, context_tokens)
    return launched


def wait_gpu_phase_completion(pid, phase_key, timeout_seconds):
    """Let GNU tail watch one process; the agent need not keep checking it."""
    remote_task(shlex.join(['timeout', str(timeout_seconds), 'tail',
                           '--pid=' + str(pid), '-f', '/dev/null']),
                timeout=timeout_seconds + 30)
    if TASK_ACTIVE_GPU_PHASE != phase_key:
        raise RuntimeError('The watched GPU phase changed before process completion')
    return read_gpu_phase()


def run_collection_module(module, *args):
    model_backed = (module.endswith('object_teacher') and args[0] == 'step' and
                    args[args.index('--name') + 1] in {'search', 'depth'})
    if model_backed:
        # Teacher reasoning can overlap; actual model-backed observations and
        # budget ledger updates remain serial on the dedicated host.
        with TASK_TOOL_LOCK:
            return _run_collection_module(module, *args)
    return _run_collection_module(module, *args)


def _run_collection_module(module, *args):
    arguments = list(args)
    if module.endswith('object_teacher') and arguments[0] == 'init':
        arguments += ['--dino-model', TASK_ROOT + '/models/grounding-dino-tiny',
                      '--sam-model', TASK_ROOT + '/models/sam2.1-hiera-tiny']
    if (module.endswith('object_teacher') and arguments[0] == 'step' and
            arguments[arguments.index('--name') + 1] in {'search', 'depth'}):
        # With the dedicated GPU restored, model-backed observations share
        # the same original 20-hour ledger as training and evaluation.
        episode = arguments[arguments.index('--episode') + 1]
        info_code = ('import json,pathlib; b=pathlib.Path(' + repr(TASK_OUT) + '); '
                     's=json.loads(pathlib.Path(' + repr(episode) + ').read_text()); '
                     "tag='teacher_tool_'+s['sample_id']+'_'+str(len(s['steps'])); "
                     "logs=[json.loads(l) for l in (b/'ledger.jsonl').read_text().splitlines()] if (b/'ledger.jsonl').exists() else []; "
                     "used=sum(r['elapsed_seconds'] for r in logs if r['task'].startswith('teacher_tool_')); "
                     "print(json.dumps({'name':tag+'_'+str(len(list((b/'stages').glob(tag+'_*')))), 'tool_seconds':used}))")
        info = json.loads(remote_task(shlex.join([TASK_PY, '-c', info_code])))
        seconds = min(280, 2340 - info['tool_seconds'])
        if seconds <= 0:
            raise RuntimeError('Actual teacher tool preparation reached its 0.65-hour allocation')
        actual_module = module if module.startswith('experiments.') else 'experiments.evidence_decision.' + module
        wrapped = [TASK_PY, '-m', 'experiments.evidence_decision.run_capability_stage',
                   '--root', TASK_OUT, '--old-budget-root', TASK_ROOT + '/results/visual_agent/implementation_20260928',
                   '--prior-ledger', TASK_OLD + '/ledger.jsonl', '--name', info['name'],
                   '--seconds', str(seconds), '--cwd', TASK_CODE, '--',
                   TASK_PY, '-m', actual_module, *map(str, arguments)]
        run_state = json.loads(remote_task('cd ' + shlex.quote(TASK_CODE) + ' && ' + shlex.join(wrapped), timeout=300))
        log = remote_task('cat ' + shlex.quote(TASK_OUT + '/stages/' + info['name'] + '/output.log'))
        result = log.strip().splitlines()[-1]
        json.loads(result)
        print(json.dumps({'gpu_tool_stage': info['name'], 'seconds': run_state['elapsed_seconds']}, ensure_ascii=False), flush=True)
    else:
        result = run_task_module(module, *arguments, timeout=300)
    if module.endswith('object_teacher'):
        print(json.dumps({'remote_action': arguments[0],
                          'sample': arguments[arguments.index('--sample-id') + 1]
                          if '--sample-id' in arguments else arguments[arguments.index('--episode') + 1].split('/')[-2],
                          'status': json.loads(result).get('status', 'initialized')}, ensure_ascii=False), flush=True)
    return result


def collection_jobs(priority_count=128, include_recovery=False):
    data_dir = Path('F:/AIC/results/visual_agent/evidence_decision_20260930/data')
    jobs = [json.loads(line) for line in (data_dir / 'blind_teacher_priority_jobs.jsonl').read_text(encoding='utf-8').splitlines()][:priority_count]
    if include_recovery:
        jobs += [json.loads(line) for line in (data_dir / 'recovery_jobs55.jsonl').read_text(encoding='utf-8').splitlines()]
    return normalise_collection_jobs(jobs)


def read_collection_jobs(filename):
    path = Path('F:/AIC/results/visual_agent/evidence_decision_20260930/data') / filename
    return normalise_collection_jobs([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()])


def normalise_collection_jobs(jobs):
    for job in jobs:
        job['manifest'] = job['manifest_path']
        job['candidate'] = job['candidate_cache_path']
        job['bucket'] = job.get('selection_bucket', 'recovery')
        job.setdefault('episode_dir', TASK_OUT + '/teachers/blind/recovery/' + job['sample_id'])
    return jobs


def diagnose_teacher_process(remote_episode, action_path, label):
    action = json.loads(Path(action_path).read_text(encoding='utf-8'))
    command = [TASK_PY, '-u', '-m', 'experiments.evidence_decision.object_teacher',
               'step', '--episode', remote_episode + '/episode.json', '--note', action['note'],
               '--name', action['name'], '--arguments', json.dumps(action['arguments'], ensure_ascii=False)]
    directory = TASK_OUT + '/teachers/diagnostics'
    code = '''import json,pathlib,subprocess,time,os
directory=pathlib.Path(DIRECTORY)
directory.mkdir(parents=True,exist_ok=True)
def memory():
    root=pathlib.Path('/sys/fs/cgroup')
    return {p.name:p.read_text() for p in root.glob('memory.*') if p.name in ('memory.max','memory.current','memory.events')}
before=memory()
started=time.monotonic()
timed_out=False
with (directory / (LABEL+'.log')).open('w') as log:
    try:
        process=subprocess.run(COMMAND,cwd=WORKDIR,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},timeout=160)
        returncode=process.returncode
    except subprocess.TimeoutExpired:
        returncode=None
        timed_out=True
result={'returncode':returncode,'timed_out':timed_out,'seconds':time.monotonic()-started,'before':before,'after':memory(),
        'OMP_NUM_THREADS':4,'MKL_NUM_THREADS':4}
(directory / (LABEL+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''.replace('DIRECTORY', repr(directory)).replace('LABEL', repr(label)).replace('COMMAND', repr(command)).replace('WORKDIR', repr(TASK_CODE))
    return remote_task(shlex.join([TASK_PY, '-c', code]), timeout=300)
