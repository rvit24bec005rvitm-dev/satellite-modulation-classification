"""Offline recovery contracts using real generated capture/CNN/report artifacts."""
from copy import deepcopy
from contextlib import closing
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time
from uuid import uuid4

import numpy as np
import pytest
from starlette.datastructures import UploadFile

from backend.services import captures, classifier, managed_comparisons, operations, saved_explanations
from backend.services import workspace_backups as backups, workspace_store as store
from utils.signal_processing import generate_iq_signal


@pytest.fixture(scope='module')
def genuine_seed(tmp_path_factory):
    """Create once in a temporary directory; never open the user's .satmod."""
    source = tmp_path_factory.mktemp('synthetic-backup-seed').resolve()
    previous = os.environ.get('SATMOD_DATA_DIR')
    os.environ['SATMOD_DATA_DIR'] = str(source)
    store.close_connections()
    try:
        iq = (generate_iq_signal('BPSK', num_samples=512, rng_seed=84175, snr_db=18)*.006).astype('<f4')
        capture = captures.import_capture(UploadFile(io.BytesIO(iq.tobytes()), filename='synthetic-backup.bin'),
                                         captures.ImportOptions(format='raw', datatype='cf32_le', sampleRate=8000,
                                                                sessionId='synthetic-backup-session', source='Generated BPSK backup fixture, not satellite RF'))
        analysis = classifier.classify(iq[:128], 'Generated backup analysis', 8000,
                                       capture_metadata=capture, capture_position=captures.timing(capture, 0))
        comparison = managed_comparisons.create({'captureId': capture['id'], 'start': 0, 'length': 128},
                                                {'captureId': capture['id'], 'start': 128, 'length': 128})
        dataset = operations.register_dataset('Synthetic backup labels', [{'captureId': capture['id'], 'start': 0, 'label': 'BPSK', 'split': 'test'}],
                                              'Known generated BPSK fixture', 'Synthetic test use')
        job = operations.submit('batch', {'captureId': capture['id'], 'start': 0, 'windows': 2, 'stride': 128})
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            current = store.get('job', job['id'])
            if current['state'] in {'completed', 'failed', 'cancelled'}:
                break
            time.sleep(.01)
        assert current['state'] == 'completed', current
        ids = {'capture': capture['id'], 'analysis': analysis['id'], 'comparison': comparison['id'],
               'dataset': dataset['id'], 'job': job['id']}
    finally:
        runner = operations.RUNNERS.pop(str(source), None)
        if runner:
            runner[0].shutdown(wait=True)
        store.close_connections()
        if previous is None:
            os.environ.pop('SATMOD_DATA_DIR', None)
        else:
            os.environ['SATMOD_DATA_DIR'] = previous
    return source, ids


@pytest.fixture
def workspace(genuine_seed, tmp_path, monkeypatch):
    seed, ids = genuine_seed
    source = tmp_path/'source'
    shutil.copytree(seed, source)
    monkeypatch.setenv('SATMOD_DATA_DIR', str(source))
    store.close_connections()
    yield source, ids
    store.close_connections()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hashes(root):
    return {p.name: digest(p) for p in root.iterdir() if p.name == backups.DATABASE or backups.CANONICAL.fullmatch(p.name)}


def archive_of(workspace, tmp_path):
    source, _ = workspace
    path = tmp_path/'private-backup.tar'
    manifest = backups.backup_workspace(source, path, stopped=True)
    return path, manifest


def archive_parts(path):
    with tarfile.open(path, 'r:') as archive:
        return [(member.name, archive.extractfile(member).read()) for member in archive]


def write_parts(path, parts):
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in parts:
            member = tarfile.TarInfo(name); member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def replace_part(path, name, content):
    parts = archive_parts(path)
    manifest = json.loads(dict(parts)[backups.MANIFEST])
    for entry in manifest['files']:
        if entry['path'] == name:
            manifest['totalBytes'] += len(content)-entry['bytes']
            entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    write_parts(path, [(key, json.dumps(manifest).encode() if key == backups.MANIFEST else content if key == name else payload)
                       for key, payload in parts])


def assert_refused_unchanged(path, destination, pattern=None, **kwargs):
    before = digest(path)
    with pytest.raises(ValueError, match=pattern):
        backups.restore_workspace(path, destination, stopped=True, **kwargs)
    assert digest(path) == before
    assert not destination.exists()
    assert not list(destination.parent.glob('.satmod-restore-*'))


def test_genuine_whole_workspace_restores_counts_exact_capture_reports_and_hashes(workspace, tmp_path, monkeypatch):
    source, ids = workspace
    before = canonical_hashes(source)
    archive, manifest = archive_of(workspace, tmp_path)
    assert canonical_hashes(source) == before
    assert manifest['recordCounts'] == {kind: 1 for kind in backups.KINDS}
    assert {entry['path'] for entry in manifest['files']} == set(before)
    with tarfile.open(archive, 'r:') as container:
        assert not any(name.endswith(('-wal', '-shm')) for name in container.getnames())
    destination = tmp_path/'restored'
    restored = backups.restore_workspace(archive, destination, stopped=True)
    assert restored == manifest
    # SQLite is consolidated; all canonical sample and report bytes are exact.
    for name, expected in before.items():
        if name != backups.DATABASE:
            assert digest(destination/name) == expected
    for entry in manifest['files']:
        assert digest(destination/entry['path']) == entry['sha256']
    monkeypatch.setenv('SATMOD_DATA_DIR', str(destination))
    store.close_connections()
    assert {kind: store.listing(kind)['total'] for kind in backups.KINDS} == manifest['recordCounts']
    assert saved_explanations.load_analysis(ids['analysis']) == json.loads((source/(ids['analysis']+'.json')).read_text())
    assert managed_comparisons.load(ids['comparison']) == json.loads((source/(ids['comparison']+'.json')).read_text())
    actual, _ = captures.window(ids['capture'], 128, 128)
    expected = np.fromfile(source/(ids['capture']+'.iq'), dtype='<f4').reshape(-1, 2)[128:256]
    np.testing.assert_array_equal(actual, expected)
    if os.name != 'nt':
        assert stat.S_IMODE(archive.stat().st_mode) == 0o600
        assert stat.S_IMODE(destination.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in destination.iterdir())


def test_committed_wal_is_consolidated_without_modifying_source(workspace, tmp_path):
    source, _ = workspace
    added = store.put('analysis', {'source': 'Synthetic WAL-only legacy report'})
    (source/(added['id']+'.json')).write_text(json.dumps(added))
    assert (source/(backups.DATABASE+'-wal')).exists()
    before = canonical_hashes(source)
    archive, manifest = archive_of(workspace, tmp_path)
    assert manifest['recordCounts']['analysis'] == 2
    assert canonical_hashes(source) == before
    destination = tmp_path/'wal-restored'
    backups.restore_workspace(archive, destination, quiesced=True)
    with closing(sqlite3.connect(destination/backups.DATABASE)) as db:
        assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 6
    assert not (destination/(backups.DATABASE+'-wal')).exists()


def test_legacy_analysis_without_retained_or_existing_capture_remains_recoverable(workspace, tmp_path):
    source, _ = workspace
    for capture_id in [None, str(uuid4())]:
        record = store.put('analysis', {'source': 'Legacy fixture', 'captureId': capture_id})
        (source/(record['id']+'.json')).write_text(json.dumps(record))
    archive, manifest = archive_of(workspace, tmp_path)
    assert manifest['recordCounts']['analysis'] == 3
    backups.restore_workspace(archive, tmp_path/'legacy-restored', stopped=True)


def test_large_canonical_capture_is_streamed_and_recovered_exactly(workspace, tmp_path, monkeypatch):
    source, ids = workspace
    payload = (source/(ids['capture']+'.iq')).read_bytes()*600
    imported = captures.import_capture(UploadFile(io.BytesIO(payload), filename='synthetic-long-backup.bin'),
                                       captures.ImportOptions(format='raw', datatype='cf32_le', sampleRate=8000,
                                                              sessionId='synthetic-large-backup', source='Repeated generated fixture, not satellite RF'))
    requests = []; original = backups._HashingReader.read
    def recording(reader, size):
        requests.append(size)
        return original(reader, size)
    monkeypatch.setattr(backups._HashingReader, 'read', recording)
    archive, manifest = archive_of(workspace, tmp_path)
    assert len(requests) > 10 and all(0 < size <= backups.CHUNK for size in requests)
    entry = next(entry for entry in manifest['files'] if entry['path'] == imported['id']+'.iq')
    assert entry['bytes'] == len(payload) > backups.CHUNK and entry['sha256'] == imported['hash']
    destination = tmp_path/'large-restored'
    backups.restore_workspace(archive, destination, stopped=True)
    assert digest(destination/(imported['id']+'.iq')) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('state', ['queued', 'running', 'cancelling'])
def test_backup_refuses_nonterminal_jobs_without_leaving_an_archive(workspace, tmp_path, state):
    source, _ = workspace
    store.put('job', {'kind': 'batch', 'state': state})
    before = canonical_hashes(source)
    output = tmp_path/'refused.tar'
    with pytest.raises(ValueError, match='jobs'):
        backups.backup_workspace(source, output, stopped=True)
    assert not output.exists() and canonical_hashes(source) == before
    assert not list(tmp_path.glob('.satmod-backup-*'))


def test_operator_acknowledgement_and_no_overwrite_are_required(workspace, tmp_path):
    source, _ = workspace
    path = tmp_path/'backup.tar'
    with pytest.raises(ValueError, match='acknowledgement'):
        backups.backup_workspace(source, path)
    path.write_bytes(b'Operator-owned existing file')
    with pytest.raises(ValueError, match='already exists'):
        backups.backup_workspace(source, path, stopped=True)
    assert path.read_bytes() == b'Operator-owned existing file'
    with pytest.raises(ValueError, match='acknowledgement'):
        backups.restore_workspace(path, tmp_path/'restored')


def test_backup_rejects_repository_or_source_output(workspace, tmp_path):
    source, _ = workspace
    for output in [source/'backup.tar', backups.REPOSITORY/('synthetic-refused-'+str(uuid4())+'.tar')]:
        with pytest.raises(ValueError, match='outside'):
            backups.backup_workspace(source, output, stopped=True)
        assert not output.exists()


@pytest.mark.parametrize('corruption', ['missing', 'hash', 'orphan', 'temporary', 'report-identity'])
def test_backup_rejects_live_referenced_artifact_mismatches(workspace, tmp_path, corruption):
    source, ids = workspace
    iq = source/(ids['capture']+'.iq')
    if corruption == 'missing':
        iq.unlink()
    elif corruption == 'hash':
        iq.write_bytes(b'\0'*iq.stat().st_size)
    elif corruption == 'orphan':
        (source/(str(uuid4())+'.json')).write_text('{}')
    elif corruption == 'temporary':
        (source/(str(uuid4())+'.upload')).write_bytes(b'Uncommitted synthetic upload')
    else:
        (source/(ids['analysis']+'.json')).write_text(json.dumps({'id': str(uuid4())}))
    with pytest.raises(ValueError):
        backups.backup_workspace(source, tmp_path/'invalid.tar', stopped=True)
    assert not (tmp_path/'invalid.tar').exists()


@pytest.mark.parametrize('corruption', ['checksum', 'missing-manifest', 'missing-file', 'unknown-file', 'duplicate-member', 'duplicate-manifest-entry', 'counts'])
def test_restore_rejects_malformed_archives_without_publishing(workspace, tmp_path, corruption):
    archive, _ = archive_of(workspace, tmp_path)
    parts = archive_parts(archive)
    if corruption == 'checksum':
        name, payload = parts[0]; parts[0] = (name, payload+b'corrupt')
    elif corruption == 'missing-manifest':
        parts = [(name, payload) for name, payload in parts if name != backups.MANIFEST]
    elif corruption == 'missing-file':
        parts = parts[1:]
    elif corruption == 'unknown-file':
        parts.append(('unexpected.txt', b'not a managed artifact'))
    elif corruption == 'duplicate-member':
        parts.append(parts[0])
    else:
        manifest = json.loads(dict(parts)[backups.MANIFEST])
        if corruption == 'duplicate-manifest-entry':
            manifest['files'].append(deepcopy(manifest['files'][0]))
        else:
            manifest['recordCounts']['analysis'] += 1
        parts = [(name, json.dumps(manifest).encode() if name == backups.MANIFEST else payload) for name, payload in parts]
    write_parts(archive, parts)
    assert_refused_unchanged(archive, tmp_path/'refused')


@pytest.mark.parametrize('name,kind', [('../outside', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                                    ('subdir/file', tarfile.REGTYPE), ('link', tarfile.SYMTYPE),
                                    ('hardlink', tarfile.LNKTYPE), ('directory', tarfile.DIRTYPE)])
def test_restore_never_extracts_paths_links_or_unknown_entry_types(tmp_path, name, kind):
    archive = tmp_path/'unsafe.tar'
    with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as container:
        member = tarfile.TarInfo(name); member.type = kind; member.linkname = '../outside'
        container.addfile(member, io.BytesIO())
    assert_refused_unchanged(archive, tmp_path/'refused')
    assert not (tmp_path.parent/'outside').exists()


def test_extended_tar_headers_are_rejected(tmp_path):
    archive = tmp_path/'pax.tar'
    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT) as container:
        member = tarfile.TarInfo(backups.DATABASE)
        member.pax_headers = {'comment': 'Synthetic unsupported extended tar header'}
        container.addfile(member, io.BytesIO())
    assert_refused_unchanged(archive, tmp_path/'pax-refused')


def test_compression_oversized_members_entry_limits_and_truncation_are_bounded(workspace, tmp_path):
    archive, _ = archive_of(workspace, tmp_path)
    compressed = tmp_path/'compressed.tar.gz'; compressed.write_bytes(gzip.compress(archive.read_bytes()))
    assert_refused_unchanged(compressed, tmp_path/'gzip-refused')
    assert_refused_unchanged(archive, tmp_path/'entry-refused', max_entries=2)
    assert_refused_unchanged(archive, tmp_path/'bytes-refused', max_bytes=100)
    bomb = tmp_path/'oversized.tar'
    member = tarfile.TarInfo(backups.DATABASE); member.size = backups.DEFAULT_MAX_BYTES+1
    bomb.write_bytes(member.tobuf(format=tarfile.USTAR_FORMAT)+b'\0'*1024)
    assert_refused_unchanged(bomb, tmp_path/'bomb-refused')
    truncated = tmp_path/'truncated.tar'; truncated.write_bytes(archive.read_bytes()[:1000])
    assert_refused_unchanged(truncated, tmp_path/'truncated-refused')
    appended = tmp_path/'appended.tar'; appended.write_bytes(archive.read_bytes()+archive.read_bytes())
    assert_refused_unchanged(appended, tmp_path/'appended-refused')


def test_existing_destination_data_and_archive_are_preserved(workspace, tmp_path):
    archive, _ = archive_of(workspace, tmp_path)
    destination = tmp_path/'existing'; destination.mkdir()
    marker = destination/'operator-data'; marker.write_bytes(b'Keep this original data')
    before = digest(archive)
    with pytest.raises(ValueError, match='empty'):
        backups.restore_workspace(archive, destination, stopped=True)
    assert marker.read_bytes() == b'Keep this original data' and digest(archive) == before
    assert not list(tmp_path.glob('.satmod-restore-*'))


@pytest.mark.skipif(os.name == 'nt', reason='Atomic replacement of an existing empty directory is a POSIX operation; Windows uses a new path.')
def test_existing_empty_destination_is_atomically_replaced(workspace, tmp_path):
    archive, _ = archive_of(workspace, tmp_path)
    destination = tmp_path/'empty'; destination.mkdir()
    backups.restore_workspace(archive, destination, stopped=True)
    assert (destination/backups.DATABASE).is_file()


def test_publish_failures_leave_original_and_archive_unchanged(workspace, tmp_path, monkeypatch):
    source, _ = workspace
    before = canonical_hashes(source)
    def failure(*args, **kwargs):
        raise OSError('Synthetic publication failure')
    with monkeypatch.context() as patch:
        patch.setattr(backups.os, 'link', failure)
        with pytest.raises(OSError, match='Synthetic'):
            backups.backup_workspace(source, tmp_path/'failed.tar', stopped=True)
    assert canonical_hashes(source) == before and not (tmp_path/'failed.tar').exists()
    assert not list(tmp_path.glob('.satmod-backup-*'))
    archive, _ = archive_of(workspace, tmp_path)
    archive_hash = digest(archive)
    for existing in [False, True]:
        destination = tmp_path/('restored-empty' if existing else 'restored-new')
        if existing:
            destination.mkdir()
        with monkeypatch.context() as patch:
            patch.setattr(backups.os, 'replace', failure)
            with pytest.raises(OSError, match='Synthetic'):
                backups.restore_workspace(archive, destination, stopped=True)
        assert destination.exists() == existing
        if existing:
            assert not list(destination.iterdir())
    assert digest(archive) == archive_hash and canonical_hashes(source) == before
    assert not list(tmp_path.glob('.satmod-restore-*'))


def test_exclusive_publication_does_not_replace_an_archive_created_during_backup(workspace, tmp_path, monkeypatch):
    source, _ = workspace
    output = tmp_path/'raced.tar'
    def race(staging, destination):
        Path(destination).write_bytes(b'Archive independently created by the operator')
        raise FileExistsError('Synthetic concurrent archive creation')
    monkeypatch.setattr(backups.os, 'link', race)
    with pytest.raises(FileExistsError):
        backups.backup_workspace(source, output, stopped=True)
    assert output.read_bytes() == b'Archive independently created by the operator'
    assert not list(tmp_path.glob('.satmod-backup-*'))


def test_restore_does_not_overwrite_data_that_arrives_during_validation(workspace, tmp_path, monkeypatch):
    archive, _ = archive_of(workspace, tmp_path)
    destination = tmp_path/'new-data'
    original = backups._validate_artifacts
    def race(*args, **kwargs):
        original(*args, **kwargs)
        destination.mkdir()
        (destination/'operator-data').write_bytes(b'Keep concurrently created data')
    monkeypatch.setattr(backups, '_validate_artifacts', race)
    with pytest.raises(ValueError, match='changed'):
        backups.restore_workspace(archive, destination, stopped=True)
    assert (destination/'operator-data').read_bytes() == b'Keep concurrently created data'
    assert not list(tmp_path.glob('.satmod-restore-*'))


def test_a_writer_violating_quiescence_cannot_publish_a_mixed_backup(workspace, tmp_path, monkeypatch):
    source, _ = workspace
    original = backups._HashingReader.read; changed = False
    def changing(reader, size):
        nonlocal changed
        data = original(reader, size)
        if not changed:
            changed = True
            store.put('job', {'kind': 'batch', 'state': 'failed', 'error': 'Synthetic concurrent write'})
        return data
    monkeypatch.setattr(backups._HashingReader, 'read', changing)
    with pytest.raises(ValueError, match='changed'):
        backups.backup_workspace(source, tmp_path/'mixed.tar', quiesced=True)
    assert not (tmp_path/'mixed.tar').exists()


@pytest.mark.parametrize('corruption', ['report-id', 'capture-hash', 'database-schema', 'dataset-frame',
                                       'dataset-capture', 'dataset-source-hash', 'active-job'])
def test_valid_manifest_cannot_hide_invalid_referenced_files(workspace, tmp_path, corruption):
    archive, _ = archive_of(workspace, tmp_path)
    _, ids = workspace
    files = dict(archive_parts(archive))
    if corruption == 'report-id':
        report = json.loads(files[ids['analysis']+'.json']); report['id'] = str(uuid4())
        replace_part(archive, ids['analysis']+'.json', json.dumps(report).encode())
    elif corruption == 'capture-hash':
        replace_part(archive, ids['capture']+'.iq', b'\0'*len(files[ids['capture']+'.iq']))
    else:
        database = tmp_path/'edited.sqlite3'; database.write_bytes(files[backups.DATABASE])
        with closing(sqlite3.connect(database)) as db:
            if corruption == 'database-schema':
                db.execute('CREATE TABLE unknown_data (id TEXT)')
            else:
                record_id = ids['job'] if corruption == 'active-job' else ids['dataset']
                payload = json.loads(db.execute('SELECT payload FROM records WHERE id=?', (record_id,)).fetchone()[0])
                if corruption == 'active-job':
                    payload['state'] = 'queued'
                elif corruption == 'dataset-capture':
                    payload['entries'][0]['captureId'] = str(uuid4())
                elif corruption == 'dataset-source-hash':
                    payload['entries'][0]['canonicalSourceHash'] = '0'*64
                else:
                    payload['entries'][0]['frameHash'] = '0'*64
                db.execute('UPDATE records SET payload=? WHERE id=?', (json.dumps(payload), record_id))
            db.commit()
        replace_part(archive, backups.DATABASE, database.read_bytes())
    assert_refused_unchanged(archive, tmp_path/'refused')


def test_symlink_source_members_archives_and_destinations_are_rejected(workspace, tmp_path):
    source, ids = workspace
    target = tmp_path/'operator-secret'; target.write_bytes(b'Never archive linked files')
    member = source/(ids['capture']+'.iq'); original = member.read_bytes(); member.unlink()
    try:
        member.symlink_to(target)
    except (OSError, NotImplementedError):
        member.write_bytes(original)
        pytest.skip('Symlink creation is unavailable on this platform.')
    with pytest.raises(ValueError):
        backups.backup_workspace(source, tmp_path/'linked.tar', stopped=True)
    member.unlink(); member.write_bytes(original)
    archive, _ = archive_of(workspace, tmp_path)
    archive_link = tmp_path/'archive-link'; archive_link.symlink_to(archive)
    with pytest.raises(ValueError, match='Symlink'):
        backups.restore_workspace(archive_link, tmp_path/'new', stopped=True)
    destination = tmp_path/'destination-link'; destination.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        backups.restore_workspace(archive, destination, stopped=True)
    assert target.read_bytes() == b'Never archive linked files'


def test_cli_help_acknowledgement_and_roundtrip(workspace, tmp_path):
    source, _ = workspace
    script = Path(__file__).resolve().parents[1]/'scripts/workspace_backup.py'
    help_result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert help_result.returncode == 0 and '--stopped' in help_result.stdout and 'unencrypted' in help_result.stdout
    refused = subprocess.run([sys.executable, str(script), 'backup', '--source', str(source), '--output', str(tmp_path/'backup.tar')], capture_output=True, text=True)
    assert refused.returncode == 2 and not (tmp_path/'backup.tar').exists()
    archive = tmp_path/'cli.tar'
    result = subprocess.run([sys.executable, str(script), 'backup', '--source', str(source), '--output', str(archive), '--stopped'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['recordCounts']['analysis'] == 1
    restored = subprocess.run([sys.executable, str(script), 'restore', '--archive', str(archive), '--destination', str(tmp_path/'cli-restored'), '--stopped'], capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout)['recordCounts']['comparison'] == 1
