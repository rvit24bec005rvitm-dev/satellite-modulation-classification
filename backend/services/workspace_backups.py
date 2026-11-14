"""Private, offline backups of schema-v1 SatModClass managed workspaces.

The operator must stop the service or otherwise quiesce every writer. SQLite's
backup API consolidates committed WAL data; copied sample/report bytes are never
rewritten. This is an integrity backup, not encryption or an authenticity proof.
No web endpoint is provided. See scripts/workspace_backup.py --help for commands.
"""
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
from uuid import UUID


FORMAT = 'satmod-workspace-backup-v1'
DATABASE = 'workspace.sqlite3'
MANIFEST = 'manifest.json'
DEFAULT_MAX_BYTES = 1024**3
DEFAULT_MAX_ENTRIES = 10002
MAX_MANIFEST_BYTES = 4*1024**2
CHUNK = 1024**2
KINDS = {'capture', 'analysis', 'dataset', 'job', 'comparison'}
TERMINAL = {'completed', 'failed', 'cancelled'}
REPOSITORY = Path(__file__).resolve().parents[2]
CANONICAL = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.(?:iq|json)\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def _limits(max_bytes, max_entries):
    if type(max_bytes) is not int or max_bytes < 1 or type(max_entries) is not int or not 2 <= max_entries <= 100002:
        raise ValueError('Use a positive byte limit and an entry limit between 2 and 100002.')
    return max_bytes + max_entries*1024 + MAX_MANIFEST_BYTES + 10240


def _path(value):
    original = Path(value).expanduser()
    if '..' in original.parts:
        raise ValueError('Parent-traversal path components are not accepted.')
    path = Path(os.path.abspath(original))
    # Check before resolve() so a symlink cannot hide behind normalization.
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Symlinks are not accepted for workspace, archive or destination paths.')
    return path


def _uuid(value):
    if not isinstance(value, str):
        raise ValueError('Invalid managed record identifier.')
    try:
        if str(UUID(value)) != value:
            raise ValueError('Noncanonical managed record identifier.')
    except (ValueError, AttributeError) as exc:
        raise ValueError('Invalid canonical managed record identifier.') from exc
    return value


def _name(value):
    if value == DATABASE:
        return value
    if not isinstance(value, str) or not CANONICAL.fullmatch(value):
        raise ValueError('Unknown or unsafe workspace archive entry.')
    _uuid(value.rsplit('.', 1)[0])
    return value


def _regular(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f'Only private regular files are supported: {path.name}')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    stream = os.fdopen(descriptor, 'rb')
    opened = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or not stat.S_ISREG(opened.st_mode):
        stream.close()
        raise ValueError('Workspace file changed while opening it.')
    return stream


def _fingerprint(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError(f'Only private regular files are supported: {path.name}')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _inventory(root, max_entries):
    files = {}
    for path in root.iterdir():
        if path.name in {DATABASE+'-wal', DATABASE+'-shm'}:
            _fingerprint(path)
            continue
        _name(path.name)
        files[path.name] = _fingerprint(path)
        if len(files)+1 > max_entries:
            raise ValueError('Workspace exceeds the archive entry limit.')
    if DATABASE not in files:
        raise ValueError('Workspace database is missing.')
    return files


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object keys are not accepted.')
        result[key] = value
    return result


def _json(value):
    try:
        return json.loads(value, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON is not accepted.')))
    except (RecursionError, UnicodeError) as exc:
        raise ValueError('Invalid or excessively nested workspace JSON.') from exc


@contextmanager
def _readonly(path):
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA trusted_schema=OFF')
        yield db
    finally:
        db.close()


def _catalog(path):
    """Validate the known metadata schema without depending on current model files."""
    try:
        with _readonly(path) as db:
            if db.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('Only managed workspace schema version 1 can be restored.')
            objects = db.execute("SELECT name,type FROM sqlite_master WHERE type IN ('table','view','trigger')").fetchall()
            if objects != [('records', 'table')]:
                raise ValueError('Unknown workspace database objects are not accepted.')
            columns = [(row[1], row[2], row[3], row[5]) for row in db.execute('PRAGMA table_xinfo(records)')]
            if columns != [('id', 'TEXT', 0, 1), ('kind', 'TEXT', 1, 0), ('created_at', 'TEXT', 1, 0), ('payload', 'TEXT', 1, 0)]:
                raise ValueError('Workspace records schema does not match this application.')
            if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Workspace SQLite integrity check failed.')
            count = db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            if count > 10000:
                raise ValueError('Workspace exceeds the managed record limit.')
            required = {}; captures = {}; counts = Counter()
            for record_id, kind, created, encoded in db.execute('SELECT id,kind,created_at,payload FROM records'):
                _uuid(record_id)
                if kind not in KINDS or not isinstance(encoded, str) or len(encoded.encode()) > 2*1024**2:
                    raise ValueError('Unsupported workspace record kind or payload size.')
                payload = _json(encoded)
                if not isinstance(payload, dict) or payload.get('id') != record_id or payload.get('createdAt') != created:
                    raise ValueError('Workspace record identity does not match its metadata row.')
                counts[kind] += 1
                if kind == 'capture':
                    n = payload.get('sampleCount'); digest = payload.get('hash')
                    if type(n) is not int or n < 128 or not isinstance(digest, str) or not SHA256.fullmatch(digest):
                        raise ValueError('Capture has invalid sample count or canonical hash.')
                    captures[record_id] = {key: payload.get(key) for key in ['hash', 'sampleCount', 'sourceHash']}
                    required[record_id+'.iq'] = ('capture', n*8, digest)
                elif kind in {'analysis', 'comparison'}:
                    required[record_id+'.json'] = (kind, None, None)
                elif kind == 'job':
                    if payload.get('state') not in TERMINAL:
                        raise ValueError('Workspace has queued, running, cancelling or unknown-state jobs; stop/cancel and wait before backup.')
                    job_kind = payload.get('kind')
                    if job_kind not in {'batch', 'evaluation', 'capture-import', 'dataset-import'}:
                        raise ValueError('Unsupported managed job kind.')
                    if payload['state'] == 'completed' and job_kind in {'batch', 'evaluation'}:
                        required[record_id+'.json'] = ('job', None, None)
            return {'counts': {kind: counts[kind] for kind in sorted(KINDS)}, 'required': required, 'captures': captures}
    except sqlite3.DatabaseError as exc:
        raise ValueError('Workspace database is invalid or incompatible.') from exc


def _report_identity(path, field, expected):
    # Current and legacy application reports write id/jobId before their large
    # analysis/prediction arrays. Read a bounded header, not an entire long job.
    with _regular(path) as stream:
        raw = stream.read(CHUNK)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        # The last UTF-8 code point may straddle the bounded header boundary.
        if exc.end != len(raw):
            raise ValueError('Report is not UTF-8 JSON.') from exc
        text = raw[:exc.start].decode('utf-8')
    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    index = 0
    def whitespace(at):
        while at < len(text) and text[at] in ' \t\r\n':
            at += 1
        return at
    index = whitespace(index)
    if not text[index:].startswith('{'):
        raise ValueError('Managed report is not a JSON object.')
    index += 1
    try:
        while True:
            key, index = decoder.raw_decode(text, whitespace(index))
            if not isinstance(key, str) or text[whitespace(index):whitespace(index)+1] != ':':
                break
            value, index = decoder.raw_decode(text, whitespace(whitespace(index)+1))
            if key == field:
                if value != expected:
                    raise ValueError('Report identity does not match its referenced file.')
                return
            index = whitespace(index)
            if text[index:index+1] != ',':
                break
            index += 1
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Report identity is missing or outside the supported 1 MiB metadata header.') from exc
    raise ValueError('Report identity is missing from its metadata header.')


def _validate_artifacts(root, database, catalog, entries):
    expected = {DATABASE, *catalog['required']}
    if set(entries) != expected:
        raise ValueError('Referenced workspace files are missing or orphaned files are present.')
    for name, (kind, size, digest) in catalog['required'].items():
        if size is not None and (entries[name]['bytes'] != size or entries[name]['sha256'] != digest):
            raise ValueError('Capture file length/hash does not match its metadata reference.')
        if kind != 'capture':
            _report_identity(root/name, 'jobId' if kind == 'job' else 'id', name[:-5])
    # Deleted recordings are legitimate soft historical references in analysis,
    # comparison and terminal job reports. Registered dataset references are hard.
    with _readonly(database) as db:
        for encoded, in db.execute("SELECT payload FROM records WHERE kind='dataset'"):
            dataset = _json(encoded)
            rows = dataset.get('entries', [])
            if not isinstance(rows, list) or len(rows) > 5000:
                raise ValueError('Invalid registered dataset entries.')
            for entry in rows:
                if not isinstance(entry, dict) or entry.get('captureId') not in catalog['captures']:
                    raise ValueError('Registered dataset references a missing capture.')
                capture = catalog['captures'][entry['captureId']]
                start = entry.get('start')
                if type(start) is not int or start < 0 or start+128 > capture['sampleCount']:
                    raise ValueError('Registered dataset window lies outside its capture.')
                if entry.get('canonicalSourceHash') not in (None, capture['hash']) or entry.get('sourceHash') not in (None, capture['sourceHash']):
                    raise ValueError('Registered dataset capture hash does not match.')
                with _regular(root/(entry['captureId']+'.iq')) as stream:
                    stream.seek(start*8); frame = stream.read(128*8)
                if len(frame) != 128*8 or entry.get('frameHash') != hashlib.sha256(frame).hexdigest():
                    raise ValueError('Registered dataset frame hash does not match its capture.')


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream; self.hash = hashlib.sha256(); self.size = 0

    def read(self, size):
        data = self.stream.read(min(size, CHUNK))
        self.hash.update(data); self.size += len(data)
        return data


def _member(name, size):
    info = tarfile.TarInfo(name)
    info.size = size; info.mode = 0o600; info.mtime = 0
    return info


def backup_workspace(source, output, *, stopped=False, quiesced=False,
                     max_bytes=DEFAULT_MAX_BYTES, max_entries=DEFAULT_MAX_ENTRIES):
    """Create a new private uncompressed archive outside the repository/source."""
    if not (stopped or quiesced):
        raise ValueError('Explicit stopped/quiesced writer acknowledgement is required.')
    archive_limit = _limits(max_bytes, max_entries)
    source = _path(source); output = _path(output)
    if not source.is_dir() or not output.parent.is_dir():
        raise ValueError('Source and archive parent directories must already exist.')
    if output == source or source in output.parents or output == REPOSITORY or REPOSITORY in output.parents:
        raise ValueError('Backup archives must be outside the source workspace and repository.')
    if output.exists():
        raise ValueError('Backup archive already exists; overwriting is prohibited.')
    before = _inventory(source, max_entries)
    if sum(value[2] for value in before.values()) > max_bytes:
        raise ValueError('Workspace exceeds the canonical byte limit.')
    temporary = Path(tempfile.mkdtemp(prefix='.satmod-backup-', dir=output.parent))
    try:
        snapshot = temporary/DATABASE
        with _readonly(source/DATABASE) as original:
            version = original.execute('PRAGMA data_version').fetchone()[0]
            page_size = original.execute('PRAGMA page_size').fetchone()[0]
            with closing(sqlite3.connect(snapshot)) as destination:
                def progress(status, remaining, total):
                    if total*page_size > max_bytes:
                        raise ValueError('Consolidated database exceeds the byte limit.')
                original.backup(destination, pages=256, progress=progress)
                destination.execute('PRAGMA journal_mode=DELETE')
            os.chmod(snapshot, 0o600)
            catalog = _catalog(snapshot)
            if set(before) != {DATABASE, *catalog['required']}:
                raise ValueError('Referenced workspace files are missing or orphaned files are present.')
            entries = {}; total = 0; staged_archive = temporary/'archive.tar'
            with staged_archive.open('xb') as stream:
                os.chmod(staged_archive, 0o600)
                with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                    for name in sorted(before):
                        path = snapshot if name == DATABASE else source/name
                        with _regular(path) as member:
                            size = os.fstat(member.fileno()).st_size
                            total += size
                            if total > max_bytes:
                                raise ValueError('Workspace exceeds the canonical byte limit.')
                            reader = _HashingReader(member)
                            archive.addfile(_member(name, size), reader)
                            if reader.size != size:
                                raise ValueError('Workspace file changed during backup.')
                            entries[name] = {'path': name, 'bytes': size, 'sha256': reader.hash.hexdigest()}
                    _validate_artifacts(source, snapshot, catalog, entries)
                    manifest = {'format': FORMAT, 'createdAt': datetime.now(timezone.utc).isoformat(),
                                'schemaVersion': 1, 'recordCounts': catalog['counts'], 'totalBytes': total,
                                'files': [entries[name] for name in sorted(entries)]}
                    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
                    if len(encoded) > MAX_MANIFEST_BYTES:
                        raise ValueError('Backup manifest exceeds its bounded size.')
                    archive.addfile(_member(MANIFEST, len(encoded)), io.BytesIO(encoded))
                stream.flush(); os.fsync(stream.fileno())
            if (original.execute('PRAGMA data_version').fetchone()[0] != version or
                    _inventory(source, max_entries) != before):
                raise ValueError('Workspace changed during backup; stop every writer and retry.')
        if staged_archive.stat().st_size > archive_limit:
            raise ValueError('Archive exceeds the byte limit including headers.')
        # Exclusive hard-link publication never replaces an existing archive,
        # including an archive created between the precheck and this operation.
        os.link(staged_archive, output)
        return manifest
    except (sqlite3.DatabaseError, tarfile.TarError) as exc:
        raise ValueError('Could not create a valid workspace backup.') from exc
    finally:
        shutil.rmtree(temporary)


def _manifest(value, max_bytes, max_entries):
    if not isinstance(value, dict) or set(value) != {'format', 'createdAt', 'schemaVersion', 'recordCounts', 'totalBytes', 'files'}:
        raise ValueError('Unsupported backup manifest fields.')
    if value['format'] != FORMAT or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise ValueError('Unsupported backup manifest version.')
    if not isinstance(value['createdAt'], str) or not isinstance(value['recordCounts'], dict) or set(value['recordCounts']) != KINDS:
        raise ValueError('Invalid backup manifest metadata.')
    if any(type(count) is not int or not 0 <= count <= 10000 for count in value['recordCounts'].values()):
        raise ValueError('Invalid backup record counts.')
    if not isinstance(value['files'], list) or not 1 <= len(value['files']) < max_entries:
        raise ValueError('Backup manifest exceeds the entry limit.')
    entries = {}; total = 0
    for entry in value['files']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Invalid manifest file descriptor.')
        name = _name(entry['path'])
        if name in entries or type(entry['bytes']) is not int or entry['bytes'] < 0 or not isinstance(entry['sha256'], str) or not SHA256.fullmatch(entry['sha256']):
            raise ValueError('Duplicate or invalid manifest file descriptor.')
        total += entry['bytes']; entries[name] = entry
    if DATABASE not in entries or type(value['totalBytes']) is not int or total != value['totalBytes'] or total > max_bytes:
        raise ValueError('Manifest byte total or database entry is invalid.')
    return entries


def restore_workspace(archive_path, destination, *, stopped=False, quiesced=False,
                      max_bytes=DEFAULT_MAX_BYTES, max_entries=DEFAULT_MAX_ENTRIES):
    """Validate into a private sibling directory, then atomically publish it."""
    if not (stopped or quiesced):
        raise ValueError('Explicit stopped/quiesced writer acknowledgement is required.')
    archive_limit = _limits(max_bytes, max_entries)
    archive_path = _path(archive_path); destination = _path(destination)
    if not destination.parent.is_dir():
        raise ValueError('Restore destination parent directory must already exist.')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Restore destination must be new or an empty directory; no existing data is overwritten.')
    with _regular(archive_path) as source:
        if os.fstat(source.fileno()).st_size > archive_limit:
            raise ValueError('Archive exceeds the configured byte limit.')
        staging = Path(tempfile.mkdtemp(prefix='.satmod-restore-', dir=destination.parent))
        try:
            entries = {}; seen = set(); manifest = None; total = 0; last_end = 0
            with tarfile.open(fileobj=source, mode='r:') as archive:
                for member in archive:
                    if len(seen) >= max_entries or member.name in seen:
                        raise ValueError('Duplicate archive entry or entry limit exceeded.')
                    seen.add(member.name)
                    if (member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.pax_headers or
                            member.sparse is not None or member.offset_data-member.offset != 512):
                        raise ValueError('Only plain regular USTAR files are accepted; links and extended headers are prohibited.')
                    if type(member.size) is not int or member.size < 0:
                        raise ValueError('Invalid archive member size.')
                    last_end = member.offset_data+((member.size+511)//512)*512
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError('Archive member data is unavailable.')
                    if member.name == MANIFEST:
                        if member.size > MAX_MANIFEST_BYTES:
                            raise ValueError('Backup manifest exceeds its bounded size.')
                        raw = extracted.read(MAX_MANIFEST_BYTES+1)
                        if len(raw) != member.size:
                            raise ValueError('Truncated archive manifest.')
                        manifest = _json(raw)
                        continue
                    name = _name(member.name)
                    total += member.size
                    if total > max_bytes:
                        raise ValueError('Archive exceeds the canonical byte limit.')
                    digest = hashlib.sha256(); remaining = member.size
                    descriptor = os.open(staging/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, 'wb') as output:
                        while remaining:
                            block = extracted.read(min(remaining, CHUNK))
                            if not block:
                                raise ValueError('Truncated archive member.')
                            output.write(block); digest.update(block); remaining -= len(block)
                        output.flush(); os.fsync(output.fileno())
                    entries[name] = {'path': name, 'bytes': member.size, 'sha256': digest.hexdigest()}
            # Reject concatenated archives, hidden trailing members and oversized
            # zero padding instead of silently ignoring bytes after the tar EOF.
            trailer = os.fstat(source.fileno()).st_size-last_end
            if not 1024 <= trailer <= 10240:
                raise ValueError('Invalid archive end padding.')
            source.seek(last_end)
            if any(source.read(trailer)):
                raise ValueError('Unexpected data after the archive entries.')
            expected = _manifest(manifest, max_bytes, max_entries)
            if entries != expected:
                raise ValueError('Archive files do not match manifest names, lengths or SHA-256 checksums.')
            catalog = _catalog(staging/DATABASE)
            if catalog['counts'] != manifest['recordCounts']:
                raise ValueError('Manifest record counts do not match the workspace database.')
            _validate_artifacts(staging, staging/DATABASE, catalog, entries)
            _path(destination)
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise ValueError('Restore destination changed; existing data is untouched.')
            # POSIX atomically replaces an empty destination directory. On a
            # platform that cannot do so, use an unused destination path instead.
            os.replace(staging, destination)
            return manifest
        except (sqlite3.DatabaseError, tarfile.TarError, EOFError) as exc:
            raise ValueError('Invalid or truncated workspace archive.') from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging)
