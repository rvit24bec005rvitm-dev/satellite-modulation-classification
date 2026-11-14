r"""Offline whole-workspace backup/restore; stop the service before either command.

Examples (run from the repository root; archive parent must already exist):
  python scripts/workspace_backup.py backup --source /srv/satmod-data \
    --output /private/operator-backups/satmod-2026-11-14.tar --stopped
  python scripts/workspace_backup.py restore \
    --archive /private/operator-backups/satmod-2026-11-14.tar \
    --destination /srv/satmod-restored --stopped

Use --quiesced only when every process writing that workspace is independently
paused. The flag is an operator acknowledgement, not a service-stop command.
No backup or restore HTTP API exists. Archives are uncompressed, unencrypted,
private-mode tar files: keep them in an operator-controlled private location
outside this repository and source workspace. Existing archives are never
overwritten. Restore validates a sibling staging directory before atomic rename;
use a new destination path (or an empty directory on POSIX). No merge is offered.
Restored files are owned by the invoking account; restore as the service owner.
SQLite WAL/SHM files are consolidated by SQLite's backup API, not archived.
Only schema-v1 metadata and referenced UUID .iq/.json artifacts are accepted;
unfinished jobs, temporary/unknown files and invalid references are rejected.
Default canonical data limit is 1 GiB with bounded tar/manifest overhead, at most
10,002 archive entries. --max-bytes/--max-entries change these explicit limits.
Report identity metadata must appear within its first 1 MiB, as in existing
application reports. Model weights, configuration, credentials and frontend
files are outside workspace backup scope and must be provisioned separately.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.workspace_backups import (  # noqa: E402
    DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES, backup_workspace, restore_workspace,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['backup', 'restore']:
        command = commands.add_parser(name)
        acknowledgement = command.add_mutually_exclusive_group(required=True)
        acknowledgement.add_argument('--stopped', action='store_true', help='I stopped every service writing this workspace.')
        acknowledgement.add_argument('--quiesced', action='store_true', help='I independently quiesced every workspace writer.')
        command.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES)
        command.add_argument('--max-entries', type=int, default=DEFAULT_MAX_ENTRIES)
        if name == 'backup':
            command.add_argument('--source', required=True, type=Path)
            command.add_argument('--output', required=True, type=Path)
        else:
            command.add_argument('--archive', required=True, type=Path)
            command.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args(argv)
    limits = {'stopped': args.stopped, 'quiesced': args.quiesced, 'max_bytes': args.max_bytes, 'max_entries': args.max_entries}
    try:
        if args.command == 'backup':
            result = backup_workspace(args.source, args.output, **limits)
        else:
            result = restore_workspace(args.archive, args.destination, **limits)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Workspace {args.command} refused: {exc}\n')
    print(json.dumps({'operation': args.command, 'format': result['format'], 'recordCounts': result['recordCounts'],
                      'files': len(result['files']), 'canonicalBytes': result['totalBytes']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
