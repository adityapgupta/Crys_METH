"""Prepare an explicit, reviewable source overlay; never transfer anything itself."""
from pathlib import Path
import tarfile
from .common import ROOT, write_json, sha


def main():
    source_root = ROOT/'utils/eval/evaluation'
    files = {p for p in source_root.rglob('*') if p.is_file()
             and not any(x in ('data','outputs','cache','artifacts','__pycache__','build') for x in p.relative_to(source_root).parts)
             and p.suffix not in ('.pyc','.gz')
             and p.name != 'evaluation.ipynb'}
    # The repository-root notebook is the user-facing entry point. Include it in
    # cluster source bundles while keeping the former in-package copy archival.
    if (ROOT/'evaluation.ipynb').is_file():
        files.add(ROOT/'evaluation.ipynb')
    utility_root = ROOT/'utils/eval'
    files.update(p for p in utility_root.rglob('*.py') if '__pycache__' not in p.parts)
    dest = ROOT/'data/eval/results/source-overlay.tar.gz'
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(dest,'w:gz') as archive:
        for path in sorted(files):
            archive.add(path,arcname=str(path.relative_to(ROOT)).replace('\\','/'))
    write_json(dest.with_suffix('.manifest.json'),{'archive_sha256':sha(dest),
               'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}})
    print(dest)
    print('SHA256:',sha(dest))
    print('Source files:',len(files),'(no datasets, model weights, environments, or caches)')


if __name__=='__main__':
    main()
