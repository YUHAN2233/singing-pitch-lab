"""Preserve wheel licenses and corresponding libsndfile source in the app."""
from importlib import metadata
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import _soundfile_data
import soundfile

assert soundfile.__libsndfile_version__ == '1.2.2'
known_blobs = {'6312ba719c68a76b43beb369dceada056dd17bf9',
               '1d270e22e00f06c43be57c84cffb46a171ffbb37'}
libraries = list(Path(_soundfile_data.__file__).parent.glob('*.dylib'))
assert len(libraries) == 1, libraries
library_bytes = libraries[0].read_bytes()
blob_hash = hashlib.sha1(f'blob {len(library_bytes)}\0'.encode() + library_bytes).hexdigest()
assert blob_hash in known_blobs, 'libsndfile differs from the pinned source build'

out = Path('licenses')
out.mkdir(exist_ok=True)
manifest = []
for dist in metadata.distributions():
    name = dist.metadata['Name']
    copied = []
    for item in dist.files or []:
        if any(word in item.name.lower() for word in ('license', 'copying', 'notice', 'copyright')):
            source = Path(dist.locate_file(item))
            if source.is_file():
                target = out / name / str(item).replace('..', '_')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied.append(str(target))
    manifest.append({'name': name, 'version': dist.version, 'license_files': copied})
(out / 'distributions.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
for item in (Path(sys.base_prefix) / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE'):
    if item.is_file():
        shutil.copyfile(item, out / 'PYTHON-LICENSE.txt')
for number, item in enumerate(Path(sys.base_prefix).rglob('license.terms')):
    shutil.copyfile(item, out / f'TCL-TK-{number}-license.terms')

# SoundFile 0.13.1 pins this submodule containing the wheel's libsndfile build.
snd_commit = 'a3e6f9769d0c7e91d2d036cf0fdbe5b4bbf18b87'
base = f'https://raw.githubusercontent.com/bastibe/libsndfile-binaries/{snd_commit}'
sources = Path('third_party_sources')
sources.mkdir(exist_ok=True)
urls = {
    'libsndfile-1.2.2.tar.xz': 'https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz',
    'libogg-1.3.5.tar.gz': 'https://downloads.xiph.org/releases/ogg/libogg-1.3.5.tar.gz',
    'libvorbis-1.3.7.tar.gz': 'https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz',
    'flac-1.4.3.tar.xz': 'https://downloads.xiph.org/releases/flac/flac-1.4.3.tar.xz',
    'opus-1.4.tar.gz': 'https://downloads.xiph.org/releases/opus/opus-1.4.tar.gz',
    'mpg123-1.32.3.tar.bz2': 'https://downloads.sourceforge.net/project/mpg123/mpg123/1.32.3/mpg123-1.32.3.tar.bz2',
    'lame-3.100.tar.gz': 'https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz',
    'libsndfile-mac_build.sh': base + '/mac_build.sh',
    'libsndfile-darwin.cmake': base + '/darwin.cmake',
    'LIBSNDFILE-COPYING': base + '/COPYING',
}
records = []
for name, url in urls.items():
    target = sources / name
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', url, '-o', str(target)], check=True)
    assert target.stat().st_size > 100, name
    if '.tar.' in name:
        with tarfile.open(target) as archive:
            assert archive.getmembers(), name
    records.append({'file': name, 'url': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(sources / 'sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
(sources / 'README.txt').write_text(
    'FFmpeg is built from the included unmodified source archive with build_ffmpeg.sh; '
    'config.log records the complete configure command. It is LGPL 2.1 or later, '
    'without GPL/nonfree codecs or external codec libraries.\n\n'
    'SoundFile 0.13.1 bundles libsndfile 1.2.2, dynamically loaded. The included '
    'libsndfile-mac_build.sh and libsndfile-darwin.cmake come from its pinned '
    f'libsndfile-binaries submodule {snd_commit}. The listed source archives '
    'are the versions in that script. Reverse engineering to debug modifications '
    'to these LGPL components is permitted. You may rebuild and replace the '
    'libsndfile dylib under Tingjian.app/Contents/Frameworks/_soundfile_data; '
    'locally re-sign the modified app as needed. No Developer ID is required '
    'by this application. sources.json records downloaded files and hashes.\n', encoding='utf-8')
