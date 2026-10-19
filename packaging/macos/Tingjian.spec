# Run from the repository root with: python -m PyInstaller packaging/macos/Tingjian.spec
from PyInstaller.utils.hooks import collect_dynamic_libs
from pathlib import Path

root = Path(SPECPATH).resolve().parents[1]
at = lambda path: str(root / path)

a = Analysis(
    [at('packaging/macos/launcher.py')],
    pathex=[str(root)],
    binaries=[(at('native/ffmpeg'), 'native')] + collect_dynamic_libs('onnxruntime'),
    datas=[(at('web'), 'web'), (at('README_MAC.md'), '.'), (at('LICENSE'), '.'),
           (at('THIRD_PARTY_NOTICES.md'), '.'), (at('FFMPEG_SOURCE.txt'), '.'),
           (at('FFMPEG_BUILD.txt'), '.'), (at('licenses'), 'licenses'),
           (at('third_party_sources'), 'third_party_sources')],
    hookspath=[at('packaging/macos/hooks')],
    excludes=['onnxruntime.datasets', 'onnxruntime.tools', 'onnxruntime.transformers'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Tingjian',
          console=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name='Tingjian')
app = BUNDLE(coll, name='Tingjian.app', bundle_identifier='com.yuhan2233.tingjian',
             info_plist={'CFBundleDisplayName': '听见',
                         'LSMinimumSystemVersion': '14.0',
                         'NSHighResolutionCapable': True})
