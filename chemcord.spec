# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller chemcord.spec
# Run prepare_dataset.py and train_model.py first so data/ and models/ exist.
# The 20 MB raw tmQM_y.csv is deliberately NOT bundled -- the app only needs
# the processed table, the metadata and the trained models.
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['chemcord.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/tmqm_processed.csv.gz', 'data'),
        ('data/dataset_meta.json', 'data'),
        ('data/geometry_labels.csv.gz', 'data'),
        ('THIRD_PARTY_LICENSES.md', '.'),
        ('models', 'models'),
    ],
    hiddenimports=['logk', 'logk_ui', 'rdkit', 'rdkit.Chem', 'rdkit.Chem.Descriptors', 'rdkit.DataStructs'] + collect_submodules('sklearn.ensemble') + collect_submodules('sklearn.utils') + ['joblib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='chemcord',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
