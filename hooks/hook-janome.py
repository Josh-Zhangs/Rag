from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules('janome')
datas = collect_data_files('janome', include_py_files=True)