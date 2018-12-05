#!/usr/bin/env python
# coding:utf-8
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
# cxfreeze_func.py build

import sys
import os.path

os.environ['TCL_LIBRARY'] = r'C:\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python36\tcl\tk8.6'
base = 'Win32GUI' if sys.platform=='win32' else None
# includes = [r"queue",r"numpy.core.multiarray"]
includes = [r"queue",r"numpy.core.multiarray"]
include_files = [r"C:\Python36\DLLs\tcl86t.dll",
                 r"C:\Python36\DLLs\tk86t.dll"]
executables = [
    # Executable(r'main_ui_multitask.py', targetName="seu_jwc_faker.exe",base=base)
    Executable(r'main_ui_multitask.py', targetName="Mistlab_edge.exe", base=base)
]
# packages = ["numpy"]
# options = {"build_exe": {"includes": includes, "include_files": include_files, "packages"}}
# packages = [r'numpy',r'requests',r"numpy.lib.format",r'tensorflow']
packages = [r'numpy',r'requests',r"numpy.lib.format"]
setup(name='qk',
      version = '1.0',
      description = 'seu qiangke',
      options={"build_exe": {"includes": includes, "include_files": include_files,"packages": packages}},
      executables = executables)