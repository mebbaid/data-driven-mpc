# python/setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            import subprocess
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(ext.name for ext in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if sys.platform.startswith('win'):
             cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
             if sys.maxsize > 2**32:
                 cmake_args += ['-A', 'x64']
                 build_args += ['--', '/m']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

setup(
    name='pyddmpc',
    version='0.1',
    author='Mohamed Elobaid',
    author_email='mohamed.elobaid@iit.it',
    description='Data-Driven MPC Python Bindings',
    long_description='',
    ext_modules=[CMakeExtension('pyddmpc', sourcedir='../')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=['pyddmpc']
)