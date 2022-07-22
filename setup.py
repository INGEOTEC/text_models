# Copyright 2020 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup
import text_models
import numpy as np

with open('README.rst') as fpt:
    long_desc = fpt.read()

setup(
    name="text_models",
    description="""Text Models""",
    long_description=long_desc,
    version=text_models.__version__,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    url='https://github.com/ingeotec/text_models',
    author="Mario Graff",
    author_email="mgraffg@ieee.org",
    include_package_data=True,
    packages=['text_models', 'text_models/tests', 'text_models/inhouse'],
    zip_safe=False,    
    install_requires=['EvoMSA']
)
