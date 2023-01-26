import setuptools

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='micrograph-modeller',
    version='0.1',
    description='Simulation of electron micrographs with cytosolic macromolecules and vesicle systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    author='McHaillet (Marten Chaillet)',
    author_email='martenchaillet@gmail.com',
    url='https://github.com/McHaillet/micrograph-modeller',
    platforms=['any'],
    python_requires='>=3.8',
    install_requires=[
        'gputil',
        'joblib>=1.0.1',
        'numpy',
        'numba',
        'matplotlib',
        'mrcfile',
        'pyvista',
        'scipy',
        'threadpoolctl',
        'tqdm',
        'voltools>=0.4.6'
    ],
    packages=setuptools.find_packages(),
    package_data={
        'micrographmodeller.detectors': ['*.csv'],
        'micrographmodeller.membrane_models': ['*.pdb'],
    },
    # include_package_data=True,
    test_suite='tests',
    scripts=[
        'micrographmodeller/bin/micrograph-modeller.py',
        'micrographmodeller/bin/mm-potential.py',
        'micrographmodeller/bin/mm-vesicle.py'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3 :: Only',
    ]
)
