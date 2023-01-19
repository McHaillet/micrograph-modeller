import setuptools

from micrographmodeller import __version__

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='micrograph-modeller',
    version=__version__,
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
        'cupy>=10.6.0',
        'gputil',
        'joblib>=1.0.1',
        'numpy',
        'numba',
        'mrcfile',
        'pyvista',
        'scipy',
        'threadpoolctl',
        'tqdm',
        'voltools>=0.4.6'
    ],
    packages=setuptools.find_packages(),
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    scripts=['micrographmodeller/micrographmodeller.py',
             'micrographmodeller/potential.py',
             'micrographmodeller/membrane.py']
)
