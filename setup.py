import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name = "augmentRNA",
    version = "0.0.1",
    author = "Christian de Frondeville",
    description = "Lightweight package that allows for the generation of augmented RNA-seq data from a base dataset, for expanding training datasets or large-scale dataset analysis.",
    long_description = long_description,
    long_description_content_type='text/markdown',
    packages = ["augmentRNA"],
    install_requires=['scipy', 'numpy', 'tqdm']
)
