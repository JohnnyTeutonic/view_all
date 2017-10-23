from distutils.core import setup, find_packages

files = ["view_all.py","clickable_image.py"]

setup(name = "view_all",
    version = "0.1-dev",
    description = "A 3D Viewer for neuronal volumetric data.",
    author = "Jonathan Reich",
    author_email = "jreich@student.unimelb.edu.au",
    url = "https://github.com/JohnnyTeutonic/view_all",
    install_requires = ['numpy', 'scikit-image', 'numba', 'gala', 'matplotlib'],
    packages = [find_packages(), "viewer"],
    scripts = ["runner"],
     )
