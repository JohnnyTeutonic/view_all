from distutils.core import setup, find_packages

files = ["view_all.py","clickable_image_new.py"]

setup(name = "view_all",
    version = "0.1-dev",
    description = "A 3D Viewer for neuronal volumetric data.",
    author = "Jonathan Reich",
    author_email = "jreich@student.unimelb.edu.au",
    url = "https://github.com/JohnnyTeutonic/view_all"
    install_requires=['numpy', 'numba', 'gala', 'matplotlib']
    #Name the folder where your packages live:
    #If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found
    #recursively."
    packages = [find_packages()]
    scripts = ["runner"]
)
