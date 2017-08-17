from distutils.core import setup

files = ["view_all.py","clickable_image_new.py"]

setup(name = "view_all",
    version = "0.1a",
    description = "yadda yadda",
    author = "Jonathan Reich",
    author_email = "jreich@student.unimelb.edu.au",
    url = "https://github.com/JohnnyTeutonic/view_all"
    #Name the folder where your packages live:
    #(If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found 
    #recursively.)
    packages = ['package'],
    #'package' package must contain files (see list above)
    #I called the package 'package' thus cleverly confusing the whole issue...
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    package_data = {'package' : files },
    #'runner' is in the root.
    scripts = ["runner"],
    long_description = """Really long text here.""" 
    #
    #This next part it for the Cheese Shop, look a little down the page.
    #classifiers = []
)
