import os
from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("relaxed_lasso", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "relaxed-lasso-continental"
DESCRIPTION = "Relaxed lasso regularization for linear regression."
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()
MAINTAINER = "Grégory Vial, Flora Estermann"
MAINTAINER_EMAIL = "gregory.vial@continental.com, flora.estermann@continental.com"
URL = ""
LICENSE = "new BSD"
DOWNLOAD_URL = ""
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "sklearn", "joblib"]
CLASSIFIERS = ["Intended Audience :: Science/Research",
               "Intended Audience :: Developers/Data Scientists",
               "Topic :: Software Development",
               "Topic :: Scientific/Engineering",
               "License :: OSI Approved",
               "Programming Language :: Python :: 3",
               "Operating System :: OS Independent"]
EXTRAS_REQUIRE = {
    "tests": [
        "pytest",
        "coverage"
    ],
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "numpydoc"
        "matplotlib"
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires=">=3.6")
