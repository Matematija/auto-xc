from setuptools import setup, find_packages

BASE_DEPENDENCIES = ["numpy", "jax", "pyscf<=2.3.0"]

setup(
    name="autoxc",
    author="Matija Medvidovic",
    url="https://github.com/Matematija/auto-xc",
    author_email="matija.medvidovic@gmail.com",
    description="Automatic differentiation tools for interfacing custom density functionals with libXC in quantum chemistry.",
    packages=find_packages(),
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
