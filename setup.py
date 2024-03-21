import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "pyldl",
    version = "0.0.1",
    author = "Paul-Tiberiu IORDACHE and Ethan LUONG",
    description = "Lightweight Deep Learning Library for educational purposes",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "package URL",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.10.0"
)
