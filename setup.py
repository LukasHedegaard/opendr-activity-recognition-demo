from setuptools import find_packages, setup


def from_file(file_name: str = "requirements.txt", comment_char: str = "#"):
    """Load requirements from a file"""
    with open(file_name, "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def long_description():
    text = open("README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


setup(
    name="opendr-stream-video-browser",
    version="0.1.0",
    description="Example of in-browser video streaming and processing using the OpenDR toolkit.",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    author="Lukas Hedegaard",
    author_email="lh@ece.au.dk",
    url="https://github.com/LukasHedegaard/opendr-stream-video-browser",
    install_requires=from_file("requirements.txt"),
    packages=find_packages(exclude=["test"]),
    keywords=["deep learning", "pytorch", "AI", "OpenDR", "video", "webcam"],
)
