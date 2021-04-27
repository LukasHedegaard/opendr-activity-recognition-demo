# OpenDR stream video browser
<div align="left">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

Example of in-browser video streaming and processing using the [OpenDR toolkit](https://opendr.eu).




## Installation
```bash
git clone https://github.com/LukasHedegaard/opendr-stream-video-browser
cd opendr-stream-video-browser
pip install -e .
```


## Running the example
```bash
python webstreaming.py --ip 0.0.0.0 --port 8000
```
Naturally, other values of `ip` and `port` can be selected.


## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

The code for webcam streaming is based on [this PyImageSearch tutorial](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/).
