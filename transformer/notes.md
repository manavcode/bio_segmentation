# Notes on Training Transformers

A place to store experiment / testing info

## Matt's testing

Config 1: 4 transformer blocks, 4 up/downscaling blocks
- 23.7GB VRAM
- didn't work after 5 epochs...

Config 2: 2 transformer blocks, 4 up/down
- weird JIT compilation exceptions on predict()?