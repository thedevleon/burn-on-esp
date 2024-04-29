# Burn on ESP

Running [burn](https://burn.dev/) on an ESP32-C6, potentially for keyword spotting or other ML tasks.

A simple MNIST dummy network (downsized to fit in the flash) can already run with around 149ms inference time, which is quite good for a microcontroller without floating point support.

Maybe burn will add some quantization options (i.e. int8 / int32) in the future, which should greatly improve the inference time.