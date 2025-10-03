import torch as tch

def cudamaxxing():
    return tch.device('cuda:0' if tch.cuda.is_available() else 'cpu')

def dtypemaxxing(dev: tch.device, dtype: str = 'auto'):
    assert dtype in ['auto','fp16','bf16','fp32']
    match dtype:
        case 'auto':
            if dev.type == 'cuda': dtypetch = tch.bfloat16 if tch.cuda.is_bf16_supported() else tch.float16
            else: dtypetch = tch.float32
        case 'fp16': dtypetch = tch.float16
        case 'bf16': dtypetch = tch.bfloat16
        case 'fp32': dtypetch = tch.float32
    return dtypetch

def quantmaxxing(bit8: bool = False, bit4: bool = False):
    kwargs = {}
    kwargs.update({'device_map': 'auto'}) if bit8 or bit4 else None
    kwargs.update({'load_in_8bit': bit8})
    kwargs.update({'load_in_4bit': bit4})
    return kwargs

def torchesque(dtype: str = 'auto', bit8: bool = False, bit4: bool = False):
    dev = cudamaxxing()
    return {
        'device': dev,
        'dtype': dtypemaxxing(dev, dtype = dtype),
        'quantargs': quantmaxxing(bit8 = bit8, bit4 = bit4)
    }