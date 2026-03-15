import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def print_engine_bindings(engine):
    print("Engine bindings:")
    for i in range(engine.num_bindings):
        print("{}: {}".format(i, engine.get_tensor_name(i)))



def load_trt_engine(trt_engine_path):
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# Load TensorRT engine
engine = load_trt_engine('checkpoints\wav2lip_b64.trt')

# Print bindings
print_engine_bindings(engine)
