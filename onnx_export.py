import torch
import onnx
from models import Wav2Lip
import onnxruntime as ort
import numpy as np

# Load your trained PyTorch model
torch_model = Wav2Lip()
# weight = torch.load("./checkpoints/wav2lip.pth")

batch_size = 64

# print(torch.load("./checkpoints/wav2lip.pth")['state_dict'])


# Initialize model with the pretrained weights

torch_model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./checkpoints/wav2lip.pth')['state_dict'].items()})

# torch_model.eval()
torch_model.eval()

# Input to the model
x = torch.randn(batch_size, 1,80, 16, requires_grad=True)
# x = torch.randn(batch_size, 1, 80, 16, requires_grad=True)

# y = torch.randn(batch_size, 6, 96, 96, requires_grad=True)
y = torch.randn(batch_size, 6, 96, 96, requires_grad=True)

torch_out = torch_model(x,y)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (x,y),                         # model input (or a tuple for multiple inputs)
                  "checkpoints/wav2lip_batch16.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                # 'input2' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

print("导出.onnx模型成功!")


# 验证模型.pth导出和.onnx是否一致

onnx_model = onnx.load("checkpoints/wav2lip_batch64.onnx")
onnx.checker.check_model(onnx_model)
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")

ort_session = ort.InferenceSession("checkpoints/wav2lip_batch64.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x),ort_session.get_inputs()[1].name: to_numpy(y)}
ort_outs = ort_session.run(None, ort_inputs)


# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")