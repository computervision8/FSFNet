import os
import math
import numpy as np
import torch
from torch.autograd import Variable
import time
from tqdm import tqdm
import matplotlib
import warnings


try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    MAX_BATCH_SIZE = 1
    MAX_WORKSPACE_SIZE = 1 << 30

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    DTYPE = trt.float32

    # Model
    INPUT_NAME = 'input'
    OUTPUT_NAME = 'output'
    print("run tensorrt")
    def allocate_buffers(engine):
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
        h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        return h_input, d_input, h_output, d_output


    def build_engine(model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = MAX_WORKSPACE_SIZE
            builder.max_batch_size = MAX_BATCH_SIZE

            with open(model_file, 'rb') as model:
                parser.parse(model.read())
                return builder.build_cuda_engine(network)


    def load_input(input_size, host_buffer):
        assert len(input_size) == 4
        b, c, h, w = input_size
        dtype = trt.nptype(DTYPE)
        img_array = np.random.randn(c, h, w).astype(dtype).ravel()
        np.copyto(host_buffer, img_array)


    def do_inference(context, h_input, d_input, h_output, d_output, iterations=None):
        # Transfer input data to the GPU.
        cuda.memcpy_htod(d_input, h_input)
        # warm-up
        for _ in range(10):
            context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        # test proper iterations
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                t_start = time.time()
                for _ in range(iterations):
                    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 3)
        # Run inference.
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
        return latency


    def compute_latency_ms_tensorrt(model, input_size, iterations=None):
        model = model.cuda()
        model.eval()
        _, c, h, w = input_size
        dummy_input = torch.randn(1, c, h, w, device='cuda')
        torch.onnx.export(model, dummy_input, "model.onnx", verbose=False, input_names=["input"], output_names=["output"])
        with build_engine("model.onnx") as engine:
            h_input, d_input, h_output, d_output = allocate_buffers(engine)
            load_input(input_size, h_input)
            with engine.create_execution_context() as context:
                latency = do_inference(context, h_input, d_input, h_output, d_output, iterations=iterations)
        # FPS = 1000 / latency (in ms)
        return latency
except:
    warnings.warn("TensorRT (or pycuda) is not installed. compute_latency_ms_tensorrt() cannot be used.")

    def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
        print("run pytorch ")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        model.eval()
        model = model.cuda()

        input = torch.randn(*input_size).cuda()

        with torch.no_grad():
            for _ in range(10):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            print('=========Speed Testing=========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in tqdm(range(iterations)):
                model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        # FPS = 1000 / latency (in ms)
        return latency


