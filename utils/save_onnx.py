import torch

def export_to_onnx(model, dataset, model_path_stem):

    """Exports the PyTorch model to ONNX format."""
    n_input = dataset.n_input_params
    onnx_path = model_path_stem + ".onnx"
    
    # Create a dummy input tensor matching the expected size and device
    dummy_input_np = dataset.full_data[:1, :n_input].astype('float32')
    # Move dummy input to the model's device
    dummy_input = torch.from_numpy(dummy_input_np).to(model.parameters().__next__().device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    return onnx_path