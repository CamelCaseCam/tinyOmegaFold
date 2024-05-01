import tinygrad.device
import torch
import tinygrad
import tinygrad.nn
import numpy as np

# Only on Cameron's machine because CUDA is bricked
import os
DEBUG = False
if os.environ.get('USER') == 'cameronk':
    tinygrad.Device.DEFAULT = "GPU" # No CUDA!
    DEBUG = True


dt2tg = { torch.float32: tinygrad.dtypes.float32, torch.float64: tinygrad.dtypes.float64 }
dv2tg = { torch.device('cpu'): "clang", torch.device('cuda', index=0): "gpu" }
dv2trch = { "clang": torch.device('cpu'), "gpu": torch.device('cuda', index=0), "CLANG" : torch.device('cpu'), "GPU": torch.device('cuda', index=0) }
def to_tinygrad(x: torch.Tensor, checknan=False) -> tinygrad.Tensor:
    if x is None:
        return None
    if checknan:
        if isinstance(x, torch.Tensor):
            xnp = x.cpu().numpy()
        else:
            xnp = x.numpy()
        if np.isnan(xnp).any():
            if isinstance(x, torch.Tensor):
                return to_tinygrad(torch.nan_to_num(x))
            return to_tinygrad(to_torch(x))
        return tinygrad.Tensor(xnp, device=dv2tg[x.device]) if isinstance(x, torch.Tensor) else x
    return tinygrad.Tensor(x.cpu().numpy(), device=dv2tg[x.device]) if isinstance(x, torch.Tensor) else x

def extract_and_print_state_dict(model_path):
    """
    Loads a PyTorch model from the given path, extracts its state dict,
    and prints the keys.

    Args:
        model_path (str): The path to the saved PyTorch model file.
    """

    # Load the model (ensure you have PyTorch installed)
    model = torch.load(model_path, map_location='cpu') # For OmegaFold class

    if isinstance(model, dict):
        # If the model is a dictionary, torch.load has already returned the state dict
        state_dict = model
    else:
        # If the model is not a dictionary, extract the state dict
        state_dict = model.state_dict()

    # Print the keys of the state dict
    print("Model State Dict Keys:")
    for key in state_dict.keys():
        print(key)

class Module:
    def _param_from_var(self, prefix, var, outdict):
        """
        Tries to add the variable to the outdict. 
        """
        if isinstance(var, torch.nn.Module):
            # Recursively get parameters from PyTorch nn.Module
            module_params = var.named_parameters()
            for module_param_name, module_param in module_params:
                full_param_name = f"{prefix}{module_param_name}"
                outdict[full_param_name] = module_param

        elif isinstance(var, torch.nn.Parameter):
                # Include torch.nn.Parameter directly
                outdict[prefix[:-1]] = var  

        elif isinstance(var, Module):
            # Call parameters() on custom Module objects
            module_params = var.parameters(prefix=prefix[:-1])    # Remove the trailing dot
            outdict.update(module_params)

        elif isinstance(var, list):
            # Process parameters within lists
            for i, item in enumerate(var):
                item_prefix = f"{prefix}{i}."
                self._param_from_var(item_prefix, item, outdict)

        elif isinstance(var, tinygrad.Tensor):
            # Add Tinygrad Tensors directly
            outdict[prefix[:-1]] = var

        elif isinstance(var, tinygrad.nn.Linear):
            outdict[f"{prefix[:-1]}.weight"] = var.weight
            outdict[f"{prefix[:-1]}.bias"] = var.bias

    def parameters(self, prefix=""):
        """
        Returns a dictionary of parameter IDs and values within the Module and its submodules.

        Args:
            prefix (str, optional): A prefix to add to each parameter ID. Defaults to "".

        Returns:
            dict: A dictionary with keys: "PREFIX.ID" and values as corresponding parameter values.
        """

        params_dict = {}
        for name, obj in self.__dict__.items():
            # Debug (if ends in _DEBUGk where k is a single character)
            name = name[:-7] if name[-7:-1] == "_DEBUG" else name
            newprefix = f"{prefix}.{name}." if prefix else f"{name}."
            self._param_from_var(newprefix, obj, params_dict)

        return params_dict

    def load_state_dict(self, state_dict):
        """
        Loads weights from a state dict into the corresponding parameters of the Module.

        Args:
            state_dict (dict): A dictionary containing parameter names and their values.
        """

        my_params = self.parameters()

        for param_id in my_params:
            if param_id not in state_dict:
                raise ValueError(f"Parameter '{param_id}' not found in the state dict.")

            param = my_params[param_id]
            sd_val = state_dict[param_id]

            if isinstance(param, torch.Tensor) or isinstance(param, torch.nn.Parameter):
                if param.shape != sd_val.shape:
                    raise ValueError(f"Shape mismatch for '{param_id}'. Expected {param.shape}, found {sd_val.shape}")

                # Load PyTorch tensor data
                # This would be dangerous if we weren't about to overwrite the values
                param._device = sd_val.device
                param._dtype = sd_val.dtype
                param.data = sd_val
            
            elif isinstance(param, tinygrad.Tensor):
                if param.shape != sd_val.shape:
                    raise ValueError(f"Shape mismatch for '{param_id}'. Expected {param.shape}, found {sd_val.shape}")

                # Load Tinygrad tensor data
                np_val = sd_val.cpu().numpy() if isinstance(sd_val, torch.Tensor) else sd_val
                # This would be dangerous if we weren't about to overwrite the values
                param.lazydata.device = dv2tg[sd_val.device]
                param.lazydata.dtype = dt2tg[sd_val.dtype]
                param.__init__(np_val, device=dv2tg[sd_val.device], dtype=dt2tg[sd_val.dtype])
            else:
                raise TypeError(f"Unsupported parameter type: {type(param)}")
    def eval(self):
        """
        Sets the Module and its submodules containing PyTorch components to evaluation mode.
        """
        for _, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                obj.eval() 
            elif isinstance(obj, Module):
                obj.eval()  # Recursively call eval on custom Module objects
    def to(self, device):
        """
        Moves all parameters of the `Module` and its submodules to the specified device. 
        This method returns the modified `Module` object to support chaining, but it operates in-place.

        Args:
            device (str or torch.device): The desired device. 
                Can be specified in Tinygrad format ('clang', 'cuda') or as a PyTorch device object.
        """

        def convert_device(param, device):
            if isinstance(param, torch.Tensor) or isinstance(param, torch.nn.Parameter):
                # Make sure the device is a PyTorch device object
                trch_device = device if isinstance(device, torch.device) else dv2trch[device]
                # Dangerous but necessary hack to change the device in-place
                dev_var = param.to(trch_device)
                param._device = dev_var.device
                param.data = dev_var.data
            elif isinstance(param, tinygrad.Tensor):
                # Look up the Tinygrad equivalent device and use .to()
                tg_device = device if isinstance(device, str) else dv2tg[device]
                # Dangerous but necessary hack to change the device in-place
                dev_var = param.to(tg_device)
                # TODO: This doesn't work - some buffers are not being moved to the correct device
                param.lazydata.device = dev_var.device
                dev_varn = dev_var.numpy()
                param.__init__(dev_varn, device=dev_var.device, dtype=dev_var.dtype)
            else:
                raise TypeError(f"Unsupported parameter type: {type(param)}")

        # Get parameters
        for param in self.alltensors():
            convert_device(param, device)
        return self
    
    def __call__(self, *args, **kwargs):
        """
        Performs a forward pass through the Module, mimicking nn.Module behavior.
        """
        # Forward pass through the module
        output = self.forward(*args, **kwargs)
        return output
    
    def getdevice(self):
        """
        Returns the torch device corresponding to the device from any PyTorch submodule.

        Returns:
            torch.device: The torch device or None if no PyTorch submodules are found.
        """
        for _, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                return next(obj.parameters()).device  # Return the device of the first parameter (if any)
            elif isinstance(obj, Module):
                device = obj.getdevice()  # Check submodules recursively
                if device is not None:
                    return device
        return None  # If no PyTorch components are found
    
    def getdtype(self):
        """
        Returns the torch data type (dtype) corresponding to the data type from any PyTorch submodule.

        Returns:
            torch.dtype: The torch data type or None if no PyTorch submodules are found.
        """
        for _, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                return next(obj.parameters()).dtype  # Return the dtype of the first parameter (if any)
            elif isinstance(obj, Module):
                dtype = obj.getdtype()  # Check submodules recursively
                if dtype is not None:
                    return dtype
        return None  # If no PyTorch components are found
    

    def _alltensors_var(self, var):    # Generator function to yield all tensors in a variable
        if isinstance(var, torch.Tensor) or isinstance(var, torch.nn.Parameter):
            yield var
        elif isinstance(var, Module):
            for tensor in var.alltensors():
                yield tensor
        elif isinstance(var, list):
            for item in var:
                for tensor in self._alltensors_var(item):
                    if tensor is not None:
                        yield tensor
        elif isinstance(var, tinygrad.Tensor):
            yield var
        elif isinstance(var, torch.nn.Module):
            # Loop through all attributes of the module to make sure we get all tensors
            for _, obj in var.__dict__.items():
                for tensor in self._alltensors_var(obj):
                    if tensor is not None:
                        yield tensor
            for _, obj in var._parameters.items():
                for tensor in self._alltensors_var(obj):
                    if tensor is not None:
                        yield tensor
            for _, obj in var._buffers.items():
                for tensor in self._alltensors_var(obj):
                    if tensor is not None:
                        yield tensor
        # Necessary hack
        elif isinstance(var, tinygrad.nn.Linear):
            yield var.weight
            yield var.bias

    
    def alltensors(self):
        """
        Returns a generator that yields all tensors within the Module and its submodules.

        Returns:
            Generator: A generator that yields tensors.
        """
        for _, obj in self.__dict__.items():
            for tensor in self._alltensors_var(obj):
                if tensor is not None:
                    yield tensor
    
            

class ModuleList(Module):
    def __init__(self, modules):
        self.modules = modules  # Store the provided list of Module objects

    def __iter__(self):
        return iter(self.modules)  # Allow iteration over modules

    def __getitem__(self, index):
        return self.modules[index]  # Enable indexing for module access

    def parameters(self, prefix=""):
        """
        Collect parameters from all modules within the ModuleList.

        Args:
            prefix (str, optional): A prefix to add to each parameter ID. Defaults to "".

        Returns:
            dict: A dictionary containing parameter names (with prefix) and associated values.
        """
        # Add . to the prefix if it is not empty
        prefix = f"{prefix}." if prefix else ""
        
        params_dict = {}
        for i, module in enumerate(self.modules):
            module_prefix = f"{prefix}{i}" if prefix else str(i)
            params_dict.update(module.parameters(prefix=module_prefix))
        return params_dict
    
# Sequential Module
class Sequential(Module):
    def __init__(self, *args):
        self.modules = list(args)  # Convert the input tuple to a list
    
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    def parameters(self, prefix=""):
        # Override so that it has the same behaviour as torch.nn.Sequential
        prefix = f"{prefix}." if prefix else ""
        params_dict = {}
        for i, module in enumerate(self.modules):
            module_prefix = f"{prefix}{i}." if prefix else str(i) + "."
            self._param_from_var(module_prefix, module, params_dict)
        return params_dict
    
    def alltensors(self):
        for module in self.modules:
            for tensor in self._alltensors_var(module):
                if tensor is not None:
                    yield tensor
    
if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU()
    )
    model.eval()
    print(model.state_dict().keys())

    model = Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU()
    )

    model.eval()
    print(model.parameters().keys())