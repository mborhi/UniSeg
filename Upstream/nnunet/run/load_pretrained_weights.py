#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch

def load_pretrained_weights_lenient(network, fname):
    saved_model = torch.load(fname)
    original_state_dict = saved_model['state_dict']
    adapted_state_dict = network.state_dict()

    exclude_layers = ['seg_outputs']

    for name, param in original_state_dict.items():
        if any(excluded in name for excluded in exclude_layers):
            print(f"Skipping loading weights for '{name}'")
            continue 
        if name in adapted_state_dict:
            if param.size() == adapted_state_dict[name].size():
                # exact match, load directly
                adapted_state_dict[name].copy_(param)
            else:
                # mismatch in shape, handle selectively
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(param.size(), adapted_state_dict[name].size()))
                slices = tuple(slice(0, ms) for ms in min_shape)

                print(f"Loading partial weights for '{name}': {param.size()} -> {adapted_state_dict[name].size()}, slices: {slices}")
                
                # Copy matching subset of weights
                adapted_state_dict[name][slices].copy_(param[slices])

    # Load updated state dict into the adapted model
    network.load_state_dict(adapted_state_dict)


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")

