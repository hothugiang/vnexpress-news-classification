from typing import List, Union, Optional, Set, Tuple
from math import ceil
from transformers import Conv1D

import torch
import wandb
import os

PROJECT_NAME = "Demonstration-enhanced CRS"
RECOMMENDATION = "recommendation"
GENERATION = "generation"
MODEL_NAME = "UNICRS"

MODEL_RELATED_PARAMS = [
    "n_examples",
    "mapping",
    "prompt_max_length",
    "learning_rate",
    "seed",
    "bias_only",
    "learning_rate",
]


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length :] = item

    return output


def convert_params_to_str(params):
    param_str = ""
    for key, value in params.items():
        s = ""
        if key in MODEL_RELATED_PARAMS:
            s = f"[{key}={value}]"
        param_str += s
    return param_str


def init_wandb_run(
    project_name,
    dataset,
    task,
    tags,
    model_name,
    model_params,
    type_of_run="full",
    run_name=None,
):

    ### project_name:
    ### task: recommendation or generation
    ### model_name: DCRS or BASElines
    ### model_params: parameters
    ### type of run: full, ablation, analysis.
    if run_name is None:
        run_name = convert_params_to_str(model_params)

    run = wandb.init(
        project=f"{project_name}",
        group=f"{dataset}-{task}/",
        job_type=type_of_run,
        tags=tags,
        entity="HuyQuangDao",
        reinit=True,
        name=f"{model_name}-{run_name}",
    )


def wandb_logging(eval_dict, step):
    for key, value in eval_dict.items():
        wandb.log(data={key: value}, step=step)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model_params(gen_model, text_encoder, bias_only=True):
    fix_modules = [text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    if bias_only:
        #### freeze parameters of the pretrained language model
        for param in gen_model.parameters():
            param.requires_grad = False
        ### only train bias parameters.
        # trainable_components = ['bias']
        # trainable_components = trainable_components + ['pooler.dense.bias']
        # ## unfreeze trainable parameters.
        for para in gen_model.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)

        for para in text_encoder.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)


def save(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {k: v for k, v in model.state_dict().items() if "edge" not in k}
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(state_dict, save_path)


def load(model, load_dir):
    load_path = os.path.join(load_dir, "model.pt")
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(load_path, map_location=torch.device("cpu")), strict=False
    )
    return model


def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [
        item for sublist in list(device_map.values()) for item in sublist
    ]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device."
            " These attention blocks were specified more than once: "
            + str(duplicate_blocks)
        )
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]

    return dict(zip(devices, layers_list))


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer
