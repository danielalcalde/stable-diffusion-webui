import datetime
import glob
import html
import os
import sys
import traceback
import tqdm
import csv

import torch

from ldm.util import default
from modules import devices, shared, processing, sd_models
import torch
from torch import _weight_norm, einsum
from einops import rearrange, repeat
import modules.textual_inversion.dataset
from modules.textual_inversion import textual_inversion
from modules.textual_inversion.learn_schedule import LearnRateScheduler
import math


class HypernetworkDoubleLinearModule(torch.nn.Module):
    multiplier = 1.0

    def __init__(self, dim, init_strength=0.1, n=4, first_bias=False, state_dict=None):
        super().__init__()
        self.dim = dim
        self.weight_norm = False
        self.first_bias = first_bias

        if state_dict is not None:
            shape = state_dict['linear1.weight'].shape
            first_bias = 'linear1.bias' in state_dict

            self.linear1 = torch.nn.Linear(shape[1], shape[0], bias=first_bias)
            self.linear2 = torch.nn.Linear(shape[0], shape[1])

            self.load_state_dict(state_dict, strict=True)
        else:
            self.linear1 = torch.nn.Linear(dim, dim//n, bias=first_bias)
            self.linear2 = torch.nn.Linear(dim//n, dim)

            std = math.sqrt(init_strength) * math.sqrt(2/(dim+dim/n))
        
            self.linear1.weight.data.normal_(mean=0.0, std=std)
            self.linear2.weight.data.normal_(mean=0.0, std=std)

            if first_bias:
                self.linear1.bias.data.zero_()
            
            self.linear2.bias.data.zero_()

        self.to(devices.device)

    def forward(self, x):
        return x + self.linear2(self.linear1(x)) * self.multiplier
    
    def weights(self):
        if self.weight_norm:
            weights = [self.linear1.weight_v, self.linear1.weight_g, self.linear2.weight_v, self.linear2.weight_g, self.linear2.bias]
        else:
            weights = [self.linear1.weight, self.linear2.weight, self.linear2.bias]
        
        if self.first_bias:
            weights += [self.linear1.bias]
        
        return weights
        
    
    def param_norm(self):
        weight = self.linear2.weight @ self.linear1.weight
        return torch.std(weight) * math.sqrt(self.dim)
    
    def apply_weight_norm(self):
        if not self.weight_norm:
            self.linear1 = torch.nn.utils.weight_norm(self.linear1, name='weight')
            self.linear2 = torch.nn.utils.weight_norm(self.linear2, name='weight')
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            self.linear1 = torch.nn.utils.remove_weight_norm(self.linear1, name='weight')
            self.linear2 = torch.nn.utils.remove_weight_norm(self.linear2, name='weight')
            self.weight_norm = False
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.weight_norm:
            state_dict['linear1.weight'] = self.linear1.weight
            state_dict['linear2.weight'] = self.linear2.weight

            del state_dict['linear1.weight_v']
            del state_dict['linear1.weight_g']
            del state_dict['linear2.weight_v']
            del state_dict['linear2.weight_g']
        
        return state_dict


class HypernetworkDoubleLinearModuleBias(HypernetworkDoubleLinearModule):
    """
    Model for backcompability with old hypernetworks
    """
    def __init__(self, dim, init_strength=0.1, n=4, state_dict=None):
        super().__init__(dim, init_strength=init_strength, n=n, first_bias=True, state_dict=state_dict)
    

class HypernetworkLinearModule(torch.nn.Module):
    multiplier = 1.0

    def __init__(self, dim, init_strength=0.1, state_dict=None):
        super().__init__()
        self.dim = dim
        self.weight_norm = False

        self.linear = torch.nn.Linear(dim, dim)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        else:
            self.linear.weight.data.normal_(mean=0.0, std=init_strength/math.sqrt(dim))
            self.linear.bias.data.zero_()

        self.to(devices.device)

    def forward(self, x):
        return x + self.linear(x) * self.multiplier
    
    def weights(self):
        if self.weight_norm:
            weights = [self.linear.weight_v, self.linear.weight_g, self.linear.bias]
        else:
            weights = [self.linear.weight, self.linear.bias]
        
        return weights
    
    def param_norm(self):
        if self.weight_norm:
            return torch.std(self.linear.weight_g) * math.sqrt(self.dim)
        else:
            return torch.std(self.linear.weight) * math.sqrt(self.dim)
    
    def apply_weight_norm(self):
        if not self.weight_norm:
            self.linear = torch.nn.utils.weight_norm(self.linear, name='weight')
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            self.linear = torch.nn.utils.remove_weight_norm(self.linear, name='weight')
            self.weight_norm = False
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.weight_norm:
            state_dict['linear1.weight'] = self.linear.weight

            del state_dict['linear.weight_v']
            del state_dict['linear.weight_g']
        
        return state_dict


def apply_strength(value=None):
    HypernetworkLinearModule.multiplier = value if value is not None else shared.opts.sd_hypernetwork_strength
    HypernetworkDoubleLinearModule.multiplier = value if value is not None else shared.opts.sd_hypernetwork_strength
    HypernetworkDoubleLinearModuleBias.multiplier = value if value is not None else shared.opts.sd_hypernetwork_strength


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, weight_norm=False, name=None, enable_sizes=None, HypernetworkModule=HypernetworkLinearModule, HypernetworkModule_init=None):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.HypernetworkModule = HypernetworkModule
        if HypernetworkModule_init is None:
            HypernetworkModule_init = HypernetworkModule

        self.weight_norm = weight_norm

        for size in enable_sizes or []:
            self.layers[size] = (HypernetworkModule_init(size), HypernetworkModule_init(size))

    def weights(self):
        res = []

        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                res += layer.weights()

        return res

    def save(self, filename):
        state_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['HypernetworkType'] = self.HypernetworkModule.__name__
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name

        torch.save(state_dict, filename)
    
    def param_norm(self):
        res = 0
        n = 0
        for k, v in self.layers.items():
            for layer in v:
                res += layer.param_norm()
                n += 1
        
        return res / n

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        if 'HypernetworkType' in state_dict:
            print(f"Loading {state_dict['HypernetworkType']} from {filename}")

            if state_dict['HypernetworkType'] == 'HypernetworkLinearModule':
                self.HypernetworkModule = HypernetworkLinearModule

            elif state_dict['HypernetworkType'] == 'HypernetworkDoubleLinearModule':
                self.HypernetworkModule = HypernetworkDoubleLinearModule
        else:
            print("Warning: HypernetworkType not found in state_dict, assuming Old HypernetworkDoubleLinearModuleBias")
            self.HypernetworkModule = HypernetworkDoubleLinearModuleBias

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (self.HypernetworkModule(size, state_dict=sd[0]), self.HypernetworkModule(size, state_dict=sd[1]))
                if self.weight_norm:
                    self.layers[size][0].apply_weight_norm()
                    self.layers[size][1].apply_weight_norm()

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)


def list_hypernetworks(path):
    res = {}
    for filename in glob.iglob(os.path.join(path, '**/*.pt'), recursive=True):
        name = os.path.splitext(os.path.basename(filename))[0]
        res[name] = filename
    return res


def load_hypernetwork(filename):
    path = shared.hypernetworks.get(filename, None)
    if path is not None:
        print(f"Loading hypernetwork {filename}")
        try:
            shared.loaded_hypernetwork = Hypernetwork()
            shared.loaded_hypernetwork.load(path)

        except Exception:
            print(f"Error loading hypernetwork {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        if shared.loaded_hypernetwork is not None:
            print(f"Unloading hypernetwork")

        shared.loaded_hypernetwork = None


def find_closest_hypernetwork_name(search: str):
    if not search:
        return None
    search = search.lower()
    applicable = [name for name in shared.hypernetworks if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return applicable[0]


def apply_hypernetwork(hypernetwork, context, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is None:
        return context, context

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context)
    context_v = hypernetwork_layers[1](context)
    return context_k, context_v


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetwork(shared.loaded_hypernetwork, context, self)
    k = self.to_k(context_k)
    v = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # same as in reconstruct_multicond_batch
    token_count = max([x.shape[0] for x in conds])
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)

def train_hypernetwork(hypernetwork_name, learn_rate, batch_size, data_root, log_directory, steps, create_image_every, save_hypernetwork_every, template_file, preview_from_txt2img, weight_norm, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    assert hypernetwork_name, 'hypernetwork not selected'

    path = shared.hypernetworks.get(hypernetwork_name, None)
    shared.loaded_hypernetwork = Hypernetwork(weight_norm=weight_norm)
    shared.loaded_hypernetwork.load(path)

    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    unload = shared.opts.unload_models_when_training

    if save_hypernetwork_every > 0:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=512, height=512, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=hypernetwork_name, model=shared.sd_model, device=devices.device, template_file=template_file, include_cond=True, batch_size=batch_size)

    if unload:
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    hypernetwork = shared.loaded_hypernetwork
    weights = hypernetwork.weights()
    for weight in weights:
        weight.requires_grad = True

    losses = torch.zeros((64,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"

    ititial_step = hypernetwork.step or 0
    if ititial_step > steps:
        return hypernetwork, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    optimizer = torch.optim.AdamW(weights, lr=scheduler.learn_rate)

    pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
    for i, entries in pbar:
        hypernetwork.step = i + ititial_step

        scheduler.apply(optimizer, hypernetwork.step)
        if scheduler.finished:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = stack_conds([entry.cond for entry in entries]).to(devices.device)
#            c = torch.vstack([entry.cond for entry in entries]).to(devices.device)
            x = torch.stack([entry.latent for entry in entries]).to(devices.device)
            loss = shared.sd_model(x, c)[0]
            del x
            del c

            losses[hypernetwork.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        mean_loss = losses.mean()
        param_norm = hypernetwork.param_norm()

        if torch.isnan(mean_loss):
            raise RuntimeError("Loss diverged.")
        pbar.set_description(f"loss: {mean_loss:.7f}, param norm: {param_norm:.7f}")

        if hypernetwork.step > 0 and hypernetwork_dir is not None and hypernetwork.step % save_hypernetwork_every == 0:
            last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name}-{hypernetwork.step}.pt')
            hypernetwork.save(last_saved_file)

        textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, len(ds), {
            "loss": f"{mean_loss:.7f}",
            "param_norm": f"{param_norm:.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if hypernetwork.step > 0 and images_dir is not None and hypernetwork.step % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{hypernetwork_name}-{hypernetwork.step}.png')

            optimizer.zero_grad()
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0] if len(processed.images)>0 else None

            if unload:
                shared.sd_model.cond_stage_model.to(devices.cpu)
                shared.sd_model.first_stage_model.to(devices.cpu)

            if image is not None:
                shared.state.current_image = image
                image.save(last_saved_image)
                last_saved_image += f", prompt: {preview_text}"

        shared.state.job_no = hypernetwork.step

        shared.state.textinfo = f"""
<p>
Loss: {mean_loss:.7f}<br/>
Param norm: {param_norm:.7f}<br/>
Step: {hypernetwork.step}<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    hypernetwork.sd_checkpoint = checkpoint.hash
    hypernetwork.sd_checkpoint_name = checkpoint.model_name
    hypernetwork.save(filename)

    return hypernetwork, filename


