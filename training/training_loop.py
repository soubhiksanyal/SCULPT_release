# Modified from StyleGAN3 codebase

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import ipdb

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = 5 # np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = 4 # np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    # images, labels = zip(*[training_set[i] for i in grid_indices])
    images, labels, b_s, b_p, b_ca = zip(*[training_set[i] for i in grid_indices])
    # ipdb.set_trace()
    return (gw, gh), np.stack(images), np.stack(labels), np.stack(b_s), np.stack(b_p)

#----------------------------------------------------------------------------

def save_image_grid_withoutalphachannel(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    # img = img.transpose(0, 2, 3, 1)
    # ipdb.set_trace()
    img[:, :3, :, :] = (img[:, :3, :, :] - lo) * (255 / (hi - lo)) * (img[:, 3:, :, :]/hi)
    img[:, 3:, :, :] = img[:, 3:, :, :] * 255 / hi
    # img[:, :3, :, :] = (img[:, :3, :, :] - lo) * (255 / (hi - lo)) #* (img[:, 3:, :, :]/hi)
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    img = img.astype(np.uint8)
    # img = img.transpose(0, 3, 1, 2)
    # ipdb.set_trace()
    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3, 4]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
    if C == 4:
        # PIL.Image.fromarray(img, 'RGBA').save(fname)
        PIL.Image.fromarray(img[:, :, :3], 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    geometry_pkl            = None,     # Network pickle to load the geometry network.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    # Load training set.
    print(device)
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    pose_cond_dim = 0
    if training_set._pose_cond and training_set.pose_cond_type=='axisangle':
        pose_cond_dim = 63
    elif training_set._pose_cond and training_set.pose_cond_type=='6Drot':
        pose_cond_dim = 378
    elif training_set._pose_cond and training_set.pose_cond_type=='vposer':
        pose_cond_dim = 32
    else:
        pose_cond_dim = training_set.label_dim

    if geometry_pkl is not None:
        pose_cond_dim = 0

    if training_set.clothtype_cond:
        pose_cond_dim = pose_cond_dim + training_set.label_dim

    if (geometry_pkl is not None):
        with dnnlib.util.open_url(geometry_pkl) as f:
            G_geometry = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)
    # if (geometry_pkl is not None):        
    #     G_geometry.eval().requires_grad_(False).to(device)
    
    if training_set.conditional_D:
        common_kwargs = dict(c_dim=pose_cond_dim, img_resolution=training_set.resolution, img_channels=6)#training_set.num_channels)
    else:
        common_kwargs = dict(c_dim=pose_cond_dim, img_resolution=training_set.resolution, img_channels=3)#training_set.num_channels)
    # ipdb.set_trace()
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module

    common_kwargs_D_patch = dict(c_dim=pose_cond_dim, img_resolution=training_set.resolution, img_channels=3, patch_d_with_glob=False)#training_set.num_channels)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs_D_patch).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    common_kwargs_D_glob = dict(c_dim=pose_cond_dim, img_resolution=training_set.resolution, img_channels=3, patch_d_with_glob=True)#training_set.num_channels)
    D_glob = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs_D_glob).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    G_ema = copy.deepcopy(G).eval()
    # ipdb.set_trace()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        if G.resume_pretrain_cape:
            for name, module in [('G', G), ('G_ema', G_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
                # ipdb.set_trace()
        else:
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        del resume_data
    # ipdb.set_trace()
    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        # if geometry_pkl is not None:
        #     c = torch.empty([batch_gpu, 63], device=device)
        # else:
        #     c = torch.empty([batch_gpu, G.c_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        b_s = torch.empty([batch_gpu, 6890, 3], device=device)
        b_p = torch.empty([batch_gpu, 75], device=device) #torch.empty([batch_gpu, 72], device=device)
        b_ca = torch.empty([batch_gpu, 1], device=device) # torch.empty([batch_gpu, 3, 4], device=device) #  
        if geometry_pkl is None:
            img = misc.print_module_summary(G, [z, c, b_s, b_p, b_ca])
        else:
            img = misc.print_module_summary(G, [z, c, b_s, b_p, b_ca, G_geometry])
        # misc.print_module_summary(D, [img, c])
        # ipdb.set_trace()
        if D.img_channels == 6:
            if D.patch_d:
                misc.print_module_summary(D, [torch.cat((img[:, :3, 0:32, 0:32],img[:, :3, 0:64, 0:64]),1), c])
            else:
                misc.print_module_summary(D, [torch.cat((img[:, :3, :, :],img[:, :3, :, :]),1), c])
        else:
            if D.patch_d:
                misc.print_module_summary(D, [img[:, :3, 0:64, 0:64], c])
                misc.print_module_summary(D_glob, [img[:, :3, :, :], c])

                # misc.print_module_summary(D, [img[:, :, 0:64, 0:64], c])
                # misc.print_module_summary(D_glob, [img[:, :, :, :], c])
            else:
                misc.print_module_summary(D, [img[:, :3, :, :], c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, D_glob, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, D_glob=D_glob, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    # ipdb.set_trace()
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval), ('D_glob', D_glob , D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            # phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, bs, bp = setup_snapshot_image_grid(training_set=training_set)
        # ipdb.set_trace()
        new_imgs = torch.from_numpy(images).to(torch.float32).to(device)
        # new_imgs[:, :3, :, :] = (new_imgs[:, :3, :, :] / 127.5 - 1) #* (new_imgs[:, 3:, :, :]/255)
        # new_imgs[:, :3, :, :] = torch.nn.functional.normalize(new_imgs[:, :3, :, :], dim=1) #dim=-1)
        save_image_grid_withoutalphachannel(new_imgs[:, :3, :, :].cpu().numpy(), os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        if training_set.conditional_D:
            save_image_grid_withoutalphachannel(new_imgs[:, 3:, :, :].cpu().numpy(), os.path.join(run_dir, 'real_normals.png'), drange=[0, 255], grid_size=grid_size)
        # save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        # ipdb.set_trace()
        # # grid_z = (torch.ones_like(torch.randn([labels.shape[0], G.z_dim], device=device))*0.1).split(batch_gpu)
        # grid_z = torch.from_numpy(np.load('/is/cluster/work/ssanyal/project_4/data/stylegan3/CAPE/new/random_normal_sample_1.npy')).type(torch.float32).to(device).repeat(labels.shape[0], 1).split(batch_gpu)
        if training_set._pose_cond and training_set.pose_cond_type=='axisangle' and not training_set.clothtype_cond:
            grid_c = torch.from_numpy(bp[:, 3:66]).to(device).split(batch_gpu)
        elif training_set._pose_cond and training_set.pose_cond_type=='6Drot' and not training_set.clothtype_cond:
            pass
        elif training_set._pose_cond and training_set.pose_cond_type=='vposer' and not training_set.clothtype_cond:
            grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        elif training_set._pose_cond and training_set.pose_cond_type=='axisangle' and training_set.clothtype_cond:
            grid_c = torch.from_numpy(np.concatenate((labels, bp[:, 3:66]), 1)).to(device).split(batch_gpu)
        elif not training_set._pose_cond and training_set.pose_cond_type=='axisangle' and training_set.clothtype_cond:
            grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        else:
            grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_bp = torch.from_numpy(bp).to(device).split(batch_gpu)
        grid_bs = torch.from_numpy(bs).to(device).split(batch_gpu)
        # images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        # save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        # torch.autograd.set_detect_anomaly(True)
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c, phase_body_shape, phase_body_pose, phase_body_cam = next(training_set_iterator)
            # ipdb.set_trace()
            # print(phase_real_img.shape)
            phase_real_img = phase_real_img.to(device).to(torch.float32)
            phase_real_img[:, :3, :, :] = (phase_real_img[:, :3, :, :] / 127.5 - 1) #* (phase_real_img[:, 3:, :, :] / 255.)
            phase_real_img[:, 3:, :, :] = phase_real_img[:, 3:, :, :] / 255.

            phase_real_img_alpha = phase_real_img[:, 3:, :, :]
            phase_real_img_alpha[phase_real_img_alpha>0.5] = 1.
            phase_real_img_alpha[phase_real_img_alpha<=0.5] = 0.
            phase_real_img[:, 3:, :, :] = phase_real_img_alpha


            # phase_real_img[:, :3, :, :] = torch.nn.functional.normalize(phase_real_img[:, :3, :, :], dim=1) #dim=-1)
            # ipdb.set_trace()
            phase_real_img = phase_real_img.split(batch_gpu)
            # # phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            # phase_real_c = phase_real_c.to(device).split(batch_gpu)
            # ipdb.set_trace()
            if training_set._pose_cond and training_set.pose_cond_type=='axisangle' and not training_set.clothtype_cond:
                phase_real_c = phase_body_pose[:,3:66].to(device).split(batch_gpu)
            elif training_set._pose_cond and training_set.pose_cond_type=='vposer' and not training_set.clothtype_cond:
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
            elif training_set._pose_cond and training_set.pose_cond_type=='axisangle' and training_set.clothtype_cond:
                phase_real_c = torch.cat((phase_real_c, phase_body_pose[:,3:66]),1).to(device).split(batch_gpu)
            elif not training_set._pose_cond and training_set.pose_cond_type=='axisangle' and training_set.clothtype_cond:
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
            else:
                phase_real_c = phase_real_c.to(device).split(batch_gpu)

            phase_body_shape = phase_body_shape.to(device).split(batch_gpu)
            phase_body_pose = phase_body_pose.to(device).split(batch_gpu)
            phase_body_cam = phase_body_cam.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            # # all_gen_z = torch.ones_like(torch.randn([len(phases) * batch_size, G.z_dim], device=device)) * 0.1 ## To remove the torch.ones
            # all_gen_z = torch.from_numpy(np.load('/is/cluster/work/ssanyal/project_4/data/stylegan3/CAPE/new/random_normal_sample_1.npy')).type(torch.float32).to(device).repeat(len(phases) * batch_size, 1)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            # ipdb.set_trace()

            ## TODO: Check the following lines in case you want send different poses during the genrerator and discriminator training as stylegan does for class training
            # # D_phase_body_shape_pose = [training_set.get_shape_pose(np.random.randint(len(training_set))) for _ in range((len(phases)-1) * batch_size)]
            # # D_phase_body_shape = [D_phase_body_shape_pose[Dbp][0] for Dbp in range(len(D_phase_body_shape_pose))]
            # # D_phase_body_pose = [D_phase_body_shape_pose[Dbp][1] for Dbp in range(len(D_phase_body_shape_pose))]

            # # ipdb.set_trace()
            
            # if not training_set.clothtype_cond:
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            # else:
            #     all_gen_c = phase_real_c

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
        # for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            # ipdb.set_trace()
            # for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
            #     loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            # print('--------------------------------------------------')
            for real_img, real_c, gen_z, gen_c, body_shape, body_pose, body_cam in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_body_shape, phase_body_pose, phase_body_cam):
                # ipdb.set_trace()
                # print(phase.name)
                if not training_set._pose_cond:
                    if G.texture_render:
                        # ipdb.set_trace()
                        loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, body_shape=body_shape, body_pose=body_pose, body_cam=body_cam, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, cur_tick=cur_tick, rank=rank, G_geometry=G_geometry)
                    else:
                        loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, body_shape=body_shape, body_pose=body_pose, body_cam=body_cam, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, cur_tick=cur_tick, rank=rank)
                else: ## TODO: check this and rewrite in case you wanna send different poses for generator training and discriminator training phase especially gen_c = ?
                    # print(phase.name)
                    if G.texture_render:
                        # ipdb.set_trace()
                        loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, body_shape=body_shape, body_pose=body_pose, body_cam=body_cam, gen_z=gen_z, gen_c=real_c, gain=phase.interval, cur_nimg=cur_nimg, cur_tick=cur_tick, rank=rank, G_geometry=G_geometry)
                    else:
                        loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, body_shape=body_shape, body_pose=body_pose, body_cam=body_cam, gen_z=gen_z, gen_c=real_c, gain=phase.interval, cur_nimg=cur_nimg, cur_tick=cur_tick, rank=rank)
            phase.module.requires_grad_(False)
            # ipdb.set_trace()
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            # ipdb.set_trace()
            # b_shp = copy.deepcopy(phase_body_shape[0])
            # b_pse = copy.deepcopy(phase_body_pose[0])
            # ipdb.set_trace()
            # images = torch.cat([G_ema(z=z, c=c, body_shape=copy.deepcopy(phase_body_shape[0][:len(z)]), body_pose=copy.deepcopy(phase_body_pose[0][:len(z)]), body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), noise_mode='const')[0].cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            if G.texture_render:
                # mesheses = torch.cat([G_ema(z=z, c=c, body_shape=bs_, body_pose=bp_, body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), G_geometry=G_geometry, noise_mode='const')[1].verts_list()[0].cpu() for z, c, bs_, bp_ in zip(grid_z, grid_c, grid_bs, grid_bp)]).numpy()
                images = torch.cat([G_ema(z=z, c=c, body_shape=bs_, body_pose=bp_, body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), G_geometry=G_geometry, noise_mode='const')[0].cpu() for z, c, bs_, bp_ in zip(grid_z, grid_c, grid_bs, grid_bp)]).numpy()
                disp_images = torch.cat([G_ema(z=z, c=c, body_shape=bs_, body_pose=bp_, body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), G_geometry=G_geometry, noise_mode='const')[2].cpu() for z, c, bs_, bp_ in zip(grid_z, grid_c, grid_bs, grid_bp)]).numpy()
                # ipdb.set_trace()
            else:
                images = torch.cat([G_ema(z=z, c=c, body_shape=bs_, body_pose=bp_, body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), noise_mode='const')[0].cpu() for z, c, bs_, bp_ in zip(grid_z, grid_c, grid_bs, grid_bp)]).numpy()
                disp_images = torch.cat([G_ema(z=z, c=c, body_shape=bs_, body_pose=bp_, body_cam=copy.deepcopy(phase_body_cam[0][:len(z)]), noise_mode='const')[2].cpu() for z, c, bs_, bp_ in zip(grid_z, grid_c, grid_bs, grid_bp)]).numpy()
            # ipdb.set_trace()
            save_image_grid_withoutalphachannel(images[:, :3, :, :], os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid_withoutalphachannel(disp_images[:,:3,:,:], os.path.join(run_dir, f'fakes_disp{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            if G.conformnet:
                if cur_tick < 50:
                    save_image_grid_withoutalphachannel(disp_images[:,3:,:,:], os.path.join(run_dir, f'fakes_normals{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        if (snapshot_data is not None) and (len(metrics) > 0) and (done or cur_tick % (network_snapshot_ticks*1) == 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                if (geometry_pkl is not None):
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'], G_geometry = G_geometry,
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                else:
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory
        # ipdb.set_trace()
        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
