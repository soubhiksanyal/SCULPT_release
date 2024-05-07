# Modified from StyleGAN3 codebase

"""Main file to start SCULPT training."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
import legacy
import ipdb

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--gen_grad_pen', help='gradient penalty for generator', metavar='BOOL',          type=bool, default=False, show_default=False)


# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)


# Renderer hyperparameters.
@click.option('--blur_radius',      help='rastarizer blur radius', metavar='FLOAT',                              type=click.FloatRange(min=0), default=1e-5, show_default=True)
@click.option('--faces_per_pixel',  help='number of faces rquired for interpolating a pixel', metavar='INT',     type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--azim',             help='azimuthal angel of the camera location in degree', metavar='FLOAT',    type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--elev',             help='elevation angel of the camera location in degree', metavar='FLOAT',    type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--only_disp_img',    help='If Turned on only displacement image is generated', metavar='BOOL',    type=bool, default=False, show_default=True)
@click.option('--resume_pretrain_cape',    help='If Turned on only displacement image is generated', metavar='BOOL',    type=bool, default=False, show_default=True)

# SMPL mesh hyperparameters.
@click.option('--smpl_uv_mask_path',     help='SMPL_uv_mask_path', metavar='NPY',          type=click.Path(exists=True), default='./data/smpl_uv_mask_256.npy', show_default=True)
@click.option('--smpl_faces_path',       help='SMPL_faces_path', metavar='NPY',            type=click.Path(exists=True), default='./data/smpl_faces.npy', show_default=True)
@click.option('--smpl_model_path',       help='SMPL models path', metavar='PKL',           type=click.Path(exists=True), default='./data/SMPL_NEUTRAL.pkl', show_default=True)
@click.option('--img2mesh_map_path',     help='unique_v2p_mapper_path', metavar='PKL',           type=click.Path(exists=True), default='./data/vertex2pixel_256.npy', show_default=True)
@click.option('--cano_shape',            help='Load v_shaped instead of betas', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--mask_disp_map',         help='mask the generated displacement maps', metavar='BOOL',   type=bool, default=False, show_default=False)
@click.option('--disp_activatn_type',    help='Activation type after displacement layer',  type=click.Choice(['sigmoid', 'tanh']), default='tanh', show_default=True)
@click.option('--disp_scale',            help='Scaling factor for the displacements after activation', metavar='FLOAT',    type=click.FloatRange(max=1.0), default=0.1, show_default=True)
@click.option('--seperate_disp_map',     help='Generate 10x3 channel disp maps for different parts', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--sep_disp_map_sampling', help='Generate 10x3 channel disp maps for different parts', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--spiral_conv',           help='Perform spiral convolution at the end on the posed mesh', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--texture_render',        help='Perform spiral convolution at the end on the posed mesh', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--geometry',              help='Load geometry network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--smpl_uv_coords_path',        help='SMPL_UV_coords_path', metavar='NPY',            type=click.Path(exists=True), default='./data/smpl_uv_obj_vertextextureUVcoords.npy', show_default=True)
@click.option('--smpl_uv_coords_faces_path',  help='SMPL_UV_coords_faces_path', metavar='NPY',            type=click.Path(exists=True), default='./data/smpl_uv_obj_vertextextureUVcoords_faces.npy', show_default=True)


# Weights for mesh smoothness 
@click.option('--edge_loss',           help='weight of the edge loss on the mesh', metavar='FLOAT',                              type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--normal_loss',         help='weight of the normal consistency loss on the mesh', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--laplacian_loss',      help='weight of the laplacian loss on the mesh', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--mesh_smooth_w',       help='weight of the laplacian loss on the mesh', metavar='FLOAT',                         type=click.FloatRange(min=0), default=1.0, show_default=True)

# Conditioning on pose
@click.option('--pose_cond',           help='Turn on conditioning on pose', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--pose_cond_type',      help='Pose conditioning type mode',                             type=click.Choice(['axisangle', '6Drot', 'vposer']), default='axisangle', show_default=True)
@click.option('--clothtype_cond',      help='Turn on conditioning on clothing type', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--conformnet',          help='Turn on to condition the texture network with geometry', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--conditional_d',       help='Turn on to condition the discriminator network with normals', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--guass_blur_normals',  help='Turn on gaussian bluring on the normal images', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--patch_d',             help='Turn on patch based discriminator on images', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--colorcond',           help='Turn on conditioning on crude color labels', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--patch_d_with_glob',   help='Turn on patch based discriminator with global discrim', metavar='BOOL',   type=bool, default=False, show_default=True)



def main(**kwargs):
    """ Check trainer_cluster_mul.sh file for starting the training process.
    There are more arguments than actually required for the SCULPT project because
    this was planned to a part of a broder project. One do not need to use all the 
    arguments or check carefully before using it."""

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), render_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.pose_cond = opts.pose_cond
    c.training_set_kwargs.pose_cond_type = opts.pose_cond_type
    c.training_set_kwargs.clothtype_cond = opts.clothtype_cond
    c.training_set_kwargs.conditional_D = opts.conditional_d
    c.training_set_kwargs.colorcond = opts.colorcond

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.disp_activatn_type = opts.disp_activatn_type
    c.G_kwargs.disp_scale = opts.disp_scale
    c.G_kwargs.only_disp_img = opts.only_disp_img
    c.G_kwargs.conformnet = opts.conformnet
    c.G_kwargs.resume_pretrain_cape = opts.resume_pretrain_cape
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.D_kwargs.patch_d = opts.patch_d
    # c.D_kwargs.patch_d_with_glob = opts.patch_d_with_glob
    # c.D_kwargs.ndf = 64
    # c.D_kwargs.n_layers_patchdiscrim = 3
    # c.D_kwargs.norm_layer_patchdiscrim = 'BatchNorm2d'
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.edge_loss = opts.edge_loss
    c.loss_kwargs.normal_loss = opts.normal_loss
    c.loss_kwargs.laplacian_loss = opts.laplacian_loss
    c.loss_kwargs.mesh_smooth_w = opts.mesh_smooth_w
    c.loss_kwargs.gen_grad_pen = opts.gen_grad_pen
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # c.G_reg_interval = 5

    c.G_kwargs.render_kwargs.blur_radius = opts.blur_radius
    c.G_kwargs.render_kwargs.faces_per_pixel = opts.faces_per_pixel
    c.G_kwargs.render_kwargs.azim = opts.azim
    c.G_kwargs.render_kwargs.elev = opts.elev
    c.G_kwargs.render_kwargs.SMPL_faces_path = opts.smpl_faces_path
    c.G_kwargs.render_kwargs.SMPL_uv_mask_path = opts.smpl_uv_mask_path
    c.G_kwargs.render_kwargs.smpl_model_path = opts.smpl_model_path
    c.G_kwargs.render_kwargs.cano_shape = opts.cano_shape
    c.G_kwargs.render_kwargs.unique_v2p_mapper_path = opts.img2mesh_map_path
    c.G_kwargs.mask_disp_map = opts.mask_disp_map
    c.G_kwargs.seperate_disp_map = opts.seperate_disp_map
    c.G_kwargs.sep_disp_map_sampling = opts.sep_disp_map_sampling
    c.G_kwargs.spiral_conv = opts.spiral_conv
    c.G_kwargs.texture_render = opts.texture_render
    c.G_kwargs.guass_blur_normals = opts.guass_blur_normals
    if opts.texture_render:
        c.G_kwargs.render_kwargs.SMPL_UV_coords_path = opts.smpl_uv_coords_path
        c.G_kwargs.render_kwargs.SMPL_UV_coords_faces_path = opts.smpl_uv_coords_faces_path

    
    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

        # if not opts.resume_pretrain_cape:
        #     c.ada_kimg = 100 # Make ADA react faster at the beginning.
        #     c.ema_rampup = None # Disable EMA rampup.
        #     c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    if opts.geometry is not None:
        c.geometry_pkl = opts.geometry
        # with dnnlib.util.open_url(c.geometry_pkl) as f:
        #     c.G_geometry = legacy.load_network_pkl(f)['G_ema']

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}-{opts.aug:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # with torch.autograd.detect_anomaly(): # Uncomment this line and comment the last main line while debugging in GPU clusters
    #     main() # pylint: disable=no-value-for-parameter
    main() # pylint: disable=no-value-for-parameter
    

#----------------------------------------------------------------------------
