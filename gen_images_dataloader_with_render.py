# Modified from StyleGAN3 codebase

"""Generate images using pretrained network pickle.
Here we have provided the precomputed parameters used to generate the images for 
the main paper and the SUPMAT video in the website. One can easily 
modify these or make different combinations of these."""

from concurrent.futures import process
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import trimesh
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.renderer import BlendParams, sigmoid_alpha_blend, softmax_rgb_blend, look_at_view_transform, MeshRasterizer, RasterizationSettings, OrthographicCameras, FoVOrthographicCameras
from pytorch3d.renderer import TexturesUV, DirectionalLights, MeshRenderer, SoftPhongShader, PointLights
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle, axis_angle_to_quaternion, quaternion_to_axis_angle
from kornia.geometry.quaternion import Quaternion

import legacy

# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

class TextureRender(nn.Module):
    def __init__(
        self,
        blend_params= None,
        soft_blending = True,
        img_size = 256,
        faces_per_pixel = 1,
        azim = 0.,
        elev = 0.,
        blur_radius = 0.,
        SMPL_faces_path = None,
        SMPL_UV_coords_path = None,
        SMPL_UV_coords_faces_path = None,
        **render_kwargs, 
    ):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=(1., 1., 1.))
        self.soft_blending = soft_blending
        if self.soft_blending:
            self.faces_per_pixel = faces_per_pixel
        else:
            self.faces_per_pixel = 1

        # self.azim = torch.tensor(azim).repeat(1)
        # self.elev = torch.tensor(elev).repeat(1)

        # azim = np.linspace(0, 360, 8).astype(np.float32)
        azim = np.linspace(0, 360, 300).astype(np.float32)
        elev = np.array([0, 30, 330]).astype(np.float32)

        az, el = np.meshgrid(azim, elev)

        self.azim = az.flatten()
        self.elev = el.flatten()

        # R, T = look_at_view_transform(elev=self.elev, azim=self.azim)
        # # self.R = R.to(device)
        # # self.T = T.to(device)

        # camera setting
        self.dis = 100.0
        self.scale = 100.0
        self.mesh_y_center = -0.3 #0.0 # 

        self.reload_cam()

        R, T = look_at_view_transform(
            eye=[self.cam_pos[0]],
            at=((0, self.mesh_y_center, 0), ),
            up=((0, 1, 0), ),
        )

        self.register_buffer('R', R.contiguous())
        self.register_buffer('T', T.contiguous())

        self.num_views = 1 # len(R) #len(azim)

        mesh_faces = np.load('./data/smpl_faces.npy').astype(np.int64) # SMPL_faces_path.astype(np.int64) # ## TODO: Replace with proper SMPL faces
        self.register_buffer('mesh_faces', torch.from_numpy(mesh_faces))

        mesh_uv_coords = np.load('./data/smpl_uv_obj_vertextextureUVcoords.npy').astype(np.float32) 
        self.register_buffer('mesh_uv_coords', torch.from_numpy(mesh_uv_coords))

        mesh_uv_coords_faces = np.load('./data/smpl_uv_obj_vertextextureUVcoords_faces.npy').astype(np.int64)
        self.register_buffer('mesh_uv_coords_faces', torch.from_numpy(mesh_uv_coords_faces))
        
        # self.cameras = OrthographicCameras(device=device, focal_length=0.9, R=R, T=T)

        self.raster_cam = dict()
        # ipdb.set_trace()
        self.raster_settings = RasterizationSettings(image_size=img_size, 
                        blur_radius= blur_radius, 
                        faces_per_pixel=self.faces_per_pixel, 
                        max_faces_per_bin=10000,
                        cull_backfaces=False)
        # self.rastrizer = MeshRasterizer(raster_settings=self.raster_settings)#.to(device)
        self.lights = DirectionalLights(direction=((0, 0, 1),), ambient_color=((0.5, 0.5, 0.5), ), diffuse_color=((0.3, 0.3, 0.3), ), specular_color=((0.2, 0.2, 0.2), )) #  AmbientLights() #  
        self.rastrizer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=self.raster_settings), 
            shader=SoftPhongShader(lights=self.lights))

    def reload_cam(self):

        self.cam_pos = [
            (0, self.mesh_y_center, self.dis),
            (self.dis, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -self.dis),
            (-self.dis, self.mesh_y_center, 0),
        ]
    
    def PixelNormalCalcuate(self, meshes, fragments):
        # verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        vertex_normals = meshes.verts_normals_packed()
        # ipdb.set_trace()
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, torch.ones_like(fragments.bary_coords), faces_normals)
        return pixel_normals

    def forward(self, mesh_verts, mesh_faces=None, body_cam=None, text_img=None, f_len=None):
        batch_size = mesh_verts.shape[0]
        num_verts = mesh_verts.shape[1]
        # num_views = self.R.shape[0]
        device = mesh_verts.device
        # ipdb.set_trace()
        # Uncomment the line below to enable random view selection for rendering  each subject in the batch
        if body_cam==None:
            rand_views = torch.randint(0, len(self.R), (batch_size, )).to(device)
        elif body_cam.shape[-1]<4:
            rand_views = body_cam.squeeze(-1).type(self.mesh_faces.dtype)
        elif len(body_cam.shape)==1:
            rand_views = body_cam.type(self.mesh_faces.dtype)
            # print(rand_views)

        if f_len==None:
            f_len=0.9
        mesh_verts_for_views = mesh_verts.repeat(1, self.num_views, 1).reshape(batch_size * self.num_views, num_verts, 3)
        if mesh_faces == None:
            # mesh_faces = self.mesh_faces.unsqueeze(0).repeat(batch_size, 1, 1)
            mesh_faces = self.mesh_faces.unsqueeze(0).repeat(batch_size * self.num_views, 1, 1)
        # meshes = Meshes(verts=mesh_verts, faces=mesh_faces)
        meshes = Meshes(verts=mesh_verts_for_views, faces=mesh_faces)
        meshes.textures = TexturesUV(maps=text_img, faces_uvs=self.mesh_uv_coords_faces.unsqueeze(0).repeat(batch_size,1,1), verts_uvs=self.mesh_uv_coords.unsqueeze(0).repeat(batch_size,1,1))
        self.raster_cam['cameras'] = FoVOrthographicCameras(
            device=device,
            R=self.R.repeat(batch_size, 1, 1),
            T=self.T.repeat(batch_size, 1),
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3), ),
        )
        self.raster_cam['blend_params'] = self.blend_params
        self.rastrizer.to(device)
        textured_img = self.rastrizer(meshes, **self.raster_cam)

        return textured_img, meshes # normalized_rendered_img_alpha, meshes #, rendered_img[:, :, :, 3:]
    
# ---------------------------------------------------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--pose_path', help='provide the pose path', type=str, default=None, show_default=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    pose_path
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    # with dnnlib.util.open_url(network_pkl) as f:
    #     D = legacy.load_network_pkl(f)['D'].to(device) # type: ignore
    #     D.eval()
    if G.texture_render:
        # G_geometry_Path = '/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/cape_multisub/onlydispimg/troch1.11trained/00000-stylegan3-t-frames_view_2_pose_dispimages_normalized_shppse_correct_13485-gpus8-batch32-gamma2-ada-_/network-snapshot-022400.pkl'
        G_geometry_Path = '/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/cape_multisub/clothcond/dispimg/scratch/00004-stylegan3-t-frames_view_2_pose_dispimages_normalized_shppse_correct_over50000_withclothinglabel-gpus8-batch32-gamma2-ada-_/network-snapshot-013440.pkl'
        with dnnlib.util.open_url(G_geometry_Path) as f:
            G_geometry = legacy.load_network_pkl(f)['G_ema'].to(device)
    # num_samples = 50
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + '/meshes', exist_ok=True)
    os.makedirs(outdir + '/texmaps', exist_ok=True)
    # ipdb.set_trace()
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    sample_size = 120
    label_ = np.load('./data/supmat_labels_120.npy')[:sample_size]
    body_shape_ = np.load('./data/supmat_shape_120.npy')[:sample_size]
    bod_pose_ = np.load('./data/supmat_pose_120.npy')[:sample_size]

    # # For pose interpolation
    # bod_pose_ = np.load('/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/zalando/clothcond/conformnet/colorcond/supmat_conditions/pose_interp/1FI22O00N-M11@7_TO_2CR22E000-K12@15.1.npy')
    # bod_pose_ = np.repeat(np.expand_dims(bod_pose_[8], 0), body_shape_.shape[0], 0)


    # z = torch.from_numpy(z_).to(device)
    z = torch.from_numpy(np.random.RandomState(seeds).randn(label_.shape[0], G.z_dim)).to(device)
    # z = torch.from_numpy(np.load('/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/zalando/clothcond/conformnet/colorcond/supmat_conditions/liner_interp_z1_seed_1_12.npy')).to(device)[11] # For linear interpolation
    label_tot = torch.from_numpy(label_).to(device)
    body_shape = torch.from_numpy(body_shape_).to(device)
    body_pose = torch.from_numpy(bod_pose_).to(device)
    body_cam = torch.zeros([z.shape[0],1]).type(torch.float32).to(device)
    label_ct = label_tot[:, :6]
    label_cc = label_tot[:, 6:]
    # ipdb.set_trace()

    randperm_ = None

    # randperm_ = torch.randperm(body_pose.shape[0])
    # body_pose = body_pose[randperm_]
    # body_pose = body_pose[36].unsqueeze(0).repeat(z.shape[0],1)

    # randperm_ = torch.randperm(label_ct.shape[0])
    # label_ct = torch.tensor([1, 0, 0, 0, 0, 0]).type(label_ct.dtype).to(label_ct.device).unsqueeze(0).repeat(label_ct.shape[0], 1) # label_ct[randperm_]

    # randperm_ = torch.tensor(13) # torch.randperm(label_cc.shape[0])[0] #
    # label_cc = label_cc[randperm_].repeat(z.shape[0],1)

    if randperm_ is not None:
        np.save(f'{outdir}/seed{seeds[0]:04d}_randperm.npy', randperm_.numpy())

    label = torch.cat((label_ct, label_cc), 1)
    # label = label[torch.randperm(label.shape[0])]



    # # different_global_rot_for_sub_0 = torch.tensor([3.0137479, 0.012339, -0.8078111]).type(body_pose.dtype).to(body_pose.device) # 30 deg
    # # body_pose[0, :3] = different_global_rot_for_sub_0

    # required_rotation_angle = (np.pi/180) * (-25)

    # different_global_rot_eular = matrix_to_euler_angles(axis_angle_to_matrix(body_pose[:, :3]), "XYZ")
    # different_global_rot_eular[:, 1] = required_rotation_angle
    # body_pose[:, :3] = matrix_to_axis_angle(euler_angles_to_matrix(different_global_rot_eular, "XYZ"))


    if pose_path is not None:
        # body_shape_0 = torch.from_numpy(np.load(pose_path)['body_shape']).type(torch.float32).to(device)
        # # body_shape_0 = torch.from_numpy(np.load('/ps/project/tag_3d/data/CAPE_data/data/cape_release/minimal_body_shape/00032/00032_minimal.npy')).type(torch.float32).to(device)
        # body_shape = body_shape_0.unsqueeze(0).repeat(num_samples, 1, 1)
        body_pose_ = torch.from_numpy(np.load(pose_path)['body_pose']).type(torch.float32).to(device)
        body_pose = body_pose_.unsqueeze(0).repeat(z.shape[0],1)

    zt_interpolate = False
    if zt_interpolate:
        t_vals = torch.from_numpy(np.linspace(0, 1, 120)).to(device)
        sub_id = 89
        seed_1 = torch.from_numpy(np.random.RandomState([10]).randn(z.shape[0], G.z_dim)).to(device)
        seed_2 = torch.from_numpy(np.random.RandomState([101]).randn(z.shape[0], G.z_dim)).to(device)

        intermediate_points = (1-t_vals)[:, None] * seed_1[sub_id][None, :] + t_vals[:, None] * seed_2[sub_id][None, :]
        z = intermediate_points
        label = label[sub_id].unsqueeze(0).repeat(z.shape[0],1)
        body_pose = body_pose[sub_id].unsqueeze(0).repeat(z.shape[0],1)
        body_shape = body_shape[sub_id].unsqueeze(0).repeat(z.shape[0],1,1)
        

    # ipdb.set_trace()
    inference_texrend = TextureRender(img_size=256).to(device)

    for seed_idx, seed in enumerate(seeds):
        ## Mapping network
        z_geo = z
        # z_geo = torch.from_numpy(np.random.RandomState(seed).randn(z.shape[0], G.z_dim)).to(device)
        # ws = G.mapping(torch.from_numpy(np.random.RandomState(seed).randn(z.shape[0], G.z_dim)).to(device), label, truncation_psi=truncation_psi)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)
        ws_geo = G_geometry.mapping(z_geo, torch.cat((label[:, :6], body_pose[:,3:66]),1), truncation_psi=1.0)

        ## Texture Network
        ws = ws.to(torch.float32).unbind(dim=1)
        ws_geo = ws_geo.to(torch.float32).unbind(dim=1)
        # ipdb.set_trace()
        # Execute layers.
        x = G.synthesis.input(ws[0])
        x_geo = G_geometry.synthesis.input(ws_geo[0])
        for name, w, w_geo in zip(G.synthesis.layer_names, ws[1:], ws_geo[1:]):
            x = x + x_geo
            x_geo = getattr(G_geometry.synthesis, name)(x_geo, w_geo, update_emas=False)
            x = getattr(G.synthesis, name)(x, w, update_emas=False)
            # print('x->',x.shape)
            # print('x_geo->',x.shape)
        if G.synthesis.output_scale != 1:
            x = x * G.synthesis.output_scale

        if G_geometry.synthesis.output_scale != 1:
            x_geo = x_geo * G_geometry.synthesis.output_scale

        # Ensure correct shape and dtype.
        UV_geo = x_geo.to(torch.float32)
        UV_tex = x.to(torch.float32)

        disp_img_geo = (UV_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
        vert_disps = G_geometry.displacement_Layer(disp_img_geo)

        disp_img_unscaled = UV_tex * 1.0

        clothed_body_shape = body_shape + vert_disps
        # posed_body = G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3])
        posed_body = G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))
        # ipdb.set_trace()
        img, meshes = G.TextureRender(posed_body, body_cam=body_cam, text_img=UV_tex.permute(0,2,3,1))

        norm_img, _ = G.Normalrender(posed_body, body_cam=body_cam)

        rendered_mesh, _ = inference_texrend(posed_body, body_cam=body_cam, text_img=UV_tex.permute(0,2,3,1))

        for i in range(len(body_shape)):
            plt.imsave(f'{outdir}/seed{seed:04d}_{i:03}.png', (rendered_mesh[i][:,:,:3] * 0.5 + 0.5).clamp(0, 1).cpu().numpy())
            # plt.imsave(f'{outdir}/seed{seed:04d}_{i:03}.png', (img[i][:,:,:3] * 0.5 + 0.5).clamp(0, 1).cpu().numpy())
            # plt.imsave(f'{outdir}/texmaps/seed{seed:04d}_dispimg_{i:03}.png', (disp_img_unscaled * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[i][:,:,:3].cpu().numpy())
            # np.save(f'{outdir}/texmaps/seed{seed:04d}_dispimg_{i:03}.npy', (disp_img_unscaled).permute(0, 2, 3, 1)[i][:,:,:3].cpu().numpy())
            # if G.texture_render:
            #     # plt.imsave(f'{outdir}/seed{seed:04d}_dispimg_geo_{i:03}.png', (disp_img_geo_unscaled * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[i].cpu().numpy())
            #     plt.imsave(f'{outdir}/seed{seed:04d}_normals_{i:03}.png', (norm_img * 0.5 + 0.5).clamp(0, 1)[i][:,:,:3].cpu().numpy())
            #     plt.imsave(f'{outdir}/seed{seed:04d}_dispimg_geo_{i:03}.png', (UV_geo * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)[i][:,:,:3].cpu().numpy())
            # plt.imsave(f'{outdir}/seed{seed:04d}_dispimg_{i:03}.png', img_mesh[i].cpu().numpy())
            mesh = trimesh.Trimesh(vertices=meshes.verts_list()[i].cpu().numpy(), faces=meshes.faces_list()[i].cpu().numpy(), process=False)
            mesh.export(f'{outdir}/meshes/seed{seed:04d}_{i:03}.obj')



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
