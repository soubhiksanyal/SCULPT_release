# Copyright (c) 2024, Soubhik Sanyal and MPI for Intelligent Systems.  All rights reserved.

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import BlendParams, sigmoid_alpha_blend, softmax_rgb_blend, look_at_view_transform, MeshRasterizer, RasterizationSettings, OrthographicCameras, FoVOrthographicCameras
from pytorch3d.renderer import TexturesUV, DirectionalLights, MeshRenderer, SoftPhongShader, PointLights
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from torch_utils import persistence
from smplx import SMPL as _SMPL
from training.lbs import lbs
import ipdb

# @persistence.persistent_class
# class NormalRender(nn.Module):
#     def __init__(
#         self,
#         blend_params= None,
#         soft_blending = True,
#         img_size = 256,
#         faces_per_pixel = 1,
#         azim = 0.,
#         elev = 0.,
#         blur_radius = 0.,
#         SMPL_faces_path = None,
#         **render_kwargs, 
#     ):
#         super().__init__()
#         self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=(0., 0., 0.))
#         self.soft_blending = soft_blending
#         if self.soft_blending:
#             self.faces_per_pixel = faces_per_pixel
#         else:
#             self.faces_per_pixel = 1

#         # self.azim = torch.tensor(azim).repeat(1)
#         # self.elev = torch.tensor(elev).repeat(1)

#         # azim = np.linspace(0, 360, 8).astype(np.float32)
#         azim = np.linspace(0, 360, 300).astype(np.float32)
#         elev = np.array([0, 30, 330]).astype(np.float32)

#         az, el = np.meshgrid(azim, elev)

#         self.azim = az.flatten()
#         self.elev = el.flatten()

#         R, T = look_at_view_transform(elev=self.elev, azim=self.azim)
#         # self.R = R.to(device)
#         # self.T = T.to(device)

#         self.register_buffer('R', R)
#         self.register_buffer('T', T)

#         self.num_views = 1 # len(R) #len(azim)

#         mesh_faces = np.load(SMPL_faces_path).astype(np.int64) # SMPL_faces_path.astype(np.int64) # ## TODO: Replace with proper SMPL faces
#         self.register_buffer('mesh_faces', torch.from_numpy(mesh_faces))
        
#         # self.cameras = OrthographicCameras(device=device, focal_length=0.9, R=R, T=T)

#         self.raster_cam = dict()

#         self.raster_settings_sil = RasterizationSettings(image_size=img_size, 
#                         blur_radius= np.log(1. / 1e-4 - 1.) * blur_radius, 
#                         faces_per_pixel=self.faces_per_pixel, 
#                         max_faces_per_bin=10000,
#                         cull_backfaces=False)
#         self.rastrizer_sil = MeshRasterizer(raster_settings=self.raster_settings_sil)#.to(device)

#         self.raster_settings_normals = RasterizationSettings(image_size=img_size, 
#                         blur_radius= 0.0, 
#                         faces_per_pixel=self.faces_per_pixel, 
#                         max_faces_per_bin=10000,
#                         cull_backfaces=False)

#         self.rastrizer_normal = MeshRasterizer(raster_settings=self.raster_settings_normals)#.to(device)

#     def PixelNormalCalcuate(self, meshes, fragments):
#         # verts = meshes.verts_packed()
#         faces = meshes.faces_packed()
#         vertex_normals = meshes.verts_normals_packed()
#         # ipdb.set_trace()
#         faces_normals = vertex_normals[faces]
#         pixel_normals = interpolate_face_attributes(
#             fragments.pix_to_face, torch.ones_like(fragments.bary_coords), faces_normals)
#         return pixel_normals

#     def softmax_normal_blend(
#         self,
#         colors: torch.Tensor,
#         fragments,
#         blend_params: BlendParams,
#         znear: Union[float, torch.Tensor] = 1.0,
#         zfar: Union[float, torch.Tensor] = 100,
#     ) -> torch.Tensor:
#         """
#         Modified from pytorch3D
#         RGB and alpha channel blending to return an RGBA image based on the method
#         proposed in [1]
#         - **RGB** - blend the colors based on the 2D distance based probability map and
#             relative z distances.
#         - **A** - blend based on the 2D distance based probability map.

#         Args:
#             colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
#             fragments: namedtuple with outputs of rasterization. We use properties
#                 - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
#                 of the faces (in the packed representation) which
#                 overlap each pixel in the image.
#                 - dists: FloatTensor of shape (N, H, W, K) specifying
#                 the 2D euclidean distance from the center of each pixel
#                 to each of the top K overlapping faces.
#                 - zbuf: FloatTensor of shape (N, H, W, K) specifying
#                 the interpolated depth from each pixel to to each of the
#                 top K overlapping faces.
#             blend_params: instance of BlendParams dataclass containing properties
#                 - sigma: float, parameter which controls the width of the sigmoid
#                 function used to calculate the 2D distance based probability.
#                 Sigma controls the sharpness of the edges of the shape.
#                 - gamma: float, parameter which controls the scaling of the
#                 exponential function used to control the opacity of the color.
#                 - background_color: (3) element list/tuple/torch.Tensor specifying
#                 the RGB values for the background color.
#             znear: float, near clipping plane in the z direction
#             zfar: float, far clipping plane in the z direction

#         Returns:
#             RGBA pixel_colors: (N, H, W, 4)

#         [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
#         Image-based 3D Reasoning'
#         """

#         N, H, W, K = fragments.pix_to_face.shape
#         device = fragments.pix_to_face.device
#         # pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
#         pixel_colors = torch.ones((N, H, W, 3), dtype=colors.dtype, device=colors.device)
#         background_ = blend_params.background_color
#         if not isinstance(background_, torch.Tensor):
#             background = torch.tensor(background_, dtype=torch.float32, device=device)
#         else:
#             background = background_.to(device)

#         # Weight for background color
#         eps = 1e-10

#         # Mask for padded pixels.
#         mask = fragments.pix_to_face >= 0

#         # Sigmoid probability map based on the distance of the pixel to the face.
#         prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

#         # The cumulative product ensures that alpha will be 0.0 if at least 1
#         # face fully covers the pixel as for that face, prob will be 1.0.
#         # This results in a multiplication by 0.0 because of the (1.0 - prob)
#         # term. Therefore 1.0 - alpha will be 1.0.
#         # alpha = torch.prod((1.0 - prob_map), dim=-1) ##TODO: Commented for stopping the alpha channel 

#         # Weights for each face. Adjust the exponential by the max z to prevent
#         # overflow. zbuf shape (N, H, W, K), find max over K.
#         # TODO: there may still be some instability in the exponent calculation.

#         # Reshape to be compatible with (N, H, W, K) values in fragments
#         if torch.is_tensor(zfar):
#             # pyre-fixme[16]
#             zfar = zfar[:, None, None, None]
#         if torch.is_tensor(znear):
#             # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
#             #  `__getitem__`.
#             znear = znear[:, None, None, None]

#         z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
#         z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
#         weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

#         # Also apply exp normalize trick for the background color weight.
#         # Clamp to ensure delta is never 0.
#         # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
#         delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

#         # Normalize weights.
#         # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
#         denom = weights_num.sum(dim=-1)[..., None] + delta

#         # Sum: weights * textures + background color
#         weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
#         weighted_background = delta * background
#         pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
#         # pixel_colors[..., 3] = 1.0 - alpha

#         return pixel_colors

#     def soft_silhouette_blend(self, fragments):
#         """
#         Blend soft silhouettes
#         Concept taken from pytorch3d SoftSilhouetteShader
#         """

#         # colors = torch.ones_like(fragments.bary_coords)
#         # # blend_params = kwargs.get("blend_params", self.blend_params)
#         # images = sigmoid_alpha_blend(colors, fragments, self.blend_params)

#         # Mask for padded pixels.
#         mask = fragments.pix_to_face >= 0

#         prob_map = torch.sigmoid(-fragments.dists / self.blend_params.sigma) * mask

#         alpha = torch.prod((1.0 - prob_map), dim=-1)
#         silhouette = 1.0 - alpha
#         return silhouette
           

#     def forward(self, mesh_verts, mesh_faces=None, body_cam=None, f_len=None):
#         batch_size = mesh_verts.shape[0]
#         num_verts = mesh_verts.shape[1]
#         # num_views = self.R.shape[0]
#         device = mesh_verts.device
#         # ipdb.set_trace()
#         # Uncomment the line below to enable random view selection for rendering  each subject in the batch
#         if body_cam==None:
#             rand_views = torch.randint(0, len(self.R), (batch_size, )).to(device)
#         elif body_cam.shape[-1]<4:
#             rand_views = body_cam.squeeze(-1).type(self.mesh_faces.dtype)
#         elif len(body_cam.shape)==1:
#             rand_views = body_cam.type(self.mesh_faces.dtype)
#             # print(rand_views)

#         if f_len==None:
#             f_len=0.9
#         mesh_verts_for_views = mesh_verts.repeat(1, self.num_views, 1).reshape(batch_size * self.num_views, num_verts, 3)
#         if mesh_faces == None:
#             # mesh_faces = self.mesh_faces.unsqueeze(0).repeat(batch_size, 1, 1)
#             mesh_faces = self.mesh_faces.unsqueeze(0).repeat(batch_size * self.num_views, 1, 1)
#         # meshes = Meshes(verts=mesh_verts, faces=mesh_faces)
#         meshes = Meshes(verts=mesh_verts_for_views, faces=mesh_faces)
#         # ipdb.set_trace()

#         # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=self.R.repeat(batch_size, 1, 1), T=self.T.repeat(batch_size, 1))

#         # Uncomment the line below to enable random view selection for rendering  each subject in the batch
#         self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=torch.index_select(self.R, 0, rand_views), T=torch.index_select(self.T, 0, rand_views))

#         # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=body_cam[:, :, :3], T=body_cam[:, :, 3])


#         # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=0.9, R=self.R, T=self.T)
#         fragments_sil = self.rastrizer_sil(meshes, **self.raster_cam)
#         fragments_normal = self.rastrizer_normal(meshes, **self.raster_cam)
#         # ipdb.set_trace()
#         pixel_normals = F.normalize(self.PixelNormalCalcuate(meshes, fragments_normal), dim=-1)
#         # rendered_img = softmax_rgb_blend(pixel_normals, fragments, self.blend_params)
#         rendered_img = self.softmax_normal_blend(pixel_normals, fragments_normal, self.blend_params)
#         # # normal_images = (F.normalize(rendered_img[:, :,:,:3], dim=3) + 1.) / 2. * normal_images[:, :, :, 3:]
#         # # normal_images = (rendered_img[:, :,:,:3] + 1.) / 2. * rendered_img[:, :, :, 3:]
#         # normal_images = F.normalize(((rendered_img[:, :,:,:3] + 1.) / 2.) * rendered_img[:, :, :, 3:], dim=3)
#         # ipdb.set_trace()
#         # print("rendered_img->", rendered_img[:, :, :, :3].max(), rendered_img.dtype)
#         normalized_rendered_img = F.normalize(rendered_img[:, :, :, :3], dim=-1) # rendered_img[:, :, :, :3]#

#         silhoutte_img = self.soft_silhouette_blend(fragments_sil)

#         # ipdb.set_trace()

#         # normalized_rendered_img_alpha = torch.cat((normalized_rendered_img, rendered_img[:, :, :, 3:]), -1)
#         normalized_rendered_img_alpha = torch.cat((normalized_rendered_img, silhoutte_img[:, :, :, None]), -1)

#         return normalized_rendered_img_alpha, meshes # normalized_rendered_img_alpha, meshes #, rendered_img[:, :, :, 3:]



@persistence.persistent_class
class NormalRender(nn.Module):
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
        background_color = 0.,
        **render_kwargs, 
    ):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=(background_color, background_color, background_color))
        self.soft_blending = soft_blending
        # if self.soft_blending:
        #     self.faces_per_pixel = faces_per_pixel
        # else:
        #     self.faces_per_pixel = 1
        self.faces_per_pixel = 1
        blur_radius = 0.
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

        mesh_faces = np.load(SMPL_faces_path).astype(np.int64) # SMPL_faces_path.astype(np.int64) # ## TODO: Replace with proper SMPL faces
        self.register_buffer('mesh_faces', torch.from_numpy(mesh_faces))
        
        # self.cameras = OrthographicCameras(device=device, focal_length=0.9, R=R, T=T)

        self.raster_cam = dict()
        # ipdb.set_trace()
        self.raster_settings = RasterizationSettings(image_size=img_size, 
                        blur_radius= np.log(1. / 1e-4 - 1.) * blur_radius, 
                        faces_per_pixel=self.faces_per_pixel, 
                        max_faces_per_bin=10000,
                        cull_backfaces=False)
        self.rastrizer = MeshRasterizer(raster_settings=self.raster_settings)#.to(device)

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

    def forward(self, mesh_verts, mesh_faces=None, body_cam=None, f_len=None):
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
        # ipdb.set_trace()

        # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=self.R.repeat(batch_size, 1, 1), T=self.T.repeat(batch_size, 1))

        # Uncomment the line below to enable random view selection for rendering  each subject in the batch
        # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=torch.index_select(self.R, 0, rand_views), T=torch.index_select(self.T, 0, rand_views))

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

        # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=f_len, R=body_cam[:, :, :3], T=body_cam[:, :, 3])


        # self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=0.9, R=self.R, T=self.T)
        fragments = self.rastrizer(meshes, **self.raster_cam)
        # ipdb.set_trace()
        pixel_normals = F.normalize(self.PixelNormalCalcuate(meshes, fragments), dim=-1)
        rendered_img = softmax_rgb_blend(pixel_normals, fragments, self.blend_params)
        # # normal_images = (F.normalize(rendered_img[:, :,:,:3], dim=3) + 1.) / 2. * normal_images[:, :, :, 3:]
        # # normal_images = (rendered_img[:, :,:,:3] + 1.) / 2. * rendered_img[:, :, :, 3:]
        # normal_images = F.normalize(((rendered_img[:, :,:,:3] + 1.) / 2.) * rendered_img[:, :, :, 3:], dim=3)
        # ipdb.set_trace()
        # print("rendered_img->", rendered_img[:, :, :, :3].max(), rendered_img.dtype)
        if self.faces_per_pixel==1:
            # ipdb.set_trace()
            # mask = (~(rendered_img[:, :, :, 1]==1))
            # normalized_rendered_img = F.normalize(rendered_img[:, :, :, :3], dim=-1) * mask.type(torch.float32).unsqueeze(-1) + (~mask).type(torch.float32).unsqueeze(-1)
            mask = (torch.norm(pixel_normals, dim=-1)).type(torch.bool)
            normalized_rendered_img = pixel_normals.squeeze(-2) + (~mask).type(torch.float32)
            # normalized_rendered_img = rendered_img[:, :, :, :3]
        else:
            normalized_rendered_img = F.normalize(rendered_img[:, :, :, :3], dim=-1) # rendered_img[:, :, :, :3]#
        # ipdb.set_trace()
        normalized_rendered_img_alpha = torch.cat((normalized_rendered_img, rendered_img[:, :, :, 3:]), -1)

        return normalized_rendered_img_alpha, meshes # normalized_rendered_img_alpha, meshes #, rendered_img[:, :, :, 3:]


# @persistence.persistent_class
# class NormalRender_old(nn.Module):
#     def __init__(
#         self,
#         blend_params= None,
#         soft_blending = True,
#         img_size = 256,
#         faces_per_pixel = 4,
#         azim = 0.,
#         elev = 0.,
#         blur_radius = 0., #0.000921
#         SMPL_faces_path = None,
#         **render_kwargs, 
#     ):
#         super().__init__()
#         self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=(-1., -1., -1.))
#         self.soft_blending = soft_blending
#         if self.soft_blending:
#             self.faces_per_pixel = faces_per_pixel
#         else:
#             self.faces_per_pixel = 1

#         self.azim = torch.tensor(azim).repeat(1)
#         self.elev = torch.tensor(elev).repeat(1)

#         R, T = look_at_view_transform(elev=self.elev, azim=self.azim)
#         # self.R = R.to(device)
#         # self.T = T.to(device)

#         self.register_buffer('R', R)
#         self.register_buffer('T', T)

#         mesh_faces = np.load(SMPL_faces_path).astype(np.int64) ## TODO: Replace with proper SMPL faces
#         self.register_buffer('mesh_faces', torch.from_numpy(mesh_faces))
        
#         # self.cameras = OrthographicCameras(device=device, focal_length=0.9, R=R, T=T)

#         self.raster_cam = dict()

#         self.raster_settings = RasterizationSettings(image_size=img_size, blur_radius=blur_radius, faces_per_pixel=self.faces_per_pixel)
#         self.rastrizer = MeshRasterizer(raster_settings=self.raster_settings)#.to(device)

#     # def vertex_normals_batch(self, vertices, faces):
#     #     """
#     #     :param vertices: [batch size, number of vertices, 3]
#     #     :param faces: [batch size, number of faces, 3]
#     #     :return: [batch size, number of vertices, 3]
#     #     """
#     #     assert (vertices.ndimension() == 3)
#     #     assert (faces.ndimension() == 3)
#     #     assert (vertices.shape[0] == faces.shape[0])
#     #     assert (vertices.shape[2] == 3)
#     #     assert (faces.shape[2] == 3)
#     #     bs, nv = vertices.shape[:2]
#     #     bs, nf = faces.shape[:2]
#     #     device = vertices.device
#     #     normals = torch.zeros(bs * nv, 3).to(device)

#     #     faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
#     #     vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

#     #     faces = faces.reshape(-1, 3)
#     #     vertices_faces = vertices_faces.reshape(-1, 3, 3)

#     #     normals.index_add_(0, faces[:, 1].long(), 
#     #                     torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
#     #     normals.index_add_(0, faces[:, 2].long(), 
#     #                     torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
#     #     normals.index_add_(0, faces[:, 0].long(),
#     #                     torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

#     #     normals = F.normalize(normals, eps=1e-6, dim=1)
#     #     normals = normals.reshape((bs, nv, 3))
#     #     # pytorch only supports long and byte tensors for indexing
#     #     return normals
        
#     # def PixelNormalCalcuate(self, meshe_verts, mesh_faces, fragments):
#     #     vertex_normals = self.vertex_normals_batch(meshe_verts, mesh_faces).view(-1,3)
#     #     faces_normals = vertex_normals.view(-1, 3)[mesh_faces.view(-1, 3)]        
#     #     # pixel_normals = interpolate_face_attributes(
#     #     #     fragments.pix_to_face, fragments.bary_coords, faces_normals)
#     #     pixel_normals = interpolate_face_attributes(
#     #         fragments.pix_to_face, torch.ones_like(fragments.bary_coords), faces_normals)
#     #     return pixel_normals

#     def PixelNormalCalcuate(self, meshes, fragments):
#         # verts = meshes.verts_packed()
#         faces = meshes.faces_packed()
#         vertex_normals = meshes.verts_normals_packed()
#         # ipdb.set_trace()
#         faces_normals = vertex_normals[faces]
#         pixel_normals = interpolate_face_attributes(
#             fragments.pix_to_face, torch.ones_like(fragments.bary_coords), faces_normals)
#         return pixel_normals

#     def forward(self, mesh_verts, mesh_faces=None):
#         batch_size = mesh_verts.shape[0]
#         device = mesh_verts.device
#         if mesh_faces == None:
#             mesh_faces = self.mesh_faces.unsqueeze(0).repeat(batch_size, 1, 1)
#         meshes = Meshes(verts=mesh_verts, faces=mesh_faces)
#         # ipdb.set_trace()
#         self.raster_cam['cameras'] = OrthographicCameras(device=device, focal_length=0.9, R=self.R.repeat(batch_size, 1, 1), T=self.T.repeat(batch_size, 1))
#         fragments = self.rastrizer(meshes, **self.raster_cam)
#         pixel_normals = F.normalize(self.PixelNormalCalcuate(meshes, fragments), dim=-1)
#         rendered_img = softmax_rgb_blend(pixel_normals, fragments, self.blend_params)
#         # # normal_images = (F.normalize(rendered_img[:, :,:,:3], dim=3) + 1.) / 2. * normal_images[:, :, :, 3:]
#         # # normal_images = (rendered_img[:, :,:,:3] + 1.) / 2. * rendered_img[:, :, :, 3:]
#         # normal_images = F.normalize(((rendered_img[:, :,:,:3] + 1.) / 2.) * rendered_img[:, :, :, 3:], dim=3)
#         # ipdb.set_trace()
#         # print("rendered_img->", rendered_img[:, :, :, :3].max(), rendered_img.dtype)
#         return rendered_img[:, :, :, :3] # F.normalize(rendered_img[:, :,:,:3], dim=3) # normal_images #   

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

        mesh_faces = np.load(SMPL_faces_path).astype(np.int64) # SMPL_faces_path.astype(np.int64) # ## TODO: Replace with proper SMPL faces
        self.register_buffer('mesh_faces', torch.from_numpy(mesh_faces))

        mesh_uv_coords = np.load(SMPL_UV_coords_path).astype(np.float32) 
        self.register_buffer('mesh_uv_coords', torch.from_numpy(mesh_uv_coords))

        mesh_uv_coords_faces = np.load(SMPL_UV_coords_faces_path).astype(np.int64)
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
        self.lights = DirectionalLights(direction=((0, 0, 1),)) # PointLights(location=[[0.0, 0, 3.0]]) # AmbientLights() #  
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



@persistence.persistent_class
class SMPL_Layer(nn.Module):
    def __init__(self, smpl_model_path, cano_shape = True, **render_kwargs):
        # super(SMPL_Layer, self).__init__(smpl_model_path)
        # super(SMPL_Layer, self).__init__(smpl_model_path, create_body_pose=False, create_betas=False, create_global_orient=False, create_transl=False)
        super().__init__()
        self.cano_shape = cano_shape
        self.smpl = _SMPL(smpl_model_path, create_body_pose=False, create_betas=False, create_global_orient=False, create_transl=False)
        ICON_compatible_rendering = torch.tensor([1.0, -1.0, -1.0])
        self.register_buffer('ICON_compatible_rendering', ICON_compatible_rendering)
        

    def forward(self, b_shape, body_pose, global_orient, transl = None, ICON_compatible_rndring_sub=[0.0, 0.0, 0.0], pose2rot: bool = True):

        # pass

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(b_shape.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if b_shape.shape[0] != batch_size:
            num_repeats = int(batch_size / b_shape.shape[0])
            b_shape = b_shape.expand(num_repeats, -1)

        vertices, joints = lbs(b_shape, full_pose, self.smpl.v_template,
                               self.smpl.shapedirs, self.smpl.posedirs,
                               self.smpl.J_regressor, self.smpl.parents,
                               self.smpl.lbs_weights, pose2rot=pose2rot, cano_shape=self.cano_shape)

        # joints = self.vertex_joint_selector(vertices, joints)
        # # Map the joints to the current dataset
        # if self.joint_mapper is not None:
        #     joints = self.joint_mapper(joints)

        if transl is not None: # if apply_trans:
            # ipdb.set_trace()
            # joints += transl.unsqueeze(dim=1)
            # vertices += transl.unsqueeze(dim=1)
            return (vertices + transl.unsqueeze(dim=1)) * self.ICON_compatible_rendering - ICON_compatible_rndring_sub

        # output = SMPLOutput(vertices=vertices if return_verts else None,
        #                     global_orient=global_orient,
        #                     body_pose=body_pose,
        #                     joints=joints,
        #                     betas=b_shapes,
        #                     full_pose=full_pose if return_full_pose else None)

        return vertices



@persistence.persistent_class
class displacement_Layer(nn.Module):
    def __init__(self, img_size, unique_v2p_mapper_path, **render_kwargs):
        super().__init__()

        v2p_mapper = np.load(unique_v2p_mapper_path, allow_pickle=True)
        unique_v2p_mapper = self.get_unique_v2p(v2p_mapper).astype(np.float32)

        xx = 2. * unique_v2p_mapper[:, 0] / img_size - 1.
        yy = 2. * unique_v2p_mapper[:, 1] /img_size - 1.

        x_y = np.vstack((yy, xx)).T

        self.register_buffer('x_y', torch.from_numpy(x_y).contiguous().unsqueeze(0).unsqueeze(0))

    def get_unique_v2p(self, vertex2pixel, is_random = True):
        vertex2pixel_unique= []
        # is_random = False
        for i, v2p in enumerate(vertex2pixel):
            if not is_random:
                vertex2pixel_unique.append(v2p[0])
            else:
                rand_idx = np.random.randint(len(v2p))
                vertex2pixel_unique.append(v2p[rand_idx])

        vertex2pixel_unique = np.array(vertex2pixel_unique)
        return vertex2pixel_unique

    def forward(self, disp_map):

        torch_displacements = F.grid_sample(disp_map, self.x_y.repeat(disp_map.shape[0], 1, 1, 1), mode='bilinear')

        return torch_displacements.squeeze(2).permute(0,2,1)


# if __name__ == '__main__':
#     device = 'cuda'

#     mesh_1 = trimesh.load('/is/cluster/scratch/ssanyal/project_4/data/stylegan3/CAPE/03375/meshes/blazerlong_babysit_trial1.000001.obj', process=False)
#     mesh_2 = trimesh.load('/is/cluster/scratch/ssanyal/project_4/data/stylegan3/CAPE/03375/meshes/blazerlong_babysit_trial2.000006.obj', process=False)

#     mesh_verts = torch.concat((torch.from_numpy(mesh_1.vertices).unsqueeze(0), torch.from_numpy(mesh_2.vertices).unsqueeze(0)), 0).type(torch.float32).to(device)
#     mesh_faces = torch.concat((torch.from_numpy(mesh_1.faces).unsqueeze(0), torch.from_numpy(mesh_2.faces).unsqueeze(0)), 0).to(device)

#     renderer = NormalRender()

#     normal_images, pixel_normals = renderer(mesh_verts, mesh_faces)

#     plt.imsave('/is/cluster/scratch/ssanyal/project_4/data/stylegan3/CAPE/03375/pytorch3d_render/mesh_1_normal_ones_K4..png', normal_images[0].cpu().numpy())


# # def map_dispmap_2_vertex(self, disp_map):
# #     disp_map = self.dis
# #     # v2p_mapper = np.load('./smpl_uv2barycentric/vertex2pixel_512.npy', allow_pickle=True)
# #     # unique_v2p_mapper = self.get_unique_v2p(v2p_mapper, False)
# #     disp_v = np.array([disp_map[i,j] for i,j in self.unique_v2p_mapper.astype(np.int32)])
# #     return disp_v