# Modified from StyleGAN3 codebase

"""Loss functions."""

import numpy as np
import torch
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import scipy.sparse as sp
import numpy as np
import copy
import ipdb
import matplotlib.pyplot as plt
from torch_utils import misc

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, D_glob=None, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, 
                    pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                    edge_loss=1.0, normal_loss=0.01, laplacian_loss=0.01, mesh_smooth_w=1.0, gen_grad_pen=False):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_glob             = D_glob
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.gen_grad_penalty   = gen_grad_pen
        self.mesh_smoothness_loss_weights = {"edge": edge_loss,
                "normal": normal_loss,
                "laplacian": laplacian_loss,
                "mesh_smooth_w": mesh_smooth_w,
            }
        self.glob_discrim_gain = 1.0
        self.vert_disp_reg_loss = torch.nn.MSELoss()
        # smpl_faces = np.load('./../data/smpl_faces.npy')
        if self.D.patch_d:
            x_ = torch.linspace(-1,1,self.G.img_resolution).type(torch.float32)
            grid = torch.meshgrid(x_, x_,indexing='ij')
            grid_torch = torch.stack(grid).unsqueeze(0).permute(0,3,2,1)
            self.grid_torch_smples = []

            # 512x512
            if self.G.img_resolution == 512:
                self.grid_torch_smples.append(grid_torch[:, :64, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 64:128, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 128:192, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 192:256, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 256:320, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 320:384, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 384:448, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 448:, 96:160, :])

            
            else: # 256x256
                self.grid_torch_smples.append(grid_torch[:, :64, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 64:128, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 128:192, 96:160, :])
                self.grid_torch_smples.append(grid_torch[:, 192:, 96:160, :])
                # grid_torch_smples.append(grid_torch[:, 64:128, 32:96, :])
                # grid_torch_smples.append(grid_torch[:, 64:128, 160:224, :])

                # self.grid_torch_smples.append(grid_torch[:, :32, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, :32, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 32:64, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 32:64, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 64:96, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 64:96, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 96:128, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 96:128, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 128:160, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 128:160, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 160:192, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 160:192, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 192:224, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 192:224, 128:160, :])

                # self.grid_torch_smples.append(grid_torch[:, 224:, 96:128, :])
                # self.grid_torch_smples.append(grid_torch[:, 224:, 128:160, :])


        # # self.relative_Edge_loss = EdgeLoss(self.G.head_verts_mask.shape[0], smpl_faces)
        # vertices_per_edge = self.get_vertices_per_edge(self.G.head_verts_mask.shape[0], smpl_faces)
        # self.edges_for = lambda x: x[:, vertices_per_edge[:, 0], :] - x[:, vertices_per_edge[:, 1], :]
        # ipdb.set_trace()

    @staticmethod
    def get_vert_connectivity(num_vertices, faces):
        """
        Returns a sparse matrix (of size #verts x #verts) where each nonzero
        element indicates a neighborhood relation. For example, if there is a
        nonzero element in position (15,12), that means vertex 15 is connected
        by an edge to vertex 12.
        Adapted from https://github.com/mattloper/opendr/
        """

        vpv = sp.csc_matrix((num_vertices,num_vertices))
        # for each column in the faces...
        for i in range(3):
            IS = faces[:,i]
            JS = faces[:,(i+1)%3]
            data = np.ones(len(IS))
            ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
            mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
            vpv = vpv + mtx + mtx.T
        return vpv

    @staticmethod
    def get_vertices_per_edge(num_vertices, faces):
        """
        Returns an Ex2 array of adjacencies between vertices, where
        each element in the array is a vertex index. Each edge is included
        only once. If output of get_faces_per_edge is provided, this is used to
        avoid call to get_vert_connectivity()
        Adapted from https://github.com/mattloper/opendr/
        """

        vc = sp.coo_matrix(StyleGAN2Loss.get_vert_connectivity(num_vertices, faces))
        result = np.hstack((col(vc.row), col(vc.col)))
        result = result[result[:,0] < result[:,1]] # for uniqueness
        return result

    # # def run_G(self, z, c, update_emas=False):
    # #     ws = self.G.mapping(z, c, update_emas=update_emas)
    # #     if self.style_mixing_prob > 0:
    # #         with torch.autograd.profiler.record_function('style_mixing'):
    # #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
    # #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
    # #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
    # #     img = self.G.synthesis(ws, update_emas=update_emas)
    # #     return img, ws

    def run_G(self, z, c, body_shape, body_pose, body_cam, G_geometry=None, update_emas=False):
        if self.G.conformnet:

            ws = self.G.mapping(z, c, update_emas=update_emas)
            if c.shape[1]==6:
                ws_geo = G_geometry.mapping(z, torch.cat((c, body_pose[:,3:66]),1), truncation_psi=1)
            else:
                ws_geo = G_geometry.mapping(z, torch.cat((c[:, :6], body_pose[:,3:66]),1), truncation_psi=1)

            ## Texture Network
            misc.assert_shape(ws, [None, self.G.synthesis.num_ws, self.G.synthesis.w_dim])
            ws = ws.to(torch.float32).unbind(dim=1)
            ws_geo = ws_geo.to(torch.float32).unbind(dim=1)
            # ipdb.set_trace()
            # Execute layers.
            x = self.G.synthesis.input(ws[0])
            x_geo = G_geometry.synthesis.input(ws_geo[0])
            for name, w, w_geo in zip(self.G.synthesis.layer_names, ws[1:], ws_geo[1:]):
                x = x + x_geo
                x_geo = getattr(G_geometry.synthesis, name)(x_geo, w_geo, update_emas=False)
                x = getattr(self.G.synthesis, name)(x, w, update_emas=update_emas)
                # print('x->',x.shape)
                # print('x_geo->',x.shape)
            if self.G.synthesis.output_scale != 1:
                x = x * self.G.synthesis.output_scale

            if G_geometry.synthesis.output_scale != 1:
                x_geo = x_geo * G_geometry.synthesis.output_scale

            # Ensure correct shape and dtype.
            misc.assert_shape(x_geo, [None, G_geometry.synthesis.img_channels, G_geometry.synthesis.img_resolution, G_geometry.synthesis.img_resolution])
            UV_geo = x_geo.to(torch.float32)

            misc.assert_shape(x, [None, self.G.synthesis.img_channels, self.G.synthesis.img_resolution, self.G.synthesis.img_resolution])
            UV_tex = x.to(torch.float32)


            disp_img_geo = (UV_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
            vert_disps = G_geometry.displacement_Layer(disp_img_geo)

            disp_img_out = UV_tex * 1.0

            clothed_body_shape = body_shape + vert_disps

            # idx_rand = torch.randperm(body_pose.shape[0]).to(body_pose.device)
            # posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[idx_rand, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))

            posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))

            img, meshes = self.G.TextureRender(posed_body, body_cam=body_cam, text_img=UV_tex.permute(0,2,3,1))

            # with torch.no_grad():
            norm_img, _ = self.G.Normalrender(posed_body, body_cam=body_cam)
            if self.G.guass_blur_normals:
                norm_img = self.G.gauss_blur(norm_img)

            # # Combine alpha from geometry and color from texture
            # norm_img_alpha = norm_img[:, :, :, 3:]
            # norm_img_alpha[norm_img_alpha>0.] = 1.
            # # norm_img_alpha[norm_img_alpha<=0.5] = 0.
            # img_new = torch.cat((img[:, :, :, :3], norm_img_alpha),-1)

            return img.permute(0,3,1,2)[:, :3, :, :], ws, meshes, vert_disps, clothed_body_shape, norm_img.permute(0,3,1,2)[:, :3, :, :]#.detach()
            # return img_new.permute(0,3,1,2), ws, meshes, vert_disps, clothed_body_shape, norm_img.permute(0,3,1,2)[:, :3, :, :]#.detach()


        else:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            # ipdb.set_trace()
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            disp_img = self.G.synthesis(ws, update_emas=update_emas)
            # ipdb.set_trace()

            if self.G.seperate_disp_map:
                if not self.G.sep_disp_map_sampling:
                    disp_img = disp_img * self.G.smpl_uv_parts.unsqueeze(0).permute(0,3,1,2)
                if self.G.only_disp_img:
                    disp_img = disp_img[:, :3, :, :] + disp_img[:, 3:6, :, :] + disp_img[:, 6:9, :, :] + disp_img[:,9:12, :, :] \
                                    + disp_img[:, 12:15, :, :] + disp_img[:, 15:18, :, :] + disp_img[:, 18:21, :, :] + disp_img[:, 21:24, :, :] \
                                    + disp_img[:, 24:27, :, :] + disp_img[:, 27:, :, :]

            if self.G.mask_disp_map:
                disp_img = disp_img * (self.G.smpl_uv_mask.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1) \
                    + (self.G.smpl_uv_mask2.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1)
            if self.G.only_disp_img:
                return disp_img, ws, 0, 0, 0, 0
            # vert_disps = self.G.displacement_Layer(disp_img)
            # clothed_body_shape = body_shape + vert_disps
            # # posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:], body_pose[:, :3])
            # img = self.G.Normalrender(clothed_body_shape).permute(0,3,1,2)
            # # print("run_G->", img.max(), img.dtype)
            # ipdb.set_trace()
            if self.G.texture_render:
                if c.shape[1] == G_geometry.c_dim:
                    ws_geo = G_geometry.mapping(z, c, truncation_psi=1)
                else:
                    ws_geo = G_geometry.mapping(z, torch.cat((c, body_pose[:,3:66]),1), truncation_psi=1)
                # ws_geo = G_geometry.mapping(z, c, truncation_psi=1)
                disp_img_geo = G_geometry.synthesis(ws_geo, noise_mode='const')
                # disp_img_geo, _ = G_geometry(z, c, body_shape, body_pose, body_cam, truncation_psi=1, noise_mode='const')
                disp_img_geo = (disp_img_geo * 0.5 + 0.5) * 2 * 0.071 - 0.071
                vert_disps_all = G_geometry.displacement_Layer(disp_img_geo)
                vert_disps = vert_disps_all
                # ipdb.set_trace()
            else:
                if self.G.resume_pretrain_cape:
                    disp_img = (disp_img * 0.5 + 0.5) * 2 * 0.071 - 0.071
                    # disp_img.requires_grad = True
                    # disp_img.retain_grad()
                    vert_disps_all = self.G.displacement_Layer(disp_img)
                    # vert_disps = vert_disps_all
                    vert_disps_all = vert_disps_all * self.G.head_verts_mask[:, None]
                    if self.G.img_channels == 30:
                        if self.G.sep_disp_map_sampling:
                            # ipdb.set_trace()
                            vert_disps = torch.index_select(vert_disps_all.reshape(body_shape.shape[0],-1), -1, self.G.parts_indexing_repeate_index.view(-1)).reshape(body_shape.shape[0], 6890,-1)
                            # ipdb.set_trace()
                        else:
                            vert_disps = vert_disps_all[:, :, :3] + vert_disps_all[:, :, 3:6] + vert_disps_all[:, :, 6:9] + vert_disps_all[:, :, 9:12] \
                                            + vert_disps_all[:, :, 12:15] + vert_disps_all[:, :, 15:18] + vert_disps_all[:, :, 18:21] + vert_disps_all[:, :, 21:24] \
                                            + vert_disps_all[:, :, 24:27] + vert_disps_all[:, :, 27:]
                    else:
                        vert_disps = vert_disps_all
                else:

                    if self.G.img_channels == 3:
                        vert_disps_all = self.G.displacement_Layer(disp_img) #self.G.threshold_layer(self.G.displacement_Layer(disp_img)) 
                        # print(vert_disps_all.max(), vert_disps_all.min())
                        vert_disps_all = vert_disps_all * self.G.head_verts_mask[:, None]
                        vert_disps = vert_disps_all * self.G.disp_scale
                        # print(vert_disps.max(), vert_disps.min())
                        # # vert_disps = self.G.threshold_layer(self.G.displacement_Layer(disp_img)) * self.G.disp_scale #displacement_Layer(disp_img)
                        # # vert_disps = self.G.threshold_layer(self.G.displacement_Layer(disp_img) * self.G.head_verts_mask[:, None]) * self.G.disp_scale #displacement_Layer(disp_img)
                    elif self.G.img_channels == 9:
                        vert_disps_all = self.G.threshold_layer(self.G.displacement_Layer(disp_img))
                        vert_disps_all = vert_disps_all * self.G.head_verts_mask[:, None]
                        vert_disps = vert_disps_all[:, :, :3] * self.G.disp_scale + vert_disps_all[:, :, 3:6] * self.G.disp_scale * 0.5 \
                                        + vert_disps_all[:, :, 6:] * self.G.disp_scale * 0.25
                        # vert_disps = vert_disps * self.head_verts_mask[:, None]

                    elif self.G.img_channels == 30:
                        # ipdb.set_trace()
                        vert_disps_all = self.G.displacement_Layer(disp_img)
                        # ipdb.set_trace()
                        vert_disps_all = vert_disps_all * self.G.head_verts_mask[:, None] * self.G.disp_scale
                        if self.G.sep_disp_map_sampling:
                            # ipdb.set_trace()
                            vert_disps = torch.index_select(vert_disps_all.reshape(body_shape.shape[0],-1), -1, self.G.parts_indexing_repeate_index.view(-1)).reshape(body_shape.shape[0], 6890,-1)
                            # ipdb.set_trace()
                        else:
                            vert_disps = vert_disps_all[:, :, :3] + vert_disps_all[:, :, 3:6] + vert_disps_all[:, :, 6:9] + vert_disps_all[:, :, 9:12] \
                                            + vert_disps_all[:, :, 12:15] + vert_disps_all[:, :, 15:18] + vert_disps_all[:, :, 18:21] + vert_disps_all[:, :, 21:24] \
                                            + vert_disps_all[:, :, 24:27] + vert_disps_all[:, :, 27:] 

                if self.G.spiral_conv:
                    vert_disps += self.G.spiral_conv_layer_1(vert_disps)
                    vert_disps += self.G.spiral_conv_layer_2(vert_disps)

            clothed_body_shape = body_shape + vert_disps#.unsqueeze(0).repeat(body_shape.shape[0], 1, 1)
            # posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:])
            posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device))
            


            # idx_rand = torch.randperm(body_pose.shape[0]).to(body_pose.device)

            # # posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[idx_rand, :3]) #self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3])
            # posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[idx_rand, :3], body_pose[:, 72:], ICON_compatible_rndring_sub=torch.tensor([0.0, 0.3, 0.0]).type(body_shape.dtype).to(body_shape.device)) #self.G.smpl_body(clothed_body_shape, body_pose[:, 3:72], body_pose[:, :3])
            if self.G.texture_render:
                img, meshes = self.G.TextureRender(posed_body, body_cam=body_cam, text_img=disp_img.permute(0,2,3,1))
                # img = img * 2.0 - 1.0
            else:
                img, meshes = self.G.Normalrender(posed_body, body_cam=body_cam) # self.G.Normalrender(clothed_body_shape)
            # ipdb.set_trace()
            # return img.permute(0,3,1,2)[:, :3, :, :], ws, meshes, (self.G.displacement_Layer(disp_img) * self.G.head_verts_mask[:, None]), clothed_body_shape 
            # return img.permute(0,3,1,2), ws, meshes, vert_disps, clothed_body_shape 
            # return img.permute(0,3,1,2), ws, meshes, (self.G.displacement_Layer(disp_img) * self.G.head_verts_mask[:, None]), clothed_body_shape
            # return img.permute(0,3,1,2), ws, meshes, self.G.displacement_Layer(disp_img), clothed_body_shape
            # return img.permute(0,3,1,2), ws, meshes, vert_disps_all, clothed_body_shape
            return img.permute(0,3,1,2)[:, :3, :, :], ws, meshes, vert_disps_all, clothed_body_shape, disp_img 

    # def run_G(self, z, c, body_shape, body_pose, body_cam, update_emas=False):
    #     # ws = self.G.mapping(z, c, update_emas=update_emas)
    #     # ipdb.set_trace()
    #     # if self.style_mixing_prob > 0:
    #     #     with torch.autograd.profiler.record_function('style_mixing'):
    #     #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
    #     #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
    #     #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
    #     # disp_img = self.G.synthesis(ws, update_emas=update_emas)
    #     # ipdb.set_trace()
    #     # if self.G.mask_disp_map:
    #     #     disp_img = disp_img * (self.G.smpl_uv_mask.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1) \
    #     #         + (self.G.smpl_uv_mask2.unsqueeze(0).repeat(disp_img.shape[1], 1, 1)).unsqueeze(0).repeat(disp_img.shape[0], 1, 1, 1)
        
    #     vert_disps = self.G.threshold_layer(self.G.disp_verts) * 0.1 # #displacement_Layer(disp_img)
    #     # vert_disps = self.G.threshold_layer(self.G.disp_verts * self.G.head_verts_mask[:, None]) * 0.1 # #displacement_Layer(disp_img)
    #     clothed_body_shape = body_shape + vert_disps.unsqueeze(0).repeat(body_shape.shape[0], 1, 1)
    #     posed_body = self.G.smpl_body(clothed_body_shape, body_pose[:, 3:], body_pose[:, :3])
    #     img, meshes = self.G.Normalrender(posed_body, body_cam=body_cam) # self.G.Normalrender(clothed_body_shape)
    #     # print("run_G->", img.max(), img.dtype)
    #     # ipdb.set_trace()
    #     # return img.permute(0,3,1,2)[:, :3, :, :], posed_body, meshes#, ws, (self.G.disp_verts * self.G.head_verts_mask[:, None])
    #     return img.permute(0,3,1,2), posed_body, meshes, (self.G.disp_verts * self.G.head_verts_mask[:, None])

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        # ipdb.set_trace()
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        # ipdb.set_trace()
        logits = self.D(img, c, update_emas=update_emas)
        return logits
    
    def run_D_glob(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        # ipdb.set_trace()
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        # ipdb.set_trace()
        logits = self.D_glob(img, c, update_emas=update_emas)
        return logits

    def update_mesh_shape_prior_losses(self, mesh, disp_verts, body_shape, clothed_body_shape):

        remaining_verts = (~self.G.head_verts_mask.type(torch.bool)).type(self.G.head_verts_mask.dtype)

        if self.G.img_channels == 3 or self.G.img_channels == 30:

            mesh_smoothness_loss = mesh_edge_loss(mesh) * self.mesh_smoothness_loss_weights['edge'] + \
                            mesh_normal_consistency(mesh) * self.mesh_smoothness_loss_weights['normal'] + \
                            mesh_laplacian_smoothing(mesh, method="uniform") * self.mesh_smoothness_loss_weights['laplacian']

            # # # mesh_smoothness_loss = mesh_smoothness_loss + ((disp_verts ** 2).mean()) * 0.01

            # # # remaining_verts = (~self.G.head_verts_mask.type(torch.bool)).type(self.G.head_verts_mask.dtype)
            # # mesh_smoothness_loss = mesh_smoothness_loss + (((disp_verts * self.G.head_verts_mask[:, None]) ** 2).mean()) * 10.0 + (((disp_verts * remaining_verts[:, None]) ** 2).mean()) * 100.0 #0.01
            # # mesh_smoothness_loss = (((disp_verts * self.G.head_verts_mask[:, None]) ** 2).mean()) * 10.0 + (((disp_verts * remaining_verts[:, None]) ** 2).mean()) * 100 #0.01
            # mesh_smoothness_loss = (((disp_verts * remaining_verts[:, None]) ** 2).mean()) * 10 #0.01
            # # # ipdb.set_trace()
            # edges1 = self.edges_for(body_shape)
            # edges2 = self.edges_for(clothed_body_shape)

            # relative_edge_loss = self.vert_disp_reg_loss(edges1, edges2)

            # mesh_smoothness_loss =  relative_edge_loss
            # # mesh_smoothness_loss = mesh_smoothness_loss + relative_edge_loss

            # # ipdb.set_trace()
        
        elif self.G.img_channels == 9:
            # ipdb.set_trace()
            mesh_smoothness_loss  = (((disp_verts * remaining_verts[:, None]) ** 2).mean()) * 100
            mesh_smoothness_loss  = mesh_smoothness_loss + (((disp_verts[:, :, :3] * self.G.head_verts_mask[:, None]) ** 2).mean()) * 1.0 \
                + (((disp_verts[:, :, 3:6] * self.G.head_verts_mask[:, None]) ** 2).mean()) * 0.1 \
                + (((disp_verts[:, :, 6:] * self.G.head_verts_mask[:, None]) ** 2).mean()) * 0.001

            edges1 = self.edges_for(body_shape)
            edges2 = self.edges_for(clothed_body_shape)

            relative_edge_loss = self.vert_disp_reg_loss(edges1, edges2)

            mesh_smoothness_loss = mesh_smoothness_loss + relative_edge_loss
            # ipdb.set_trace()
        return mesh_smoothness_loss


    def accumulate_gradients(self, phase, real_img, real_c, body_shape, body_pose, body_cam, gen_z, gen_c, gain, cur_nimg, cur_tick, rank, G_geometry=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_globmain', 'D_globreg', 'D_globboth']
        if self.pl_weight == 0:
            if self.mesh_smoothness_loss_weights['mesh_smooth_w'] == 0:
                phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
            phase = {'D_globreg': 'none', 'D_globboth': 'D_globmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        if self.D.patch_d:
            random_patch_no = torch.randint(0,len(self.grid_torch_smples),(1,))

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                if self.G.texture_render:
                    # ipdb.set_trace()
                    gen_img, _gen_ws, meshes, disp_verts, clothed_body_shape, disp_img_to_grad = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, G_geometry)
                else:
                    gen_img, _gen_ws, meshes, disp_verts, clothed_body_shape, disp_img_to_grad = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam)
                    # gen_img, meshes = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam)
                # ipdb.set_trace()
                # print("gen_img->", gen_img.max(), gen_img.min(), gen_img[0, :5, :5, :])

                gen_img_patch = torch.nn.functional.grid_sample(gen_img, self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                gen_logits = self.run_D(gen_img_patch, gen_c, blur_sigma=blur_sigma)

                gen_logits_glob = self.run_D_glob(gen_img, gen_c, blur_sigma=blur_sigma)
                


                # if self.D.img_channels == 6:
                #     if self.D.patch_d:
                #         gen_img_patch = torch.nn.functional.grid_sample(torch.cat((gen_img, disp_img_to_grad),1), self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                #         gen_logits, gen_logits_glob = self.run_D(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                #     else:
                #         gen_logits = self.run_D(torch.cat((gen_img, disp_img_to_grad),1), gen_c, blur_sigma=blur_sigma)
                # else:
                #     if self.D.patch_d:
                #         gen_img_patch = torch.nn.functional.grid_sample(gen_img, self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                #         gen_logits, gen_logits_glob = self.run_D(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                #     else:
                #         gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)


                if (cur_tick % 80 == 0) and (cur_tick > 390):
                    self.glob_discrim_gain = self.glob_discrim_gain * 1.0
                    # print("global discrim weight reduced to: ", self.glob_discrim_gain)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + self.glob_discrim_gain * torch.nn.functional.softplus(-gen_logits_glob) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                # mesh_smoothness_loss = self.update_mesh_shape_prior_losses(meshes, disp_verts, body_shape, clothed_body_shape)
                # training_stats.report('Loss/G/mesh_smoothness_loss', mesh_smoothness_loss)
                if self.G.only_disp_img:
                    tot_gen_loss = loss_Gmain
                else:
                    tot_gen_loss = loss_Gmain #+ (mesh_smoothness_loss * self.mesh_smoothness_loss_weights['mesh_smooth_w'])




                # generatedpatch_to_pose_grad = torch.autograd.grad(outputs=[gen_img_patch.sum()], inputs=[body_pose], create_graph=True, only_inputs=True)[0]
                # generatedpatch_to_pose_penalty = generatedpatch_to_pose_grad.square().sum()
                # training_stats.report('Loss/G/disentanglementloss', 0.01 * generatedpatch_to_pose_penalty)
                # tot_gen_loss = loss_Gmain + 0.01 * generatedpatch_to_pose_penalty


                training_stats.report('Loss/G/tot_gen_loss', tot_gen_loss)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                # loss_Gmain.mean().mul(gain).backward()
                tot_gen_loss.mean().mul(gain).backward()
            # # ipdb.set_trace()
            # if (rank==0) and (cur_tick%10 == 0):
            #     disp_grad = (disp_img_to_grad.grad - disp_img_to_grad.grad[0].min())/(disp_img_to_grad.grad[0].max() - disp_img_to_grad.grad[0].min())
            #     plt.imsave('/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/cape_multisub/lr_check/posed/withaOutlpha/masked_head/resumed/grads1/' + str(cur_tick) + '_1.png' ,disp_grad.permute(0,2,3,1).detach().cpu().numpy()[0]) 
            #     disp_grad = (disp_img_to_grad.grad - disp_img_to_grad.grad[1].min())/(disp_img_to_grad.grad[1].max() - disp_img_to_grad.grad[1].min())
            #     plt.imsave('/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/cape_multisub/lr_check/posed/withaOutlpha/masked_head/resumed/grads1/' + str(cur_tick) + '_2.png' ,disp_grad.permute(0,2,3,1).detach().cpu().numpy()[1])
            #     disp_grad = (disp_img_to_grad.grad - disp_img_to_grad.grad[-1].min())/(disp_img_to_grad.grad[-1].max() - disp_img_to_grad.grad[-1].min())
            #     plt.imsave('/is/cluster/fast/ssanyal/project_4/stylegan3/optimising_dynamic_displacements/cape_multisub/lr_check/posed/withaOutlpha/masked_head/resumed/grads1/' + str(cur_tick) + '_3.png' ,disp_grad.permute(0,2,3,1).detach().cpu().numpy()[-1])               
            if self.gen_grad_penalty:
                # ipdb.set_trace()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 0.01)

        #     # if phase in ['Gmain', 'Gboth']:
        #     #     with torch.autograd.profiler.record_function('Gmeshl_forward'):
        #     #         gen_img, _gen_ws = self.run_G(gen_z, gen_c, body_shape, body_pose)
        #     #         # # ipdb.set_trace()
        #     #         # # print("gen_img->", gen_img.max(), gen_img.min(), gen_img[0, :5, :5, :])
        #     #         # gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
        #     #         # training_stats.report('Loss/scores/fake', gen_logits)
        #     #         # training_stats.report('Loss/signs/fake', gen_logits.sign())
        #     #         # loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
        #     #         # training_stats.report('Loss/G/loss', loss_Gmain)
        #     #         mesh_smoothness_loss = self.update_mesh_shape_prior_losses(_gen_ws)
        #     #         training_stats.report('Loss/G/mesh_smoothness_loss', mesh_smoothness_loss)
        #     #         tot_gen_loss = (mesh_smoothness_loss * self.mesh_smoothness_loss_weights['mesh_smooth_w'])
        #     #         # training_stats.report('Loss/G/tot_gen_loss', tot_gen_loss)
        #     #     with torch.autograd.profiler.record_function('Gmeshl_backward'):
        #     #         # loss_Gmain.mean().mul(gain).backward()
        #     #         tot_gen_loss.mean().mul(gain).backward()
        #     #     if self.gen_grad_penalty:
        #     #         # ipdb.set_trace()
        #     #         torch.nn.utils.clip_grad_norm_(self.G.parameters(), 0.01)

        if not self.G.texture_render:
            if phase in ['Greg', 'Gboth']:
                # ipdb.set_trace()
                with torch.autograd.profiler.record_function('Gsmoothness_forward'):
                    gen_img, _gen_ws, meshes, disp_verts, clothed_body_shape, _ = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, G_geometry)
                    mesh_smoothness_loss = self.update_mesh_shape_prior_losses(meshes, disp_verts, body_shape, clothed_body_shape)
                    training_stats.report('Loss/G/mesh_smoothness_loss', mesh_smoothness_loss)

                with torch.autograd.profiler.record_function('Gsmoothness_backward'):
                    mesh_smoothness_loss.mean().mul(self.mesh_smoothness_loss_weights['mesh_smooth_w'] * gain).backward()

        # ipdb.set_trace()
        # # Gpl: Apply path length regularization.
        # if phase in ['Greg', 'Gboth']:
        #     ipdb.set_trace()
        #     with torch.autograd.profiler.record_function('Gpl_forward'):
        #         batch_size = gen_z.shape[0] // self.pl_batch_shrink
        #         gen_img, gen_ws, _meshes, _, _ = self.run_G(gen_z[:batch_size], gen_c[:batch_size], body_shape[:batch_size], body_pose[:batch_size], body_cam[:batch_size])
        #         pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        #         with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
        #             pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        #         pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #         pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #         self.pl_mean.copy_(pl_mean.detach())
        #         pl_penalty = (pl_lengths - pl_mean).square()
        #         training_stats.report('Loss/pl_penalty', pl_penalty)
        #         loss_Gpl = pl_penalty * self.pl_weight
        #         training_stats.report('Loss/G/reg', loss_Gpl)
        #     with torch.autograd.profiler.record_function('Gpl_backward'):
        #         loss_Gpl.mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                if self.G.texture_render:
                    gen_img, _gen_ws, _, _dvs, _, norm_img = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, G_geometry, update_emas=True)
                else:
                    gen_img, _gen_ws, _, _dvs, _, _ = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                # gen_img, _gen_ws = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                if self.D.img_channels == 6:
                    if self.D.patch_d:
                        gen_img_patch = torch.nn.functional.grid_sample(torch.cat((gen_img, norm_img), 1), self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                        gen_logits = self.run_D(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                    else:
                        gen_logits = self.run_D(torch.cat((gen_img, norm_img), 1), gen_c, blur_sigma=blur_sigma, update_emas=True)
                    # ipdb.set_trace()
                else:
                    if self.D.patch_d:
                        gen_img_patch = torch.nn.functional.grid_sample(gen_img, self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                        gen_logits = self.run_D(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                    else:
                        gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                # ipdb.set_trace()
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                if self.D.img_channels == 6:
                    if self.G.guass_blur_normals:
                        real_img[:,3:,:,:] = self.G.gauss_blur(real_img[:,3:,:,:])
                    real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                else:
                    real_img_tmp = real_img[:, :3, :, :].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    # real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                # if self.G.mask_disp_map:
                #     real_img_tmp = real_img_tmp * (self.G.smpl_uv_mask.unsqueeze(0).repeat(real_img_tmp.shape[1], 1, 1)).unsqueeze(0).repeat(real_img_tmp.shape[0], 1, 1, 1) \
                #                 + (self.G.smpl_uv_mask2.unsqueeze(0).repeat(real_img_tmp.shape[1], 1, 1)).unsqueeze(0).repeat(real_img_tmp.shape[0], 1, 1, 1)
                if self.D.patch_d:
                    real_img_patch = torch.nn.functional.grid_sample(real_img_tmp, self.grid_torch_smples[random_patch_no].to(real_img_tmp.device).repeat(real_img_tmp.shape[0],1,1,1))
                    real_logits = self.run_D(real_img_patch, real_c, blur_sigma=blur_sigma)
                else:
                    real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                # gen_img, _gen_ws, _, _dvs, _ = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                # real_gen = copy.deepcopy(gen_img.detach())
                # real_gen.requires_grad = True
                # real_logits = self.run_D(real_gen, real_c, blur_sigma=blur_sigma)
                # ipdb.set_trace()
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        if self.D.patch_d:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_patch], create_graph=True, only_inputs=True)[0]
                        else:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_gen], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['D_globmain', 'D_globboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                if self.G.texture_render:
                    gen_img, _gen_ws, _, _dvs, _, norm_img = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, G_geometry, update_emas=True)
                else:
                    gen_img, _gen_ws, _, _dvs, _, _ = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                # gen_img, _gen_ws = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                if self.D_glob.img_channels == 6:
                    if self.D_glob.patch_d:
                        gen_img_patch = torch.nn.functional.grid_sample(torch.cat((gen_img, norm_img), 1), self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                        gen_logits = self.run_D_glob(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                    else:
                        gen_logits = self.run_D_glob(torch.cat((gen_img, norm_img), 1), gen_c, blur_sigma=blur_sigma, update_emas=True)
                    # ipdb.set_trace()
                else:
                    if self.D_glob.patch_d:
                        gen_img_patch = torch.nn.functional.grid_sample(gen_img, self.grid_torch_smples[random_patch_no].to(gen_img.device).repeat(gen_img.shape[0],1,1,1))
                        gen_logits = self.run_D_glob(gen_img_patch, gen_c, blur_sigma=blur_sigma)
                    else:
                        gen_logits = self.run_D_glob(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake_glob', gen_logits)
                training_stats.report('Loss/signs/fake_glob', gen_logits.sign())
                # ipdb.set_trace()
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_glob_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['D_globmain', 'D_globreg', 'D_globboth']:
            name = 'Dreal_glob' if phase == 'D_globmain' else 'Dr1_glob' if phase == 'D_globreg' else 'Dreal_Dr1_glob'
            with torch.autograd.profiler.record_function(name + '_forward'):
                if self.D_glob.img_channels == 6:
                    if self.G.guass_blur_normals:
                        real_img[:,3:,:,:] = self.G.gauss_blur(real_img[:,3:,:,:])
                    real_img_tmp = real_img.detach().requires_grad_(phase in ['D_globreg', 'D_globboth'])
                else:
                    real_img_tmp = real_img[:, :3, :, :].detach().requires_grad_(phase in ['D_globreg', 'D_globboth'])
                    # real_img_tmp = real_img.detach().requires_grad_(phase in ['D_globreg', 'D_globboth'])
                # if self.G.mask_disp_map:
                #     real_img_tmp = real_img_tmp * (self.G.smpl_uv_mask.unsqueeze(0).repeat(real_img_tmp.shape[1], 1, 1)).unsqueeze(0).repeat(real_img_tmp.shape[0], 1, 1, 1) \
                #                 + (self.G.smpl_uv_mask2.unsqueeze(0).repeat(real_img_tmp.shape[1], 1, 1)).unsqueeze(0).repeat(real_img_tmp.shape[0], 1, 1, 1)
                if self.D_glob.patch_d:
                    real_img_patch = torch.nn.functional.grid_sample(real_img_tmp, self.grid_torch_smples[random_patch_no].to(real_img_tmp.device).repeat(real_img_tmp.shape[0],1,1,1))
                    real_logits = self.run_D_glob(real_img_patch, real_c, blur_sigma=blur_sigma)
                else:
                    real_logits = self.run_D_glob(real_img_tmp, real_c, blur_sigma=blur_sigma)
                # gen_img, _gen_ws, _, _dvs, _ = self.run_G(gen_z, gen_c, body_shape, body_pose, body_cam, update_emas=True)
                # real_gen = copy.deepcopy(gen_img.detach())
                # real_gen.requires_grad = True
                # real_logits = self.run_D(real_gen, real_c, blur_sigma=blur_sigma)
                # ipdb.set_trace()
                training_stats.report('Loss/scores/real_glob', real_logits)
                training_stats.report('Loss/signs/real_glob', real_logits.sign())

                loss_Dreal = 0
                if phase in ['D_globmain', 'D_globboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_glob', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['D_globreg', 'D_globboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        if self.D_glob.patch_d:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_patch], create_graph=True, only_inputs=True)[0]
                        else:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_gen], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) * 10.0
                    training_stats.report('Loss/r1_penalty_glob', r1_penalty)
                    training_stats.report('Loss/D/reg_glob', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
#----------------------------------------------------------------------------


def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

# class EdgeLoss(torch.nn.Module):
#     def __init__(self, num_vertices, faces, vertex_masks=None, mask_weights=None, mesh_sampler=None):
#         super().__init__()
#         vertices_per_edge = EdgeLoss.get_vertices_per_edge(num_vertices, faces)
#         vertex_weights = EdgeLoss.get_vertex_weights(num_vertices, vertex_masks, mask_weights, mesh_sampler)
#         self.edge_weights = (vertex_weights[vertices_per_edge[:, 0]] + vertex_weights[vertices_per_edge[:, 1]]) / 2.
#         self.edges_for = lambda x: x[:, vertices_per_edge[:, 0], :] - x[:, vertices_per_edge[:, 1], :]

#     def forward(self, vertices1, vertices2):
#         """
#         Given two meshes of the same topology, returns the relative edge differences.
#         """

#         batch_size = vertices1.shape[0]
#         device = vertices1.device

#         edge_weights = torch.from_numpy(self.edge_weights).to(vertices1.dtype)
#         edge_weights = edge_weights[None, :, None].repeat(batch_size, 1, 1).to(device)

#         edges1 = torch.multiply(edge_weights, self.edges_for(vertices1))
#         edges2 = torch.multiply(edge_weights, self.edges_for(vertices2))
#         return torch.nn.MSELoss()(edges1, edges2)

#     @staticmethod
#     def get_vert_connectivity(num_vertices, faces):
#         """
#         Returns a sparse matrix (of size #verts x #verts) where each nonzero
#         element indicates a neighborhood relation. For example, if there is a
#         nonzero element in position (15,12), that means vertex 15 is connected
#         by an edge to vertex 12.
#         Adapted from https://github.com/mattloper/opendr/
#         """

#         vpv = sp.csc_matrix((num_vertices,num_vertices))
#         # for each column in the faces...
#         for i in range(3):
#             IS = faces[:,i]
#             JS = faces[:,(i+1)%3]
#             data = np.ones(len(IS))
#             ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
#             mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
#             vpv = vpv + mtx + mtx.T
#         return vpv

#     @staticmethod
#     def get_vertices_per_edge(num_vertices, faces):
#         """
#         Returns an Ex2 array of adjacencies between vertices, where
#         each element in the array is a vertex index. Each edge is included
#         only once. If output of get_faces_per_edge is provided, this is used to
#         avoid call to get_vert_connectivity()
#         Adapted from https://github.com/mattloper/opendr/
#         """

#         vc = sp.coo_matrix(EdgeLoss.get_vert_connectivity(num_vertices, faces))
#         result = np.hstack((col(vc.row), col(vc.col)))
#         result = result[result[:,0] < result[:,1]] # for uniqueness
#         return result

#     @staticmethod
#     def get_vertex_weights(num_vertices, vertex_masks=None, mask_weights=None, mesh_sampler=None):
#         if vertex_masks is None or mask_weights is None:
#             return np.ones(num_vertices)

#         if (vertex_masks['vertex_count'] != num_vertices) and (mesh_sampler is None):
#             raise RuntimeError("Mismatch of vertex counts with the loaded mask: %d != %d" % (num_vertices, vertex_masks['vertex_count']))

#         vertex_weights = np.ones(vertex_masks['vertex_count'])
#         vertex_weights[vertex_masks['face']] = mask_weights['w_edge_face']
#         vertex_weights[vertex_masks['left_ear']] = mask_weights['w_edge_ears']
#         vertex_weights[vertex_masks['right_ear']] = mask_weights['w_edge_ears']
#         vertex_weights[vertex_masks['left_eyeball']] = mask_weights['w_edge_eyeballs']
#         vertex_weights[vertex_masks['right_eyeball']] = mask_weights['w_edge_eyeballs']
#         vertex_weights[vertex_masks['left_eye_region']] = mask_weights['w_edge_eye_region']
#         vertex_weights[vertex_masks['right_eye_region']] = mask_weights['w_edge_eye_region']
#         vertex_weights[vertex_masks['lips']] = mask_weights['w_edge_lips']
#         vertex_weights[vertex_masks['neck']] = mask_weights['w_edge_neck']
#         vertex_weights[vertex_masks['nostrils']] = mask_weights['w_edge_nostrils']
#         vertex_weights[vertex_masks['scalp']] = mask_weights['w_edge_scalp']
#         vertex_weights[vertex_masks['boundary']] = mask_weights['w_edge_boundary']

#         if vertex_masks['vertex_count'] != num_vertices:
#             # Transfer vertex mask to the sampled mesh resolution
#             source_level = mesh_sampler.get_level(vertex_masks['vertex_count'])
#             target_level = mesh_sampler.get_level(num_vertices)

#             num_levels = mesh_sampler.get_number_levels()
#             for _ in range(num_levels):
#                 if source_level == target_level:
#                     break
#                 vertex_weights = mesh_sampler.downsample(vertex_weights)
#                 source_level = mesh_sampler.get_level(vertex_weights.shape[0])
#             vertex_weights = vertex_weights.reshape(-1,)
#             if source_level != target_level:
#                 raise RuntimeError("Unable to downsample mesh to target level")
#         return vertex_weights
