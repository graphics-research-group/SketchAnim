import cv2
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import math
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict as edict
from shapely.geometry import Point, Polygon
import random
import colorsys
from PIL import Image

from DeformationPyramid.model.nets import Deformation_Pyramid
from DeformationPyramid.model.loss import compute_truncated_chamfer_distance

from CDT import GetBoundaryFromSilhouette, GetCDT

CPU = 'cpu'


'''
Mapping sketch skeleton from video skeleton using shape matching. 
'''
def SkeletonMap(moving_silhouette,
                fixed_silhouette,
                moving_skeleton_vertices,
                device=CPU,
                skeleton_edges=None, 
                return_plot=False,
                verbose=False):
    
    # Get boundary of silhouettes
    moving_boundary = GetBoundaryFromSilhouette(moving_silhouette, num_samples=50, max_distance = -1, padding=0)
    fixed_boundary = GetBoundaryFromSilhouette(fixed_silhouette, num_samples=50, max_distance = -1, padding=0)

    if verbose:
        print(f'Moving Boundary Shape: {moving_boundary.shape}')
        print(f'Fixed Boundary Shape: {fixed_boundary.shape}')


    deformed_boundary = get_deformed_boundary(moving_boundary, fixed_boundary, device, moving_silhouette, fixed_silhouette)
    deformed_boundary = np.array(deformed_boundary)

    if verbose:
        print(f'Deformed Boundary Length: {len(deformed_boundary)}')
        print(f'Deformed Boundary Shape: {deformed_boundary.shape}')

    fixed_skeleton_vertices = []

    for point in moving_skeleton_vertices:
        weights = mvc(moving_boundary, point)
        fixed_point = interpolate_point(deformed_boundary, weights)
        fixed_skeleton_vertices.append(fixed_point)

    fixed_skeleton_vertices = np.array(fixed_skeleton_vertices)

    if verbose:
        print(f'Fixed Skeleton Shape: {fixed_skeleton_vertices.shape}')

    if return_plot:
        moving_skeleton_visualise = plot_skeleton_on_silhouette(moving_silhouette, moving_skeleton_vertices, skeleton_edges)
        fixed_skeleton_visualise = plot_skeleton_on_silhouette(fixed_silhouette, fixed_skeleton_vertices, skeleton_edges)

        return fixed_skeleton_vertices, moving_skeleton_visualise, fixed_skeleton_visualise

    return fixed_skeleton_vertices

def plot_skeleton_on_silhouette(silhouette, vertices, edges, output_shape=None):
    silhouette_rgb = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2RGB)
    return plot_skeleton_on_image(silhouette_rgb, vertices, edges, output_shape)

def plot_skeleton_on_image(image, vertices, edges, output_shape=None, point_proportion=0.015, line_proportion=0.005,
                           vertex_colour=(255,0,255), edge_colour=(0,255,0)):
    skeleton_visualise = np.array(image)
    if output_shape is not None:
        skeleton_visualise = cv2.resize(skeleton_visualise, output_shape)
        vertices = np.array(vertices)
        vertices *= np.array(np.array(output_shape)/np.array(image.shape[:2]))
    width = skeleton_visualise.shape[1]
    if edges is not None:
        for bone in edges:
            point0 = (int(vertices[bone[0]][0]), int(vertices[bone[0]][1]))
            point1 = (int(vertices[bone[1]][0]), int(vertices[bone[1]][1]))
            cv2.line(skeleton_visualise, point0, point1, edge_colour, round(width*line_proportion))
    for point in vertices:
        cv2.circle(skeleton_visualise, (int(point[0]), int(point[1])), round(width*point_proportion), (255,0,255), -1)
    return skeleton_visualise

'''
Computes mean value coordinate for the given triangle vertices.
'''
class MVCPoint:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f'({self.x},{self.y})'
    
def mvc(vertices_list, pt):
    vertices = []
    for vertex in vertices_list:
        vertices.append(MVCPoint(vertex[0], vertex[1]))
    vertices = np.asarray(vertices)

    nSize = len(list(vertices))
    bary = np.zeros((nSize))
    p = MVCPoint(pt[0], pt[1])

    s = []
    for i in range(nSize):
        dx = vertices[i].x - p.x
        dy = vertices[i].y - p.y
        s.append(MVCPoint(dx, dy))
    
    epsilon = 1e-6
    ip = 0
    im = 0
    rp, dl, mu = 0, 0, 0
    ri, Ai, Di = [], [], []

    for i in range(nSize):
        ip = (i+1) % nSize
        ri.append(math.sqrt(s[i].x**2 + s[i].y**2))
        Ai.append(0.5*(s[i].x*s[ip].y - s[ip].x*s[i].y))
        Di.append(s[ip].x*s[i].x + s[ip].y*s[i].y)

        if ri[i] <= epsilon:
            bary[i] = 1
            return bary
        if abs(Ai[i]) <= epsilon and Di[i] < 0:
            dx = vertices[ip].x - vertices[i].x
            dy = vertices[ip].y - vertices[i].y
            dl = math.sqrt(dx*dx + dy*dy)
            dx = p.x - vertices[i].x
            dy = p.y - vertices[i].y
            mu = math.sqrt(dx*dx + dy*dy)/dl
            bary[i]  = 1.0-mu;
            bary[ip] = mu
            return bary
        
    w = np.zeros((nSize), dtype=float)
    wsum = 0
    for i in range(nSize):
        # w[i] = 0
        ip = (i+1) % nSize
        im = (nSize-1+i) % nSize
        if abs(Ai[im]) > epsilon:
            w[i] = w[i] + (ri[im] - Di[im]/ri[i]) / Ai[im]
        if abs(Ai[i]) > epsilon:
            w[i] = w[i] + (ri[ip] - Di[i]/ri[i]) / Ai[i]
        
    for i in range(nSize):
        wsum += w[i]
    
    for i in range(nSize):
        w[i] /= wsum
        bary[i] = w[i]
    
    return bary

def interpolate_point(vertices, weights):
    return np.sum(np.array(vertices) * np.expand_dims(weights, axis=1), axis=0)

def get_deformed_boundary(moving_boundary, fixed_boundary, device, moving_silhouette, fixed_silhouette):

    moving_mesh = GetCDT(moving_boundary, flags='p')
    fixed_mesh = GetCDT(fixed_boundary, flags='p')

    convert_to_ply('moving_test.ply', moving_mesh, moving_silhouette)
    convert_to_ply('fixed_test.ply', fixed_mesh, fixed_silhouette)
    config.device = device

    S = 'moving_test.ply'
    T = 'fixed_test.ply'

    # read S, sample pts
    src_mesh = o3d.io.read_triangle_mesh( S )
    src_mesh.compute_vertex_normals()
    pcd1 =  src_mesh.sample_points_uniformly(number_of_points=config.samples)
    pcd1.paint_uniform_color([0, 0.706, 1])
    src_pcd = np.asarray(pcd1.points, dtype=np.float32)

    # read T, sample pts
    tgt_mesh = o3d.io.read_triangle_mesh( T )
    tgt_mesh.compute_vertex_normals()
    pcd2 =  tgt_mesh.sample_points_uniformly(number_of_points=config.samples)
    tgt_pcd = np.asarray(pcd2.points, dtype=np.float32)

    orig_src_mesh = o3d.cuda.pybind.geometry.TriangleMesh(src_mesh)

    # Run the model on the video and sketch triangulation

    """load data"""
    src_pcd, tgt_pcd = map( lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd ] )

    """construct model"""
    NDP = Deformation_Pyramid(depth=config.depth,
                                width=config.width,
                                device=config.device,
                                k0=config.k0,
                                m=config.m,
                                nonrigidity_est=config.w_reg > 0,
                                rotation_format=config.rotation_format,
                                motion=config.motion_type)

    """cancel global translation"""
    src_mean = src_pcd.mean(dim=0, keepdims=True)
    tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
    src_pcd = src_pcd - src_mean
    tgt_pcd = tgt_pcd - tgt_mean

    s_sample = src_pcd
    t_sample = tgt_pcd

    BCE = nn.BCELoss()

    for level in range(NDP.n_hierarchy):

        """freeze non-optimized level"""
        NDP.gradient_setup(optimized_level=level)

        optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for iter in range(config.iters):
            s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)
            loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)
            if level > 0 and config.w_reg > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + config.w_reg * reg_loss

            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # use warped points for next level
        s_sample = s_sample_warped.detach()

    """warp-original mesh vertices"""
    NDP.gradient_setup(optimized_level=-1)
    mesh_vert = torch.from_numpy(np.asarray(src_mesh.vertices, dtype=np.float32)).to(config.device)
    mesh_vert = mesh_vert - src_mean
    warped_vert, data = NDP.warp(mesh_vert)
    warped_vert = warped_vert.detach().cpu().numpy()
    src_mesh.vertices = o3d.utility.Vector3dVector(warped_vert)

    # Translating the result to match the fixed triangulation
    # The result is shape-matched but may not be in the same position. We thus move centroid of result to the centroid of the fixed triangulation.
    moving_polygon = Polygon(np.asarray(src_mesh.vertices))
    fixed_polygon = Polygon(np.asarray(tgt_mesh.vertices))

    moving_centroid = np.array([moving_polygon.centroid.x, moving_polygon.centroid.y, 0])
    fixed_centroid = np.array([fixed_polygon.centroid.x, fixed_polygon.centroid.y, 0])
    delta_centroid = moving_centroid - fixed_centroid
    src_mesh_vertices = np.array(src_mesh.vertices) - delta_centroid

    deformation_mesh = src_mesh

    deformed_boundary_vertices = []

    fixed_height, fixed_width = fixed_silhouette.shape[:2]

    polygon = Polygon(fixed_boundary)
    for point in src_mesh_vertices:
        pointt = Point((point[0]+1)*fixed_width/2, (point[1]+1)*fixed_height/2)
        closest_point = polygon.exterior.interpolate(polygon.exterior.project(pointt))
        deformed_boundary_vertices.append([closest_point.x, closest_point.y])
    
    return deformed_boundary_vertices

# -----------------------------------------------------------------------------------------------------------

# TODO: Refactor this code to remove the use of external files

def convert_to_ply(ply_path, triangulation, image):
    with open(ply_path, 'w') as file:
        file.write(f"""ply
format ascii 1.0
element vertex {triangulation['vertices'].shape[0]}
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face {triangulation['triangles'].shape[0]}
property list uchar uint vertex_indices
end_header
""")
        
        image_height, image_width = image.shape[:2]

        for vertex in triangulation['vertices']:
            file.write(f'{2*(vertex[0]/image_width)-1} {2*(vertex[1]/image_height)-1} 0.0 0.0 0.0 1.0\n')
        for face in triangulation['triangles']:
            file.write(f'3 {face[0]} {face[1]} {face[2]}\n')

config = {
    "gpu_mode": True,
    "use_ldmk": False,
    "iters": 1000000,
    "lr": 0.01,
    "max_break_count": 15,
    "break_threshold_ratio": 0.0003,
    "samples": 6000,
    "motion_type": "Sim3",
    "rotation_format": "euler",
    "m": 9,
    "k0": -8,
    "depth": 3,
    "width": 128,
    "act_fn": "relu",
    "w_reg": 0,
    "w_ldmk": 0,
    "w_cd": 0.1
}

config = edict(config)