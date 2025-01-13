import os
import numpy as np
import cv2
import triangle as tr
import torch
import imageio
from PIL import Image
import tempfile

from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    OrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)


def RenderFrameAnimation(
        moving_img_filepath,
        triangulation,
        mesh_vertices_across_frames,
        output_directory,
        video_frames=None,
        device='cpu',
        verbose=False):
    triangulation_faces = triangulation['triangles']
    triangulation_vertices_firstframe = triangulation['vertices']

    moving_img_pil = Image.open(moving_img_filepath)
    moving_img = np.asarray(moving_img_pil)

    img_height, img_width = moving_img.shape[:2]

    with tempfile.TemporaryDirectory() as temp_dir:
        moving_rgb_filepath = os.path.join(temp_dir,'sketch_rgb.png')
        moving_img_rgb = Image.new("RGB", moving_img_pil.size, "white")
        moving_img_rgb.paste(moving_img_pil, (0, 0), moving_img_pil)
        moving_img_rgb.save(moving_rgb_filepath)

        def add_mtl(mtl_path, img_path):
            with open(mtl_path, 'w') as file:
                file.write(f"""newmtl texture_material
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 1.000000 1.000000 1.000000
d 1.0
illum 0
map_Kd {img_path}""")

            # Confirm that the file was saved
            assert os.path.exists(mtl_path), "File not found"


        # FUNCTION TO ADD THE OBJ FILE
        def add_obj(obj_filepath, mtl_filepath, triangulation_vertices, triangulation_faces, width=img_width, height=img_height):
            with open(obj_filepath, 'w') as file:
                # Write the text to the file
                file.write(f"""
# Material library
mtllib {mtl_filepath}

# Vertex positions
""")
                # Add the vertex positions, transformed from image space to orthographic camera space
                for point in triangulation_vertices:
                    camera_space_x = 2*(1 - (point[0] / width))-1
                    camera_space_y = 2*(1 - (point[1] / height))-1
                    camera_space_z = 0
                    if len(point) >= 3:
                        camera_space_z = point[2]
                    file.write(f"v {camera_space_x} {camera_space_y} {camera_space_z}\n")

                # Add texture map coordinates
                file.write("\n# Texture coordinates\n")

                for point in triangulation_vertices_firstframe:
                    camera_space_x = (point[0] / width)
                    camera_space_y = 1 - (point[1] / height)
                    file.write(f"vt {camera_space_x} {camera_space_y}\n")

                # Add Face Information Indices
                file.write("\n# Face\n")
                for face in triangulation_faces:
                    file.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")

                # End of file
                file.write("\n# End of File\n")
            # Confirm that the file was saved
            assert os.path.exists(obj_filepath), "File not found"

        objs_path = os.path.join(temp_dir, 'pytorch3d_objs')
        frames_path = os.path.join(temp_dir, 'rendered_images')

        os.makedirs(objs_path)

        add_mtl(os.path.join(objs_path, 'texture.mtl'), os.path.join(os.getcwd(), moving_img_filepath))
        add_obj(os.path.join(objs_path, 'triangulation.obj'), os.path.join(objs_path, 'texture.mtl'),
                triangulation_vertices_firstframe, triangulation_faces, width=img_width, height=img_height)

        mesh = load_objs_as_meshes([os.path.join(objs_path, 'triangulation.obj')], device=device)

        # SHADER CLASS
        from pytorch3d.renderer.mesh.rasterizer import Fragments
        class TextureMapShader(ShaderBase):
            def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
                texels = meshes.sample_textures(fragments)
                images = texels
                return images
        # RENDERER
        R, T = look_at_view_transform(100, 0, 180) # Camera turns 180 degrees, distance 1 away from image
        cameras = OrthographicCameras(device=device, R=R, T=T) # Camera renders information in orthographic view
        raster_settings = RasterizationSettings(
            image_size=1024,
            blur_radius=0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=TextureMapShader(
                device=device,
                cameras=cameras,
                lights=lights
            ),
        )

        rendered_frames = []
        # MTL needs to be added once. Supply the path to the texture file (the image that needs to be mapped in final triangulation)
        add_mtl(os.path.join(objs_path, 'texture.mtl'), os.path.join(os.getcwd(), moving_rgb_filepath))
        for i in range(mesh_vertices_across_frames.shape[0]):
            triangulation_vertices = mesh_vertices_across_frames[i]
            img_filepath = os.path.join(frames_path, 'frame_'+"{:04d}".format(i)+'.jpg')
            # OBJ file needs to be created for every frame
            add_obj(os.path.join(objs_path, f'triangulation_frame' + '{:04d}'.format(i) + '.obj'), os.path.join(objs_path, 'texture.mtl'),
                    triangulation_vertices,triangulation_faces)
            mesh = load_objs_as_meshes([os.path.join(objs_path, f'triangulation_frame' + '{:04d}'.format(i) + '.obj')], device=device)
            images = renderer(mesh)
            images_cpu = images.cpu().numpy()
            images_cpu = images_cpu.squeeze()
            rendered_frames.append(images_cpu)

        # RENDER ANIMATION
        
        # Render full animation
        frames = []
        for i in range(len(rendered_frames)):
            image2 = np.array(rendered_frames[i]*255, dtype = np.uint8)
            # image2 = cv2.resize(image2, (512,512))
            frames.append(image2)
        output_animation_filepath = os.path.join(output_directory, 'renderer_animation.gif')
        imageio.mimsave(output_animation_filepath, frames, loop=0)
        if verbose:
            print(f'Animation saved in {output_animation_filepath}')

        # Render side by side animation
        frames = []

        # gif_reader = imageio.get_reader(video_filepath)
        # video_frames = [frame for frame in gif_reader]

        for i in range(len(rendered_frames)):
            # image1 = np.array(Image.open(os.path.join(frames_path, 'frame_'+'{:04d}'.format(i)+'.jpg')), dtype = np.uint8)
            image1 = np.array(video_frames[i], dtype = np.uint8)[:,:,:3]
            image2 = np.array(rendered_frames[i]*255, dtype = np.uint8)
            image1 = cv2.resize(image1, (512,512))
            image2 = cv2.resize(image2, (512,512))
            side_by_side = np.concatenate((image1, image2), axis=1)
            frames.append(side_by_side)
        output_sidebyside_filepath = os.path.join(output_directory, 'renderer_sidebyside.gif')
        imageio.mimsave(output_sidebyside_filepath, frames, loop=0)
        if verbose:
            print(f'Sidebyside saved in {output_sidebyside_filepath}')



# ---------------------------------------------------------------------------------------

