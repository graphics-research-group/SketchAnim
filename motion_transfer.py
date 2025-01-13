import os
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

from cffi import FFI
import ctypes

'''
Mapping the video motion to input sketch.
Input:
    1) 'fixed_vertices_across_frames'
    2) 'moving_vertices_rest_pose'
    3) 'edges' - bones of the skeleton 
    4) 'sketch_img' - input sketch image
    5) 'video_img' - rest pose of the input video frame
Output:
     Moving vertices across the frames.
'''
def SkeletonMotionTransfer(fixed_vertices_across_frames, moving_vertices_rest_pose, edges, sketch_img, video_img):

    image_centre = np.array(sketch_img.shape[:2])/2
    silhouette_centre = np.array([
        (min([i for i in range(len(sketch_img)) if any(pixel[3]>0 for pixel in sketch_img[i])])+
         max([i for i in range(len(sketch_img)) if any(pixel[3]>0 for pixel in sketch_img[i])])
         )/2,
        (min([i for i in range(sketch_img.shape[1]) if any(pixel[3]>0 for pixel in sketch_img[:,i,:])])+
         max([i for i in range(sketch_img.shape[1]) if any(pixel[3]>0 for pixel in sketch_img[:,i,:])])
         )/2,
    ])

    print(image_centre)
    print(silhouette_centre)

    num_bones = len(edges)
    num_frames = len(fixed_vertices_across_frames)
    num_points = len(moving_vertices_rest_pose)

    moving_bone_lengths = np.zeros(num_bones)

    for i in range(num_bones):
        bone = edges[i]
        moving_bone_lengths[i] = np.linalg.norm(moving_vertices_rest_pose[bone[0]] - moving_vertices_rest_pose[bone[1]])

    bone_directions = np.zeros((num_frames, num_bones, 2))

    for frame_no in range(num_frames):
        for bone_no in range(num_bones):
            bone = edges[bone_no]
            vector = fixed_vertices_across_frames[frame_no][bone[1]] - fixed_vertices_across_frames[frame_no][bone[0]]
            direction = np.linalg.norm(vector)
            bone_directions[frame_no][bone_no] = vector/direction

    moving_vertices_across_frames = np.zeros((num_frames, num_points, 2))

    for frame_no in range(num_frames):
        moving_vertices_across_frames[frame_no][0] = (fixed_vertices_across_frames[frame_no][0] * np.array(
            [np.array(sketch_img.shape[:2])/np.array(video_img.shape[:2])])) + np.array([0,500])
        # moving_vertices_across_frames[frame_no][0] = moving_vertices_rest_pose[0] + image_centre - silhouette_centre
        for bone_no in range(num_bones):
            bone = edges[bone_no]
            point0 = moving_vertices_across_frames[frame_no][bone[0]]
            point1 = point0 + (moving_bone_lengths[bone_no] * bone_directions[frame_no][bone_no])
            moving_vertices_across_frames[frame_no][bone[1]] = point1

    return moving_vertices_across_frames

'''
Estimate bounded biharmonic weights.
Input:
    1) 'triangulation' - triangulation of the input sketch
    2) 'skeleton vertices' - vertex points of the skeleton handle
    3) 'skeleton_edges'
Output:
    BBW weights (NxB) where N is the number of vertices and B is the number of bones
'''
def GetBBWWeights(triangulation, skeleton_vertices, skeleton_edges):
    vs = triangulation['vertices']
    faces = triangulation['triangles']
    handles = skeleton_vertices
    bones = [edge for bone in skeleton_edges for edge in bone]

    W = bbw(vs, faces, handles, [], bones)
    return W


'''
Estimate the deformed mesh across the frames.
Input:
    1) 'mesh' - mesh of the sketch input 
    2) 'vertices_across_frames' - vertex points across the frame
    3) 'vertices_rest_pose' - vertices of the rest pose of the video frame
    4) 'edges' - skeleton edges
    5) bone_depths' - depth order of the skeleton vertices
Output:
    return the deformed mesh. 
'''

def GetMeshTransformsFromSkeleton(mesh, vertices_across_frames, vertices_rest_pose, edges, bone_depths):
    num_frames = len(vertices_across_frames)
    
    W = GetBBWWeights(mesh, vertices_rest_pose, edges)

    # Calculate bone transformation matrices
    transformation_matrices_list = []
    z_scaling = -0.05
    for frame_no in range(num_frames):
        transformation_matrices = []
        bone_no = 0
        for bone in edges:
            v1a = np.array([vertices_rest_pose[bone[0]][0], vertices_rest_pose[bone[0]][1], 1])
            v1b = np.array([vertices_rest_pose[bone[1]][0], vertices_rest_pose[bone[1]][1], 1])
            v2a = np.array([vertices_across_frames[frame_no][bone[0]][0], vertices_across_frames[frame_no][bone[0]][1], bone_depths[bone[0]]*z_scaling])
            v2b = np.array([vertices_across_frames[frame_no][bone[1]][0], vertices_across_frames[frame_no][bone[1]][1], bone_depths[bone[1]]*z_scaling])

            M = get_3d_transformation_matrix(v1a, v1b, v2a, v2b)
            transformation_matrices.append(M)

        transformation_matrices = np.array(transformation_matrices)
        transformation_matrices_list.append(transformation_matrices)

    transformation_matrices_list = np.array(transformation_matrices_list)

    # Calculate mesh transformations using LBS
    triangulation_vertices = mesh['vertices']
    homogeneous_triangulation_vertices = np.hstack((triangulation_vertices, np.ones((triangulation_vertices.shape[0],2))))
    triangulation_across_frames = []
    for frame_no in range(num_frames):
        triangulation_across_frames.append(LBS(homogeneous_triangulation_vertices, transformation_matrices_list[frame_no], W))

    triangulation_across_frames = np.array(triangulation_across_frames)

    return triangulation_across_frames

'''
Compute depth order of the skeleton points.
'''
def GetVirtualVertexDepth(edges):
    num_verts = len(edges)+1
    root = 0
    for v in range(num_verts):
        if list(edges.flatten()).count(root) < list(edges.flatten()).count(v):
            root = v

    graph = []
    for _ in range(num_verts):
        graph.append([])
    for e in edges:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])

    depths = np.full((num_verts), -1)

    queue = [(root, 1)]
    while len(queue) > 0:
        curr, d = queue.pop()
        depths[curr] = d
        for v in graph[curr]:
            if depths[v] < 0:
                queue.append((v, d+1))

    depths[5] = 0
    depths[6] = 0

    return depths

# ---------------------------------------------------------------------------------------------------------
# Estimate the defomation vertex point given the vertices, transformation matric, and bbw weigths
def LBS(vertices, transformation_matrix, weights):
    num_vertices = vertices.shape[0]
    num_bones = transformation_matrix.shape[0]
    dim = vertices.shape[1]
    
    skinned_vertices = np.zeros((num_vertices, dim))

    for i in range(num_vertices):
        weighted_transform = np.zeros(dim)
        for j in range(num_bones):
            # weighted_transform += weights[i, j] * np.dot(vertices[i], transformation_matrix[j])
            weighted_transform += weights[i, j] * np.dot(transformation_matrix[j], vertices[i])
        skinned_vertices[i] = weighted_transform

    return skinned_vertices

# Estimate the rotation matices from skeleton joint transformation
def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_3d_transformation_matrix(v1a, v1b, v2a, v2b):
    vec1 = v1b - v1a
    vec2 = v2b - v2a
    s = np.linalg.norm(vec2)/np.linalg.norm(vec1)
    
    T2 = np.array([[ 1,0,0,-v1a[0]],
                    [0,1,0,-v1a[1]],
                    [0,0,1,-v1a[2]],
                    [0,0,0,1]], dtype = np.float32)
    R_3x3 = rotation_matrix_from_vectors(vec1, vec2)
    R = np.eye(4)
    R[:3,:3] = R_3x3
    R[3,3] = 1
    S = np.array([  [s,0,0,0,],
                    [0,s,0,0,],
                    [0,0,s,0,],
                    [0,0,0,1,]])
    T1 = np.array([[ 1,0,0,v2a[0]],
                    [0,1,0,v2a[1]],
                    [0,0,1,v2a[2]],
                    [0,0,0,1]], dtype = np.float32)
    
    M = np.dot(T1, np.dot(S, np.dot(R, T2)))

    return M

index_t = ctypes.c_int
real_t = ctypes.c_double

ffi = FFI()
ffi.cdef("""
typedef double real_t;
typedef int index_t;

// Returns 0 for success, anything else is an error.
int bbw(
    /// Input Parameters
    // 'vertices' is a pointer to num_vertices*kVertexDimension floating point values,
    // packed: x0, y0, z0, x1, y1, z1, ...
    // In other words, a num_vertices-by-kVertexDimension matrix packed row-major.
    int num_vertices, real_t* vertices,
         
    // 'faces' is a pointer to num_faces*3 integers,
    // where each face is three vertex indices: f0.v0, f0.v1, f0.v2, f1.v0, f1.v1, f1.v2, ...
    // Face i's vertices are: vertices[ faces[3*i]*2 ], vertices[ faces[3*i+1]*2 ], vertices[ faces[3*i+2]*2 ]
    // In other words, a num_faces-by-3 matrix packed row-major.
    int num_faces, index_t* faces,
         
    // 'skeleton_vertices' is a pointer to num_skeleton_vertices*kVertexDimension floating point values,
    // packed the same way as 'vertices' (NOTE: And whose positions must also exist inside 'vertices'.)
    int num_skeleton_vertices, real_t* skeleton_vertices,
         
    // 'skeleton_point_handles' is a pointer to num_skeleton_point_handles integers,
    // where each element "i" in skeleton_point_handles references the vertex whose data
    // is located at skeleton_vertices[ skeleton_point_handles[i]*kVertexDimension ].
    int num_skeleton_point_handles, index_t* skeleton_point_handles,
         
    // TODO: Take skeleton bone edges and cage edges
    // ** Taking skeleton bone edges
    int num_skeleton_bone_handles, index_t* skeleton_bone_handles,
    
    /// Output Parameters
    // 'Wout' is a pointer to num_vertices*num_skeleton_vertices values.
    // Upon return, W will be filled with each vertex in 'num_vertices' weight for
    // each skeleton vertex in 'num_skeleton_vertices'.
    // The data layout is that all 'num_skeleton_vertices' weights for vertex 0
    // appear before all 'num_skeleton_vertices' weights for vertex 1, and so on.
    // In other words, a num_vertices-by-num_skeleton_vertices matrix packed row-major.
    real_t* Wout
    );
""")

def platform_shared_library_suffix():
    import sys
    result = '.so'
    if 'win' in sys.platform.lower(): result = '.dll'
    ## No else if, because we want darwin to override win (which is a substring of darwin)
    if 'darwin' in sys.platform.lower(): result = '.dylib'
    return result

# libbbw = ffi.dlopen( os.path.join( os.path.dirname( __file__ ), 'bbw' + platform_shared_library_suffix() ) )
libbbw = ffi.dlopen( os.path.join( os.path.curdir, 'bbw_wrapper/bbw' + platform_shared_library_suffix() ) )

from cffi import FFI

class BBWError( Exception ): pass

def bbw( vertices, faces, skeleton_handle_vertices, skeleton_point_handles, skeleton_bone_handles ):
    '''
    Given an N-by-(2 or 3) numpy array 'vertices' of 2D or 3D vertices,
    an M-by-3 numpy array 'faces' of indices into 'vertices',
    an H-by-(2 or 3) numpy.array 'skeleton_handle_vertices' of 2D or 3D vertices,
    a numpy array 'skeleton_point_handles' of indices into 'skeleton_handle_vertices'
    which are the point handles,
    returns a N-by-H numpy.array of weights per vertex per handle.
    
    NOTE: All the vertices in 'skeleton_handle_vertices' must also exist in 'vertices'.
    '''
    
    import numpy
    
    ## Make sure the input values have their data in a way easy to access from C.
    vertices = numpy.ascontiguousarray( numpy.asarray( vertices, dtype = real_t ) )
    faces = numpy.ascontiguousarray( numpy.asarray( faces, dtype = index_t ) )
    skeleton_handle_vertices = numpy.ascontiguousarray( numpy.asarray( skeleton_handle_vertices, dtype = real_t ) )
    skeleton_point_handles = numpy.ascontiguousarray( numpy.asarray( skeleton_point_handles, dtype = index_t ) )
    skeleton_bone_handles = numpy.ascontiguousarray( numpy.asarray( skeleton_bone_handles, dtype = index_t ) )
    
    ## We allow for 2D or 3D vertices and skeleton_handle_vertices, but
    ## the dimensions must match.
    assert vertices.shape[1] == skeleton_handle_vertices.shape[1]
    
    assert len( vertices.shape ) == 2
    assert vertices.shape[1] in (2,3)
    ## Turn 2D vertices into 3D vertices by using z = 0.
    if vertices.shape[1] == 2:
        vertices2d = vertices
        vertices = numpy.ascontiguousarray( numpy.zeros( ( len( vertices ), 3 ), dtype = real_t ) )
        vertices[:,:2] = vertices2d
    
    assert len( faces.shape ) == 2
    assert faces.shape[1] == 3
    
    assert len( skeleton_handle_vertices.shape ) == 2
    assert skeleton_handle_vertices.shape[1] in (2,3)
    ## Turn 2D vertices into 3D vertices by using z = 0.
    if skeleton_handle_vertices.shape[1] == 2:
        skeleton_handle_vertices2d = skeleton_handle_vertices
        skeleton_handle_vertices = numpy.ascontiguousarray( numpy.zeros( ( len( skeleton_handle_vertices ), 3 ), dtype = real_t ) )
        skeleton_handle_vertices[:,:2] = skeleton_handle_vertices2d
    
    assert len( skeleton_point_handles.shape ) == 1
    assert len( skeleton_point_handles ) == len( set( skeleton_point_handles ) )
    
    num_skeleton_components = len( skeleton_point_handles ) + len( skeleton_bone_handles ) // 2

    # Wout = numpy.empty( ( len( vertices ), len( skeleton_handle_vertices ) ), dtype = real_t )
    Wout = numpy.empty( ( len( vertices ), num_skeleton_components ), dtype = real_t )
#     debugger()
    result = libbbw.bbw(
        len( vertices ),                 ffi.cast( 'real_t*',  vertices.ctypes.data ),
        len( faces ),                    ffi.cast( 'index_t*', faces.ctypes.data ),
        len( skeleton_handle_vertices ), ffi.cast( 'real_t*',  skeleton_handle_vertices.ctypes.data ),
        len( skeleton_point_handles ),   ffi.cast( 'index_t*', skeleton_point_handles.ctypes.data ),
        len( skeleton_bone_handles ) // 2,   ffi.cast( 'index_t*', skeleton_bone_handles.ctypes.data ),
        
        ffi.cast( 'real_t*', Wout.ctypes.data )
        )
    if result != 0:
        raise BBWError( f'bbw() reported an error ({result})' )
    
    return Wout