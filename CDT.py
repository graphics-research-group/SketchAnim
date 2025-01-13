import triangle as tr
import numpy as np
import cv2

# create silhouette on the given image
def CreateSilhouette(image, output_shape=None):
    scaled_image = image
    if output_shape is not None:
        scaled_image = cv2.resize(image, output_shape)
    height, width = scaled_image.shape[:2]
    threshold = 0
    sketch_silhouette = np.zeros(scaled_image.shape[:2],dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if scaled_image[i,j,3] > threshold:
                sketch_silhouette[i][j] = 255
    return sketch_silhouette

# plot the triangulate of the given input image
def PlotTriangulationOnImage(image, triangulation_vertices, triangulation_faces, vertex_colour = (0, 169, 0), edge_colour = (0, 50, 0)):
    img_copy = image.copy()
    
    width =img_copy.shape[1]

    # For each triangle, draw corresponding edges on the image
    img_triangulation = img_copy.copy()
    # for simplice in triangulation.simplices:
    for simplice in triangulation_faces:
        point0 = (int(triangulation_vertices[simplice[0]][0]), int(triangulation_vertices[simplice[0]][1]))
        point1 = (int(triangulation_vertices[simplice[1]][0]), int(triangulation_vertices[simplice[1]][1]))
        point2 = (int(triangulation_vertices[simplice[2]][0]), int(triangulation_vertices[simplice[2]][1]))
        img_triangulation = cv2.line(img_triangulation, point0, point1, edge_colour, round(width*0.001))
        img_triangulation = cv2.line(img_triangulation, point1, point2, edge_colour, round(width*0.001))
        img_triangulation = cv2.line(img_triangulation, point0, point2, edge_colour, round(width*0.001))

    # Displaying the points on the original figure
    for point in triangulation_vertices:
        cv2.circle(img_copy, (int(point[0]), int(point[1])), 2, vertex_colour, -1)
    
    return img_triangulation

'''
Get Contrained Delaunay Triangulation for a given polygon
Input:
    1) `boundary_points` - List of vertices on the boundary in order.
    2) `interior_points` (optional) - List of interior points to be added. If left none, points will be added as per Poisson Disk Sampling
    3) `skeleton_vertices` (optional) - List of skeleton vertices to be added
    4) `skeleton_edges` (optional) - List of skeleton edges to be added
Output:
    A triangle object `cdt`. `cdt['vertices'], cdt['triangles']` give the triangulation vertices and faces respectively.
'''
def GetCDT(boundary_points, interior_points = None, skeleton_vertices = None, skeleton_edges = None, flags='pa50'):
    vertices = boundary_points
    segments = []
    for i in range(len(boundary_points)):
        segments.append((i, (i+1) % len(boundary_points)))
    
    if interior_points is not None:
        vertices = list(vertices) + list(interior_points)
    
    if skeleton_edges is not None:
        num_vertices = len(vertices)
        for i in range(len(skeleton_edges)):
            segments.append((skeleton_edges[i][0] + num_vertices, skeleton_edges[i][1] + num_vertices))
    
    if skeleton_vertices is not None:
        vertices = list(vertices) + list(skeleton_vertices)
    
    # print(f'Vertices: {vertices}')
    # print(f'Segments: {segments}')

    cdt = tr.triangulate({'vertices': vertices, 'segments': segments}, flags)

    return cdt

'''
Given a silhouette, sample some points from its boundary and return the boundary vertices.
Input:
    1) `silhouette` - Numpy Array image of the silhouette
    2) `padding` (default=0) - Number of pixels by which samples should be away from the boundary.
        This ensures triangulation does not cut the silhouette boundary.
Output:
    (N,2) size Numpy arrayof boundary points, where N is the number of boundary points
'''
def GetBoundaryFromSilhouette(silhouette, num_samples, padding=0, max_distance=4):
    kernel = np.ones((3,3), np.uint8)
    dilated_silhouette = cv2.dilate(silhouette, kernel, iterations=padding)
    contours, _ = cv2.findContours(dilated_silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Any noise that is disconnected from the silhouette should have fewer boundary points from the main silhouette
    # Filter out any contours with fewer than maximum points to remove noise
    largest_contour_index = 0
    for i in range(len(contours)):
        if len(contours[largest_contour_index]) < len(contours[i]):
            largest_contour_index = i
    
    main_contour = np.array(contours[largest_contour_index])

    print(len(main_contour))

    step = len(main_contour)//num_samples
    spin=0

    # There should not be too much gap between any 2 vertices.
    boundary_points = [main_contour[0]]
    for i in range(1, len(main_contour)):
        prev_point = boundary_points[-1]
        D = np.linalg.norm(prev_point - main_contour[i])
        if max_distance >= 0 and D > max_distance:
            s = max_distance / D
            t = 0
            while t < 1:
                new_pt = t*main_contour[i] + (1-t)*prev_point
                boundary_points.append(new_pt)
                t += s
        # else:
        #     boundary_points.append(main_contour[i])
        spin += 1
        if spin >= step:
            spin = 0
            boundary_points.append(main_contour[i])

    return np.array(boundary_points).squeeze()

def PlotBoundaryOnImage(image, boundary_points):
    boundary_image = np.array(image)
    width = boundary_image.shape[1]
    for point in boundary_points:
        cv2.circle(boundary_image, (int(point[0]), int(point[1])), round(width*0.01), (255, 0, 255), -1)
    return boundary_image

def render_triangulation(
        image,
        triangulation_vertices,
        triangulation_faces,
        vertex_colour = (0, 169, 0),
        edge_colour = (0, 169, 0),
        boundary_vertices = None,
        render_edges = True,
        point_proportion=0.005,
        line_proportion=0.005
        ):
    img_copy = image.copy()
    
    # For each triangle, draw corresponding edges on the image
    img_triangulation = img_copy.copy()
    width = img_triangulation.shape[1]
    
    if render_edges:
        for simplice in triangulation_faces:
            point0 = (int(triangulation_vertices[simplice[0]][0]), int(triangulation_vertices[simplice[0]][1]))
            point1 = (int(triangulation_vertices[simplice[1]][0]), int(triangulation_vertices[simplice[1]][1]))
            point2 = (int(triangulation_vertices[simplice[2]][0]), int(triangulation_vertices[simplice[2]][1]))
            img_triangulation = cv2.line(img_triangulation, point0, point1, edge_colour, round(width*line_proportion))
            img_triangulation = cv2.line(img_triangulation, point1, point2, edge_colour, round(width*line_proportion))
            img_triangulation = cv2.line(img_triangulation, point0, point2, edge_colour, round(width*line_proportion))

    if boundary_vertices is not None:
        for point in boundary_vertices:
            cv2.circle(img_triangulation, (int(point[0]), int(point[1])), round(width*point_proportion), vertex_colour, -1)
    else:
        # Displaying the points on the original figure
        for point in triangulation_vertices:
            cv2.circle(img_triangulation, (int(point[0]), int(point[1])), round(width*point_proportion), vertex_colour, -1)
    
    return img_triangulation

# Test
if __name__ == '__main__':
    boundary_vertices = [[0,0], [0,10], [10,0], [10,10]]
    skeleton_vertices = [[5,5], [6,6]]
    skeleton_edges = [[0,1]]
    cdt = GetCDT(boundary_points=boundary_vertices, skeleton_vertices=skeleton_vertices, skeleton_edges=skeleton_edges)
    print(cdt['vertices'])
    print(cdt['triangles'])