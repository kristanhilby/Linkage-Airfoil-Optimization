import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def generate_airfoil_dat_from_points(points, display_plot=False, export_dat=False, additional_points=60, trailing_edge_smooth_factor=0.1):
    """
    Generate a smooth, mirrored, airfoil shape from a series of control points and save it to a .dat file.

    Parameters:
    - points: A numpy array of control points (n x 2) where each row is [x, y].
    - display_plot: Boolean flag to display the airfoil plot (default is False).
    - export_dat: Boolean flag to save the generated airfoil file (default is False).
    - additional_points: Number of extra points on the bottom surface (default is 60).
    - trailing_edge_smoothing_factor: Degree of curvature on the bottom surface (default is 0.1).
    
    Returns:
    - reordered_array_with_ends: [x,y] array of the generated airfoil coordinates.
    """

    slope_1 = (points[3]-points[1])/(points[2]-points[0])
    slope_2 = (points[7]-points[5])/(points[6]-points[4])

    offset_distance = points[9]
    #offset_distance = 0.05
    #offset_distance = abs(slope_1+slope_2)/2

    direction = np.array([points[4] - points[2], points[5] - points[3]])

    direction_perp = [-direction[1], direction[0]]

    unit_vec = direction_perp / np.linalg.norm(direction_perp)

    midpoint = np.array([(points[2] + points[4])/2, (points[3] + points[5])/2])

    offsetpt = midpoint + offset_distance*unit_vec

    # Separate the points into x and y and add l0 = 1
    # Assuming input vector x = (x1, y1, x2, y2, x3, y3, x4, y4, y6)
    # x = (0, points[0], points[2], offsetpt[0], points[4], points[6], 1)
    # y = (0, points[1], points[3], offsetpt[1], points[5], points[7], 0)

    x = (0, points[0], points[2])
    y = (0, points[1], points[3])

    x_center = (points[2], offsetpt[0], points[4])
    y_center = (points[3], offsetpt[1], points[5])

    x_right = (points[4], points[6], 1)
    y_right = (points[5], points[7], 0)

    # Smooth the airfoil shape using splines
    tck, u = splprep([x, y], s=0.0, per=False, k=2)  # s is the smoothing factor
    x_spline, y_spline = splev(np.linspace(0, 1, 200), tck)

    tck2, u2 = splprep([x_center, y_center], s=0.0, per=False, k=2)  # s is the smoothing factor
    x_spline_center, y_spline_center = splev(np.linspace(0, 1, 200), tck2)

    tck3, u3 = splprep([x_right, y_right], s=0.0, per=False, k=2)  # s is the smoothing factor
    x_spline_right, y_spline_right = splev(np.linspace(0, 1, 200), tck3)

    # Identify the first and last points
    x1, y1 = x_spline[0], y_spline[0]  # First point
    x2, y2 = x_spline_right[-1], y_spline_right[-1]  # Last point

    # Generate symmetric control points for trailing edge curve
    cp1 = (0.5, y1 + trailing_edge_smooth_factor)  # Control point at symmetry axis
    cp2 = (0.5, y2 + trailing_edge_smooth_factor)

    # Generate points for the symmetric connection
    t = np.linspace(0, 1, additional_points)
    bezier_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * cp1[0] + t**2 * x2
    bezier_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * cp1[1] + t**2 * y2

    bezier_x_transpose = np.flip(bezier_x)
    bezier_y_transpose = np.flip(bezier_y)

    # Combine main airfoil and trailing edge curve
    x_final = np.concatenate([x_spline[:-1], x_spline_center, x_spline_right[1:], bezier_x_transpose[1:-1]])
    y_final = np.concatenate([y_spline[:-1], y_spline_center, y_spline_right[1:], bezier_y_transpose[1:-1]])

    airfoil_coordinates = np.array([x_final, y_final]).T

    halfway_index = len(airfoil_coordinates) // 2
    index_1 = np.argmin(np.abs(airfoil_coordinates[0:halfway_index,0]-points[0]) + np.abs(airfoil_coordinates[0:halfway_index,1] - points[1]))
    index_2 = np.argmin(np.abs(airfoil_coordinates[0:halfway_index,0]-points[2]) + np.abs(airfoil_coordinates[0:halfway_index,1] - points[3]))
    #index_3 = np.argmin(np.abs(airfoil_coordinates[:,0]-points[4]))
    #index_center = np.argmin(np.abs(airfoil_coordinates[:,0] - offsetpt[0]) + np.abs(airfoil_coordinates[:,1] - offsetpt[1]))
    index_3 = np.argmin(np.abs(airfoil_coordinates[:,0] - points[4]) + np.abs(airfoil_coordinates[:,1] - points[5]))

    link1_mirror_x, link1_mirror_y = mirror_points(airfoil_coordinates[0:index_1-1,0], airfoil_coordinates[0:index_1-1,1])
    link2_mirror_x, link2_mirror_y = mirror_points(airfoil_coordinates[index_1:index_2,0], airfoil_coordinates[index_1:index_2,1])
    #linkc_mirror_x, linkc_mirror_y = mirror_points(airfoil_coordinates[index_2:index_center,0], airfoil_coordinates[index_2:index_center,1])

    link1_mirrored_x, link1_mirrored_y = transform_curve(link1_mirror_x, link1_mirror_y, (link1_mirror_x[0], link1_mirror_y[0]), (link1_mirror_x[-1], link1_mirror_y[-1]), (1,0), (points[6], points[7]))
    link2_mirrored_x, link2_mirrored_y = transform_curve(link2_mirror_x, link2_mirror_y, (link2_mirror_x[0], link2_mirror_y[0]), (link2_mirror_x[-1], link2_mirror_y[-1]), (points[6], points[7]), (points[4], points[5]))
    #linkc_mirrored_x, linkc_mirrored_y = transform_curve(linkc_mirror_x, linkc_mirror_y, (linkc_mirror_x[0], linkc_mirror_y[0]), (linkc_mirror_x[-1], linkc_mirror_y[-1]), (points[4], points[5]), (offsetpt[0], offsetpt[1]))

    #airfoil_coordinates_x = np.concatenate([airfoil_coordinates[0:index_3-1,0], np.flip(link2_mirrored_x), np.flip(link1_mirrored_x), bezier_x_transpose[1:-1]])
    airfoil_coordinates_x = np.concatenate([airfoil_coordinates[0:index_3,0], np.flip(link2_mirrored_x), np.flip(link1_mirrored_x), bezier_x_transpose[1:-1]])
    airfoil_coordinates_y = np.concatenate([airfoil_coordinates[0:index_3,1], np.flip(link2_mirrored_y), np.flip(link1_mirrored_y), bezier_y_transpose[1:-1]])
    #airfoil_coordinates_y = np.concatenate([airfoil_coordinates[0:index_3-1,1], np.flip(link2_mirrored_y), np.flip(link1_mirrored_y), bezier_y_transpose[1:-1]])

    airfoil_coordinates = np.column_stack((airfoil_coordinates_x, airfoil_coordinates_y))

    # Find the index of the rightmost point
    rightmost_index = np.argmax(airfoil_coordinates[:, 0])

    # Reorder the array
    reordered_array = np.concatenate([airfoil_coordinates[rightmost_index:], airfoil_coordinates[:rightmost_index]])

    # Now, the rightmost point is both at the start and the end
    #reordered_array_with_ends = np.vstack([reordered_array, reordered_array[0]])
    reordered_array_with_ends = np.vstack([reordered_array, reordered_array[0]])

    # Display the plot if the flag is set to True
    if display_plot:
        plt.figure(figsize=(10, 5))
        plt.scatter(reordered_array_with_ends[:,0], reordered_array_with_ends[:,1], label="Airfoil Shape")
        plt.scatter(x, y, color='red', label="Control Points")
        plt.axis("equal")
        plt.legend()
        plt.title("Generated Airfoil Shape")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    if 0:
        plt.figure(figsize=(10, 5))
        plt.plot(reordered_array_with_ends[:,0], label="Airfoil Shape")
        #plt.plot(np.flip(link1_mirrored_x), label="Airfoil Shape")
        plt.ylim(-1, 2) 
        plt.legend()
        plt.show()

    # Export the airfoil shape to a .dat file if the flag is set to True
    if export_dat:
        with open("airfoil_test.dat", "w") as f:
            f.write("Airfoil Shape Data\n")
            for xi, yi in zip(reordered_array_with_ends[:,0], reordered_array_with_ends[:,1]):
                f.write(f"{xi:.6f} {yi:.6f}\n")
        print('Dat file successfully written')

    return reordered_array_with_ends

def mirror_points(x, y, mirror_axis='vertical'):
    """
    Mirror points across the center of the curve.
    
    Parameters:
    - x, y: Arrays of x and y coordinates
    - mirror_axis: 'vertical' for mirroring across the curve's center y-axis, 
                   'horizontal' for mirroring across the curve's center x-axis
    
    Returns:
    - x_mirrored, y_mirrored: Mirrored coordinates
    """
    # Compute the center of the curve
    x_c = np.mean(x)
    y_c = np.mean(y)
    
    if mirror_axis == 'vertical':
        x_mirrored = 2 * x_c - x  # Reflect x about the center x_c
        y_mirrored = y
    elif mirror_axis == 'horizontal':
        x_mirrored = x
        y_mirrored = 2 * y_c - y  # Reflect y about the center y_c
    else:
        raise ValueError("Unsupported mirror axis. Use 'vertical' or 'horizontal'.")
    
    return x_mirrored, y_mirrored

def transform_curve(x, y, old_start, old_end, new_start, new_end):
    """
    Calculates and applies a translation and rotation to transform the old curve to the desired location.
    
    Parameters:
    - x, y: Arrays of x and y coordinates to be rotated and translated
    - old_start, old_end: Old start and end points as (x, y) tuples
    - new_start, new_end: New start and end points as (x, y) tuples
    
   Returns:
    - x_new, y_new: Transformed coordinates
    """
    
    # Calculate vectors for the old and new lines
    old_vector = (old_end[0] - old_start[0], old_end[1] - old_start[1])
    new_vector = (new_end[0] - new_start[0], new_end[1] - new_start[1])
    
    # Calculate the rotation angle using atan2
    old_angle = np.arctan2(old_vector[1], old_vector[0])
    new_angle = np.arctan2(new_vector[1], new_vector[0])
    rotation_angle = new_angle - old_angle

    pivot = (x[0], y[0])  # Rotate around the first point

    x_pivot, y_pivot = pivot

    # Translate points to origin
    x_translated = x - x_pivot
    y_translated = y - y_pivot

    # Compute rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    # Apply rotation
    x_int = cos_theta * x_translated - sin_theta * y_translated
    y_int = sin_theta * x_translated + cos_theta * y_translated

    # Translate back
    x_int += x_pivot
    y_int += y_pivot

    # Rotate
    # x_int = cos_theta * x - sin_theta * y
    # y_int = sin_theta * x + cos_theta * y

    # Calculate translation amounts
    dx = new_start[0] - x_int[0]
    dy = new_start[1] - y_int[0]

    # Apply the translation to the rotated curve
    x_new = x_int + dx
    y_new = y_int + dy

    return x_new, y_new

if __name__ == "__main__":
    #linkage_coordinates = [-1/2, 1/4, 1/4, 1/2, 3/4, 1/2, 3/2, 1/4, 1/4] # Some random value
    #linkage_coordinates = [-0.25, 0.25, 0.25, 0.5, 1/2+1/(2*np.sqrt(2)), 1/4, 1+1/(2*np.sqrt(2)), 0, 1/10] # Some random value
    linkage_coordinates = [-0.2257,  0.293 ,  0.2837,  0.3876,  0.9008,  0.3173,  1.3673, 0.0292,  0.1647]
    #linkage_coordinates = [-0.252, 0.160, -0.01, 0.304, 1.069, 0.09, 1.288, -0.084, 0.05]
    #linkage_coordinates = [-0.252, 0.160, -0.01, 0.304, 1.069, 0.09, 1.288, -0.084, 0.05, 0.1]
    #linkage_coordinates = [-0.39677463,  0.10094373,  0.14537092,  0.28675162,  0.87894111,  0.16552976, 1.40441522, -0.0632132,   0.19109849]
    #linkage_coordinates = [-0.23843949,  0.18377234,  0.15978613,  0.37020255,  0.88036828,  0.14023597,
  #1.30081941,  0.01154749,  0.16441546,  0.18385662]
    #linkage_coordinates = [-0.3303333,   0.13226088,  0.27309111,  0.37596664,  0.77091291,  0.32440575, 1.35164858,  0.01761675,  0.27045064]
    airfoil_coordinates = generate_airfoil_dat_from_points(linkage_coordinates, trailing_edge_smooth_factor=0.1)
    #airfoil_coordinates_x, airfoil_coordinates_y  = mirror_linkages(points = linkage_coordinates, x_smooth=airfoil_coordinates_unsymmetric[:,0], y_smooth=airfoil_coordinates_unsymmetric[:,1])

    if 1:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.1,0.1,0.5,0.8])
        #ax.scatter(range(len(airfoil_coordinates[:, 0])), airfoil_coordinates[:,0])
        ax.scatter(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], color="red")
        ax.scatter([linkage_coordinates[0], linkage_coordinates[2], linkage_coordinates[4], linkage_coordinates[6]], [linkage_coordinates[1], linkage_coordinates[3], linkage_coordinates[5], linkage_coordinates[7]])
        #ax.scatter(airfoil_coordinates_unsymmetric[100:200, 0], airfoil_coordinates_unsymmetric[100:200, 1], color="black")
        #ax.plot(airfoil_coordinates_unsymmetric[200:-1, 0], airfoil_coordinates_unsymmetric[200:-1, 1], color="blue")
        plt.axis("equal")
        #plt.ylim(-2, 2) 
        plt.grid()
        plt.show()
