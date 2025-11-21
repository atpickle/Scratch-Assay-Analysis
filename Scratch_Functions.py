def apply_threshold(folder_path):
    import os
    import cv2
    from skimage import io
    from skimage.filters import threshold_li
    import numpy as np

    """
    Apply Li's or Otsu's threshold to all TIFF images in a folder and save the thresholded images.
    Apply Li's threshold if the image name ends in _C_1 or _C_2, otherwise apply Otsu's threshold.
    Skip images that end in _C_5.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.

    Returns
    -------
    None
    """
    # Get a list of all TIFF files in the folder
    tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tif') or file.endswith('.tiff')]

    # Create the "threshold" folder if it doesn't exist
    threshold_folder = os.path.join(folder_path, "Threshold")
    os.makedirs(threshold_folder, exist_ok=True)

    # Iterate over each TIFF file
    for tiff_file in tiff_files:
        # Skip images that end in _C_5
        if tiff_file.endswith('_C_5.tif') or tiff_file.endswith('_C_5.tiff'):
            print(f"Skipping {tiff_file}")
            continue

        # Construct the full path to the TIFF file
        tiff_path = os.path.join(folder_path, tiff_file)

        # Read the image
        image = io.imread(tiff_path, as_gray=True)

        # Ensure the image is in the correct range [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Apply Li's thresholding if the image name ends in _C_1 or _C_2
        if tiff_file.endswith('_C_1.tif') or tiff_file.endswith('_C_1.tiff') or tiff_file.endswith('_C_2.tif') or tiff_file.endswith('_C_2.tiff'):
            li_thresh = threshold_li(image)
            thresholded_image = (image > li_thresh).astype(np.uint8) * 255
        else:
            # Apply Otsu's thresholding for the rest
            _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save the thresholded image in the "threshold" folder with the same filename
        thresholded_image_path = os.path.join(threshold_folder, tiff_file)
        io.imsave(thresholded_image_path, thresholded_image)



def Remove_Islands(folder_path, microglial_disk, astrocyte_disk, neuronal_disk, DAPI_disk, save, display):
    import os
    import cv2
    import numpy as np
    from skimage.morphology import disk, white_tophat
    from skimage import io
    import matplotlib.pyplot as plt
    """
    Remove islands from images based on the specified radius.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    radius_microglia : int
        Radius of the disk-shaped structuring element for microglia images.
    radius_astrocytes : int
        Radius of the disk-shaped structuring element for astrocytes images.
    radius_neurons : int
        Radius of the disk-shaped structuring element for neurons images.
    radius_dapi : int
        Radius of the disk-shaped structuring element for DAPI images.
    save : bool
        Whether to save the processed images.
    display : bool
        Whether to display the processed images.

    Returns
    -------
    None
    """
    # Get a list of all TIFF files in the folder
    tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tif') or file.endswith('.tiff')]

    # Create the "processed" folder if it doesn't exist
    processed_folder = os.path.join(folder_path, "0_Threshold_Processed")
    os.makedirs(processed_folder, exist_ok=True)

    # Iterate over each TIFF file
    for tiff_file in tiff_files:
        # Construct the full path to the TIFF file
        tiff_path = os.path.join(folder_path, tiff_file)

        # Read the image
        image = io.imread(tiff_path, as_gray=True)

        # Determine the appropriate radius based on the file suffix
        if tiff_file.endswith("_C_0.tiff") or tiff_file.endswith("_C_0.tif"):
            radius = microglial_disk
        elif tiff_file.endswith("_C_1.tiff") or tiff_file.endswith("_C_1.tif"):
            radius = astrocyte_disk
        elif tiff_file.endswith("_C_2.tiff") or tiff_file.endswith("_C_2.tif"):
            radius = neuronal_disk
        elif tiff_file.endswith("_C_4.tiff") or tiff_file.endswith("_C_4.tif"):
            radius = DAPI_disk
        else:
            continue  # Skip files that do not match the required suffixes

        # Apply white tophat filter to remove small islands
        footprint = disk(radius)
        tophat_image = white_tophat(image, footprint)

        # Subtract the tophat image from the original image
        final_image = image - tophat_image
        
        if save:
            # Save the final processed image in the "processed" folder with the same filename
            processed_image_path = os.path.join(processed_folder, tiff_file)
            io.imsave(processed_image_path, final_image)

        if display:
            # Display original, tophat, and final images side by side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(tophat_image, cmap='gray')
            axes[1].set_title('Tophat')
            axes[1].axis('off')

            axes[2].imshow(final_image, cmap='gray')
            axes[2].set_title('Final')
            axes[2].axis('off')

            plt.show()

def process_image(threshold_image_path, min_size, gap_size, output_folder, bisecting_lines_data, bisecting_lines_coords, display_results=False, save_results=True):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from matplotlib_scalebar.scalebar import ScaleBar
    """
    Process the threshold image by removing small islands, filling gaps in black regions,
    and filling all regions except for the largest empty region.

    Parameters
    ----------
    threshold_image_path : str
        Path to the threshold image.
    min_size : int
        Minimum size of islands to keep.
    gap_size : int
        The size of the gaps to fill.
    output_folder : str
        Path to the folder where the generated images will be saved.
    bisecting_lines_data : list
        List to collect bisecting line data for all images.
    bisecting_lines_coords : dict
        Dictionary to store bisecting line coordinates for images with the same base name.
    display_results : bool, optional
        Whether to display the images. Default is False.
    save_results : bool, optional
        Whether to save the images. Default is True.

    Returns
    -------
    None
    """
    def remove_islands(image, min_size):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
        cleaned_image = np.zeros_like(image)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_image[labels == i] = 255
        return cleaned_image

    def fill_gaps_in_black_regions(image, gap_size):
        kernel = np.ones((gap_size, gap_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def fill_except_largest_empty_region(image):
        inverted_image = cv2.bitwise_not(image)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        mask = np.zeros_like(image)
        if largest_contour is not None:
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        result_image = cv2.bitwise_not(mask)
        invert_image = cv2.bitwise_not(result_image)
        contours, _ = cv2.findContours(invert_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_contour = contour
        result_image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if best_contour is not None:
            cv2.drawContours(result_image_colored, [best_contour], -1, (0, 255, 0), 2)
        return result_image, result_image_colored, max_area, largest_contour

    def extend_line_to_edges(pt1, pt2, image_shape):
        height, width = image_shape[:2]
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 == x2:
            return (x1, 0), (x2, height - 1)
        elif y1 == y2:
            return (0, y1), (width - 1, y2)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y_at_x0 = intercept
        y_at_xmax = slope * (width - 1) + intercept
        x_at_y0 = -intercept / slope
        x_at_ymax = (height - 1 - intercept) / slope
        points = []
        if 0 <= y_at_x0 < height:
            points.append((0, int(y_at_x0)))
        if 0 <= y_at_xmax < height:
            points.append((width - 1, int(y_at_xmax)))
        if 0 <= x_at_y0 < width:
            points.append((int(x_at_y0), 0))
        if 0 <= x_at_ymax < width:
            points.append((int(x_at_ymax), height - 1))
        if len(points) >= 2:
            return points[0], points[1]
        else:
            return pt1, pt2

    original_image = cv2.imread(threshold_image_path, cv2.IMREAD_GRAYSCALE)
    original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cleaned_image = remove_islands(original_image, min_size)
    if save_results:
        cleaned_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_islands_removed.tif')
        cv2.imwrite(cleaned_image_path, cleaned_image)
    filled_image = fill_gaps_in_black_regions(cleaned_image, gap_size)
    if save_results:
        filled_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_filled.tif')
        cv2.imwrite(filled_image_path, filled_image)
    result_image, result_image_colored, largest_contour_area, largest_contour = fill_except_largest_empty_region(filled_image)
    image_name = os.path.basename(threshold_image_path)
    base_name = image_name.split('_C_')[0]
    if image_name.endswith('_C_1.tif') or image_name.endswith('_C_1.tiff'):
        if largest_contour is not None:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            longest_side = max(rect[1])
            angle = rect[2]
            center = (int(rect[0][0]), int(rect[0][1]))
            angle_rad = np.deg2rad(angle if longest_side == rect[1][0] else angle + 90)
            line_length = int(longest_side / 2)
            x_offset = int(line_length * np.cos(angle_rad))
            y_offset = int(line_length * np.sin(angle_rad))
            pt1 = (center[0] - x_offset, center[1] - y_offset)
            pt2 = (center[0] + x_offset, center[1] + y_offset)
            pt1, pt2 = extend_line_to_edges(pt1, pt2, original_image.shape)
            bisecting_lines_coords[base_name] = (pt1, pt2)
            cv2.line(result_image_colored, pt1, pt2, (0, 0, 255), 2)
            cv2.line(original_image_colored, pt1, pt2, (0, 0, 255), 2)
            bisecting_lines_data.append({
                'base_name': base_name,
                'pt1_x': pt1[0],
                'pt1_y': pt1[1],
                'pt2_x': pt2[0],
                'pt2_y': pt2[1]
            })
    if save_results:
        result_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_contour.tif')
        cv2.imwrite(result_image_path, result_image_colored)
        original_image_bisected_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_bisected.tif')
        cv2.imwrite(original_image_bisected_path, original_image_colored)
    
    if display_results:
        # Display the images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(filled_image, cmap='gray')
        axes[1].set_title('Filled Image')
        axes[1].axis('off')

        axes[2].imshow(result_image_colored)
        axes[2].set_title('Result Image')
        axes[2].axis('off')

        plt.show()

def apply_bisecting_lines(folder_path, bisecting_lines_coords, output_folder, display_results=False, save_results=True):
    for filename in os.listdir(folder_path):
        if not filename.endswith('_C_1.tif') and not filename.endswith('_C_1.tiff') and (filename.endswith('.tif') or filename.endswith('.tiff')):
            image_path = os.path.join(folder_path, filename)
            image_name = os.path.basename(image_path)
            base_name = image_name.split('_C_')[0]
            if base_name in bisecting_lines_coords:
                pt1, pt2 = bisecting_lines_coords[base_name]
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                cv2.line(original_image_colored, pt1, pt2, (0, 0, 255), 2)
                if save_results:
                    original_image_bisected_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_bisected.tif')
                    cv2.imwrite(original_image_bisected_path, original_image_colored)
                if display_results:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(original_image_colored)
                    plt.title(f"Bisected Image: {filename}")
                    plt.axis('off')
                    plt.show()

def process_folder(folder_path, min_size, gap_size, display_results=False, save_results=True):
    import os
    bisecting_lines_data = []
    bisecting_lines_coords = {}
    output_folder = os.path.join(folder_path, "Processed_Results")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            if filename.endswith('_C_1.tif') or filename.endswith('_C_1.tiff'):
                process_image(image_path, min_size, gap_size, output_folder, bisecting_lines_data, bisecting_lines_coords, display_results, save_results)
    apply_bisecting_lines(folder_path, bisecting_lines_coords, output_folder, display_results, save_results)
    bisecting_lines_df = pd.DataFrame(bisecting_lines_data)
    if save_results:
        excel_path = os.path.join(output_folder, "Bisect_Calculator_Bisecting_Lines.xlsx")
        bisecting_lines_df.to_excel(excel_path, index=False)




# def calculate_contour_areas(image_path, pt1, pt2, line_distance_um, scale_um_per_pixel, output_folder, results, display_results=False, save_results=True):
#     import cv2
#     import numpy as np
#     import pandas as pd
#     import os
#     import matplotlib.pyplot as plt
#     """
#     Calculate the percentage of white to black areas within the parallel line boundary for images ending in _C_0.

#     Parameters
#     ----------
#     image_path : str
#         Path to the image.
#     pt1 : tuple
#         The first point of the line.
#     pt2 : tuple
#         The second point of the line.
#     line_distance_um : float
#         Distance between the center line and the parallel lines in micrometers.
#     scale_um_per_pixel : float
#         Scale in micrometers per pixel.
#     output_folder : str
#         Path to the folder where the generated images will be saved.
#     results : dict
#         Dictionary to collect the percentage of white to black areas and the total contour area for each image.
#     display_results : bool, optional
#         Whether to display the images. Default is False.
#     save_results : bool, optional
#         Whether to save the images. Default is True.

#     Returns
#     -------
#     None
#     """
#     def measure_contour_areas_within_regions(image, pt1_parallel1, pt2_parallel1, pt1_parallel2, pt2_parallel2):
#         """
#         Calculate the area of each contour within the regions defined by the parallel lines.

#         Parameters
#         ----------
#         image : numpy.ndarray
#             The input image.
#         pt1_parallel1 : tuple
#             The first point of the first parallel line.
#         pt2_parallel1 : tuple
#             The second point of the first parallel line.
#         pt1_parallel2 : tuple
#             The first point of the second parallel line.
#         pt2_parallel2 : tuple
#             The second point of the second parallel line.

#         Returns
#         -------
#         contour_area_percentage : float
#             The percentage of white to black areas within the regions.
#         total_contour_area_microns : float
#             The total area of the contours within the regions in square micrometers.
#         overlay_image : numpy.ndarray
#             The image with the overlay of the counted contours.
#         """
#         # Create a mask for the regions between the parallel lines
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         cv2.line(mask, pt1_parallel1, pt2_parallel1, 255, 2)
#         cv2.line(mask, pt1_parallel2, pt2_parallel2, 255, 2)
#         cv2.fillPoly(mask, [np.array([pt1_parallel1, pt2_parallel1, pt2_parallel2, pt1_parallel2])], 255)

#         # Apply the mask to the image
#         masked_image = cv2.bitwise_and(image, image, mask=mask)

#         # Threshold the masked image to obtain a binary mask
#         _, binary_mask = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY)

#         # Find contours in the binary mask
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Calculate the total area of the contours in square micrometers
#         total_contour_area_pixels = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0)
#         total_contour_area_microns = total_contour_area_pixels * (scale_um_per_pixel ** 2)

#         # Calculate the total area within the parallel lines in square micrometers
#         total_area_within_lines_pixels = cv2.contourArea(np.array([pt1_parallel1, pt2_parallel1, pt2_parallel2, pt1_parallel2]))
#         total_area_within_lines_microns = total_area_within_lines_pixels * (scale_um_per_pixel ** 2)

#         # Calculate the percentage of white to black areas within the regions
#         contour_area_percentage = (total_contour_area_microns / total_area_within_lines_microns) * 100 if total_area_within_lines_microns > 0 else 0

#         # Overlay the counted contours with bright yellow on the original image
#         overlay = np.zeros_like(image)
#         overlay[binary_mask > 0] = 255  # Set white pixels to 255 in the overlay
#         overlay_colored = cv2.merge([overlay, overlay, np.zeros_like(overlay)])  # Create a yellow overlay
#         color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         overlay_image = cv2.addWeighted(color_image, 1, overlay_colored, 0.5, 0)

#         # Draw the contours on the overlay image
#         cv2.drawContours(overlay_image, contours, -1, (0, 255, 255), 2)  # Yellow color

#         return contour_area_percentage, total_contour_area_microns, overlay_image

#     # Convert the line distance from micrometers to pixels
#     line_distance_px = line_distance_um / scale_um_per_pixel

#     # Read the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image_name = os.path.basename(image_path)

#     # Calculate the direction vector of the line
#     direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
#     direction = direction / np.linalg.norm(direction)

#     # Calculate the perpendicular vector
#     perpendicular = np.array([-direction[1], direction[0]])

#     # Calculate the points for the parallel lines
#     pt1_parallel1 = (int(pt1[0] + line_distance_px * perpendicular[0]), int(pt1[1] + line_distance_px * perpendicular[1]))
#     pt2_parallel1 = (int(pt2[0] + line_distance_px * perpendicular[0]), int(pt2[1] + line_distance_px * perpendicular[1]))
#     pt1_parallel2 = (int(pt1[0] - line_distance_px * perpendicular[0]), int(pt1[1] - line_distance_px * perpendicular[1]))
#     pt2_parallel2 = (int(pt2[0] - line_distance_px * perpendicular[0]), int(pt2[1] - line_distance_px * perpendicular[1]))

#     # Draw the parallel lines on the image
#     color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     cv2.line(color_image, pt1_parallel1, pt2_parallel1, (0, 0, 255), 10)  # Red color
#     cv2.line(color_image, pt1_parallel2, pt2_parallel2, (0, 0, 255), 10)  # Red color

#     # Measure the contour areas within the regions and get the overlay image
#     contour_area_percentage, total_contour_area_microns, overlay_image = measure_contour_areas_within_regions(image, pt1_parallel1, pt2_parallel1, pt1_parallel2, pt2_parallel2)

#     # Save the result to the results dictionary
#     results[image_name] = {
#         'Contour Area Percentage': contour_area_percentage,
#         'Total Contour Area (sq microns)': total_contour_area_microns
#     }

#     # Create the "Quantified Area" folder if it doesn't exist
#     quantified_area_folder = os.path.join(output_folder, "Quantified Area")
#     os.makedirs(quantified_area_folder, exist_ok=True)

#     # Save the overlay image to the "Quantified Area" folder
#     if save_results:
#         overlay_image_path = os.path.join(quantified_area_folder, image_name)
#         cv2.imwrite(overlay_image_path, overlay_image)

#     # Display the image with parallel lines and the overlay image
#     if display_results:
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
#         plt.title('Image with Parallel Lines')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
#         plt.title('Overlay Image')
#         plt.axis('off')

#         plt.show()

# def process_folder_for_contour_areas(folder_path, line_distance_um, scale_um_per_pixel, display_results=False, save_results=True):
#     """
#     Process all images in a folder by calculating the percentage of white to black areas within the parallel line boundary.

#     Parameters
#     ----------
#     folder_path : str
#         Path to the folder containing the images.
#     line_distance_um : float
#         Distance between the center line and the parallel lines in micrometers.
#     scale_um_per_pixel : float
#         Scale in micrometers per pixel.
#     display_results : bool, optional
#         Whether to display the images. Default is False.
#     save_results : bool, optional
#         Whether to save the images. Default is True.

#     Returns
#     -------
#     None
#     """
#     results = {}

#     # Set the path to the Excel file containing the coordinates
#     excel_path = os.path.join(folder_path, 'Processed_Results', 'Bisect_Calculator_Bisecting_Lines.xlsx')

#     # Read the Excel file
#     coordinates_df = pd.read_excel(excel_path)

#     # Iterate over all images in the main directory
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.tif') or filename.endswith('.tiff'):
#             image_path = os.path.join(folder_path, filename)
#             image_name = os.path.basename(image_path)
#             base_name = '_'.join(image_name.split('_')[:-2])

#             # Find the coordinates for the current image
#             line_data = coordinates_df[coordinates_df['base_name'] == base_name]
#             if line_data.empty:
#                 continue

#             pt1 = (int(line_data['pt1_x'].values[0]), int(line_data['pt1_y'].values[0]))
#             pt2 = (int(line_data['pt2_x'].values[0]), int(line_data['pt2_y'].values[0]))

#             # Process the image
#             calculate_contour_areas(image_path, pt1, pt2, line_distance_um, scale_um_per_pixel, folder_path, results, display_results, save_results)

#     # Convert the results dictionary to a DataFrame
#     results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
#     results_df.columns = ['Image Name', 'Contour Area Percentage', 'Total Contour Area (sq microns)']

#     # Save the results to an Excel file
#     if save_results:
#         results_excel_path = os.path.join(folder_path, 'contour_areas.xlsx')
#         results_df.to_excel(results_excel_path, index=False)

# def calculate_black_area(folder_path, scale_um_per_pixel=0.641, output_excel='Total Scratch Area.xlsx', display_images=False, save_images=True):
#     import os
#     import numpy as np
#     from skimage import io
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from skimage.measure import find_contours
#     """
#     Iterate through all .tif and .tiff images in the specified folder.
#     If the image name ends with "_filled", calculate the area of the black regions in microns,
#     display the image with a contour line around the black area, and save the results to an Excel sheet.

#     Parameters
#     ----------
#     folder_path : str
#         Path to the folder containing the TIFF images.
#     scale_um_per_pixel : float
#         Conversion factor from pixels to microns.
#     output_excel : str
#         Name of the output Excel file.
#     display_images : bool, optional
#         Whether to display the images with contour lines. Default is False.
#     save_images : bool, optional
#         Whether to save the images with contour lines. Default is True.

#     Returns
#     -------
#     None
#     """
#     results = []

#     # Create the "Contours" folder if it doesn't exist
#     contours_folder = os.path.join(folder_path, "Contours")
#     os.makedirs(contours_folder, exist_ok=True)

#     # Iterate over all files in the folder
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith("_filled.tif") or file_name.endswith("_filled.tiff"):
#             # Construct the full path to the image file
#             image_path = os.path.join(folder_path, file_name)

#             # Read the image
#             image = io.imread(image_path)

#             # Calculate the area of the black regions (assuming black is represented by 0)
#             black_area_pixels = np.sum(image == 0)

#             # Convert the area from pixels to microns
#             black_area_microns = black_area_pixels * (scale_um_per_pixel ** 2)

#             # Append the result to the list
#             results.append([file_name, black_area_microns])

#             # Find contours at a constant value of 0.5
#             contours = find_contours(image, level=0.5)

#             # Display or save the image with contour lines
#             plt.figure(figsize=(8, 8))
#             plt.imshow(image, cmap='gray')
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
#             plt.title(f"Image: {file_name}\nBlack Area: {black_area_microns:.2f} square microns")
#             plt.axis('off')

#             if display_images:
#                 plt.show()

#             if save_images:
#                 contour_image_path = os.path.join(contours_folder, file_name.replace('.tif', '_contour.png').replace('.tiff', '_contour.png'))
#                 plt.savefig(contour_image_path, bbox_inches='tight')
#                 plt.close()

#     # Create a DataFrame from the results
#     df = pd.DataFrame(results, columns=['Image Name', 'Black Area (square microns)'])

#     # Construct the full path to the output Excel file
#     output_excel_path = os.path.join(folder_path, output_excel)

#     # Save the DataFrame to an Excel file
#     df.to_excel(output_excel_path, index=False)

#     print(f"Results saved to {output_excel_path}")

    
def process_image(threshold_image_path, min_size, gap_size, output_folder, bisecting_lines_data, bisecting_lines_coords, display_results=False, save_results=True):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from matplotlib_scalebar.scalebar import ScaleBar

    """
    Process the threshold image by removing small islands, filling gaps in black regions,
    and filling all regions except for the largest empty region.

    Parameters
    ----------
    threshold_image_path : str
        Path to the threshold image.
    min_size : int
        Minimum size of islands to keep.
    gap_size : int
        The size of the gaps to fill.
    output_folder : str
        Path to the folder where the generated images will be saved.
    bisecting_lines_data : list
        List to collect bisecting line data for all images.
    bisecting_lines_coords : dict
        Dictionary to store bisecting line coordinates for images with the same base name.
    display_results : bool, optional
        Whether to display the images. Default is False.
    save_results : bool, optional
        Whether to save the images. Default is True.

    Returns
    -------
    None
    """
    def remove_islands(image, min_size):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
        cleaned_image = np.zeros_like(image)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_image[labels == i] = 255
        return cleaned_image

    def fill_gaps_in_black_regions(image, gap_size):
        kernel = np.ones((gap_size, gap_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def fill_except_largest_empty_region(image):
        inverted_image = cv2.bitwise_not(image)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        mask = np.zeros_like(image)
        if largest_contour is not None:
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        result_image = cv2.bitwise_not(mask)
        invert_image = cv2.bitwise_not(result_image)
        contours, _ = cv2.findContours(invert_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                best_contour = contour
        result_image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if best_contour is not None:
            cv2.drawContours(result_image_colored, [best_contour], -1, (0, 255, 0), 2)
        return result_image, result_image_colored, max_area, largest_contour

    def extend_line_to_edges(pt1, pt2, image_shape):
        height, width = image_shape[:2]
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 == x2:
            return (x1, 0), (x2, height - 1)
        elif y1 == y2:
            return (0, y1), (width - 1, y2)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y_at_x0 = intercept
        y_at_xmax = slope * (width - 1) + intercept
        x_at_y0 = -intercept / slope
        x_at_ymax = (height - 1 - intercept) / slope
        points = []
        if 0 <= y_at_x0 < height:
            points.append((0, int(y_at_x0)))
        if 0 <= y_at_xmax < height:
            points.append((width - 1, int(y_at_xmax)))
        if 0 <= x_at_y0 < width:
            points.append((int(x_at_y0), 0))
        if 0 <= x_at_ymax < width:
            points.append((int(x_at_ymax), height - 1))
        if len(points) >= 2:
            return points[0], points[1]
        else:
            return pt1, pt2

    original_image = cv2.imread(threshold_image_path, cv2.IMREAD_GRAYSCALE)
    original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cleaned_image = remove_islands(original_image, min_size)
    if save_results:
        cleaned_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_islands_removed.tif')
        cv2.imwrite(cleaned_image_path, cleaned_image)
    filled_image = fill_gaps_in_black_regions(cleaned_image, gap_size)
    if save_results:
        filled_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_filled.tif')
        cv2.imwrite(filled_image_path, filled_image)
    result_image, result_image_colored, largest_contour_area, largest_contour = fill_except_largest_empty_region(filled_image)
    image_name = os.path.basename(threshold_image_path)
    base_name = image_name.split('_C_')[0]
    if image_name.endswith('_C_2.tif') or image_name.endswith('_C_2.tiff'):
        if largest_contour is not None:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            longest_side = max(rect[1])
            angle = rect[2]
            center = (int(rect[0][0]), int(rect[0][1]))
            angle_rad = np.deg2rad(angle if longest_side == rect[1][0] else angle + 90)
            line_length = int(longest_side / 2)
            x_offset = int(line_length * np.cos(angle_rad))
            y_offset = int(line_length * np.sin(angle_rad))
            pt1 = (center[0] - x_offset, center[1] - y_offset)
            pt2 = (center[0] + x_offset, center[1] + y_offset)
            pt1, pt2 = extend_line_to_edges(pt1, pt2, original_image.shape)
            bisecting_lines_coords[base_name] = (pt1, pt2)
            cv2.line(result_image_colored, pt1, pt2, (0, 0, 255), 2)
            cv2.line(original_image_colored, pt1, pt2, (0, 0, 255), 2)
            bisecting_lines_data.append({
                'base_name': base_name,
                'pt1_x': pt1[0],
                'pt1_y': pt1[1],
                'pt2_x': pt2[0],
                'pt2_y': pt2[1]
            })
    if save_results:
        result_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_contour.tif')
        cv2.imwrite(result_image_path, result_image_colored)
        original_image_bisected_path = os.path.join(output_folder, os.path.splitext(os.path.basename(threshold_image_path))[0] + '_bisected.tif')
        cv2.imwrite(original_image_bisected_path, original_image_colored)
    
    if display_results:
        # Display the images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(filled_image, cmap='gray')
        axes[1].set_title('Filled Image')
        axes[1].axis('off')

        axes[2].imshow(result_image_colored)
        axes[2].set_title('Result Image')
        axes[2].axis('off')

        plt.show()

def apply_bisecting_lines(folder_path, bisecting_lines_coords, output_folder, display_results=False, save_results=True):
    import os
    import pandas as pd
    import cv2
    import matplotlib.pyplot as plt
    for filename in os.listdir(folder_path):
        if not filename.endswith('_C_2.tif') and not filename.endswith('_C_2.tiff') and (filename.endswith('.tif') or filename.endswith('.tiff')):
            image_path = os.path.join(folder_path, filename)
            image_name = os.path.basename(image_path)
            base_name = image_name.split('_C_')[0]
            if base_name in bisecting_lines_coords:
                pt1, pt2 = bisecting_lines_coords[base_name]
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                cv2.line(original_image_colored, pt1, pt2, (0, 0, 255), 2)
                if save_results:
                    original_image_bisected_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_bisected.tif')
                    cv2.imwrite(original_image_bisected_path, original_image_colored)
                if display_results:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(original_image_colored)
                    plt.title(f"Bisected Image: {filename}")
                    plt.axis('off')
                    plt.show()

def process_folder(folder_path, min_size, gap_size, display_results=False, save_results=True):
    import os
    import pandas as pd

    bisecting_lines_data = []
    bisecting_lines_coords = {}
    output_folder = os.path.join(folder_path, "Processed_Results")
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            if filename.endswith('_C_2.tif') or filename.endswith('_C_2.tiff'):
                process_image(image_path, min_size, gap_size, output_folder, bisecting_lines_data, bisecting_lines_coords, display_results, save_results)
    apply_bisecting_lines(folder_path, bisecting_lines_coords, output_folder, display_results, save_results)
    bisecting_lines_df = pd.DataFrame(bisecting_lines_data)
    if save_results:
        excel_path = os.path.join(output_folder, "Bisect_Calculator_Bisecting_Lines.xlsx")
        bisecting_lines_df.to_excel(excel_path, index=False)






def calculate_black_area(folder_path, scale_um_per_pixel=0.641, output_excel='Total Scratch Area.xlsx', display_images=False, save_images=True):
    """
    Iterate through all .tif and .tiff images in the specified folder.
    If the image name ends with "_filled", calculate the area of the black regions in microns,
    display the image with a contour line around the black area, and save the results to an Excel sheet.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    scale_um_per_pixel : float
        Conversion factor from pixels to microns.
    output_excel : str
        Name of the output Excel file.
    display_images : bool, optional
        Whether to display the images with contour lines. Default is False.
    save_images : bool, optional
        Whether to save the images with contour lines. Default is True.

    Returns
    -------
    None
    """
    import os
    import numpy as np
    from skimage import io
    import pandas as pd
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    results = []

    # Create the "Contours" folder if it doesn't exist
    contours_folder = os.path.join(folder_path, "Contours")
    os.makedirs(contours_folder, exist_ok=True)

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_filled.tif") or file_name.endswith("_filled.tiff"):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, file_name)

            # Read the image
            image = io.imread(image_path)

            # Calculate the area of the black regions (assuming black is represented by 0)
            black_area_pixels = np.sum(image == 0)

            # Convert the area from pixels to microns
            black_area_microns = black_area_pixels * (scale_um_per_pixel ** 2)

            # Append the result to the list
            results.append([file_name, black_area_microns])

            # Find contours at a constant value of 0.5
            contours = find_contours(image, level=0.5)

            # Display or save the image with contour lines
            plt.figure(figsize=(8, 8))
            plt.imshow(image, cmap='gray')
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
            plt.title(f"Image: {file_name}\nBlack Area: {black_area_microns:.2f} square microns")
            plt.axis('off')

            if display_images:
                plt.show()

            if save_images:
                contour_image_path = os.path.join(contours_folder, file_name.replace('.tif', '_contour.png').replace('.tiff', '_contour.png'))
                plt.savefig(contour_image_path, bbox_inches='tight')
                plt.close()

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['Image Name', 'Black Area (square microns)'])

    # Construct the full path to the output Excel file
    output_excel_path = os.path.join(folder_path, output_excel)

    # Save the DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)

    print(f"Results saved to {output_excel_path}")






def process_folder_for_contour_areas(folder_path, line_distance_um, scale_um_per_pixel, display_results=False, save_results=True):
    """
    Process all images in a folder by calculating the percentage of white to black areas within the parallel line boundary.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the images.
    line_distance_um : float
        Distance between the center line and the parallel lines in micrometers.
    scale_um_per_pixel : float
        Scale in micrometers per pixel.
    display_results : bool, optional
        Whether to display the images. Default is False.
    save_results : bool, optional
        Whether to save the images. Default is True.

    Returns
    -------
    None
    """
    import os
    import pandas as pd
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    results = {}

    # Set the path to the Excel file containing the coordinates
    excel_path = os.path.join(folder_path, 'Processed_Results', 'Bisect_Calculator_Bisecting_Lines.xlsx')

    # Read the Excel file
    coordinates_df = pd.read_excel(excel_path)

    # Iterate over all images in the main directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            image_name = os.path.basename(image_path)
            base_name = '_'.join(image_name.split('_')[:-2])

            # Find the coordinates for the current image
            line_data = coordinates_df[coordinates_df['base_name'] == base_name]
            if line_data.empty:
                continue

            pt1 = (int(line_data['pt1_x'].values[0]), int(line_data['pt1_y'].values[0]))
            pt2 = (int(line_data['pt2_x'].values[0]), int(line_data['pt2_y'].values[0]))

            # Process the image
            calculate_contour_areas(image_path, pt1, pt2, line_distance_um, scale_um_per_pixel, folder_path, results, display_results, save_results)

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    results_df.columns = ['Image Name', 'Contour Area Percentage', 'Total Contour Area (sq microns)']

    # Save the results to an Excel file
    if save_results:
        results_excel_path = os.path.join(folder_path, 'contour_areas.xlsx')
        results_df.to_excel(results_excel_path, index=False)

def calculate_contour_areas(image_path, pt1, pt2, line_distance_um, scale_um_per_pixel, output_folder, results, display_results=False, save_results=True):
    """
    Calculate the percentage of white to black areas within the parallel line boundary for images ending in _C_0.

    Parameters
    ----------
    image_path : str
        Path to the image.
    pt1 : tuple
        The first point of the line.
    pt2 : tuple
        The second point of the line.
    line_distance_um : float
        Distance between the center line and the parallel lines in micrometers.
    scale_um_per_pixel : float
        Scale in micrometers per pixel.
    output_folder : str
        Path to the folder where the generated images will be saved.
    results : dict
        Dictionary to collect the percentage of white to black areas and the total contour area for each image.
    display_results : bool, optional
        Whether to display the images. Default is False.
    save_results : bool, optional
        Whether to save the images. Default is True.

    Returns
    -------
    None
    """
    import cv2
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    def measure_contour_areas_within_regions(image, pt1_parallel1, pt2_parallel1, pt1_parallel2, pt2_parallel2):
        """
        Calculate the area of each contour within the regions defined by the parallel lines.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.
        pt1_parallel1 : tuple
            The first point of the first parallel line.
        pt2_parallel1 : tuple
            The second point of the first parallel line.
        pt1_parallel2 : tuple
            The first point of the second parallel line.
        pt2_parallel2 : tuple
            The second point of the second parallel line.

        Returns
        -------
        contour_area_percentage : float
            The percentage of white to black areas within the regions.
        total_contour_area_microns : float
            The total area of the contours within the regions in square micrometers.
        overlay_image : numpy.ndarray
            The image with the overlay of the counted contours.
        """
        import cv2
        # Create a mask for the regions between the parallel lines
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.line(mask, pt1_parallel1, pt2_parallel1, 255, 2)
        cv2.line(mask, pt1_parallel2, pt2_parallel2, 255, 2)
        cv2.fillPoly(mask, [np.array([pt1_parallel1, pt2_parallel1, pt2_parallel2, pt1_parallel2])], 255)

        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Threshold the masked image to obtain a binary mask
        _, binary_mask = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the total area of the contours in square micrometers
        total_contour_area_pixels = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0)
        total_contour_area_microns = total_contour_area_pixels * (scale_um_per_pixel ** 2)

        # Calculate the total area within the parallel lines in square micrometers
        total_area_within_lines_pixels = cv2.contourArea(np.array([pt1_parallel1, pt2_parallel1, pt2_parallel2, pt1_parallel2]))
        total_area_within_lines_microns = total_area_within_lines_pixels * (scale_um_per_pixel ** 2)

        # Calculate the percentage of white to black areas within the regions
        contour_area_percentage = (total_contour_area_microns / total_area_within_lines_microns) * 100 if total_area_within_lines_microns > 0 else 0

        # Overlay the counted contours with bright yellow on the original image
        overlay = np.zeros_like(image)
        overlay[binary_mask > 0] = 255  # Set white pixels to 255 in the overlay
        overlay_colored = cv2.merge([overlay, overlay, np.zeros_like(overlay)])  # Create a yellow overlay
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay_image = cv2.addWeighted(color_image, 1, overlay_colored, 0.5, 0)

        # Draw the contours on the overlay image
        cv2.drawContours(overlay_image, contours, -1, (0, 255, 255), 2)  # Yellow color

        return contour_area_percentage, total_contour_area_microns, overlay_image

    # Convert the line distance from micrometers to pixels
    line_distance_px = line_distance_um / scale_um_per_pixel

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name = os.path.basename(image_path)

    # Calculate the direction vector of the line
    direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    direction = direction / np.linalg.norm(direction)

    # Calculate the perpendicular vector
    perpendicular = np.array([-direction[1], direction[0]])

    # Calculate the points for the parallel lines
    pt1_parallel1 = (int(pt1[0] + line_distance_px * perpendicular[0]), int(pt1[1] + line_distance_px * perpendicular[1]))
    pt2_parallel1 = (int(pt2[0] + line_distance_px * perpendicular[0]), int(pt2[1] + line_distance_px * perpendicular[1]))
    pt1_parallel2 = (int(pt1[0] - line_distance_px * perpendicular[0]), int(pt1[1] - line_distance_px * perpendicular[1]))
    pt2_parallel2 = (int(pt2[0] - line_distance_px * perpendicular[0]), int(pt2[1] - line_distance_px * perpendicular[1]))

    # Draw the parallel lines on the image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.line(color_image, pt1_parallel1, pt2_parallel1, (0, 0, 255), 10)  # Red color
    cv2.line(color_image, pt1_parallel2, pt2_parallel2, (0, 0, 255), 10)  # Red color

    # Measure the contour areas within the regions and get the overlay image
    contour_area_percentage, total_contour_area_microns, overlay_image = measure_contour_areas_within_regions(image, pt1_parallel1, pt2_parallel1, pt1_parallel2, pt2_parallel2)

    # Save the result to the results dictionary
    results[image_name] = {
        'Contour Area Percentage': contour_area_percentage,
        'Total Contour Area (sq microns)': total_contour_area_microns
    }

    # Create the "Quantified Area" folder if it doesn't exist
    quantified_area_folder = os.path.join(output_folder, "Quantified Area")
    os.makedirs(quantified_area_folder, exist_ok=True)

    # Save the overlay image to the "Quantified Area" folder
    if save_results:
        overlay_image_path = os.path.join(quantified_area_folder, image_name)
        cv2.imwrite(overlay_image_path, overlay_image)

    # Display the image with parallel lines and the overlay image
    if display_results:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Parallel Lines')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title('Overlay Image')
        plt.axis('off')

        plt.show()