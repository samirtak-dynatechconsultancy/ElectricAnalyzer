# import os
# import cv2
# import numpy as np
# import pandas as pd

# def draw_connections_from_csv(csv_path, image_path, output_path):
#     try:
#         # Read CSV
#         df = pd.read_csv(csv_path)

#         # Load image
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Could not read image: {image_path}")

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Create white canvas same size as original
#         canvas = np.ones_like(img) * 255

#         for _, row in df.iterrows():
#             # Get connection points
#             p1 = (int(row["wire1_connection_point_x"]), int(row["wire1_connection_point_y"]))
#             p2 = (int(row["wire2_connection_point_x"]), int(row["wire2_connection_point_y"]))

#             # Draw line in black
#             cv2.line(canvas, p1, p2, (0, 0, 0), 2, cv2.LINE_AA)

#             # Mark start point in red
#             cv2.circle(canvas, p1, 4, (255, 0, 0), -1)  # Red

#             # Mark end point in green
#             cv2.circle(canvas, p2, 4, (0, 255, 0), -1)  # Green

#         # Ensure output directory exists
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Save image
#         cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
#         print(f"Image saved at: {output_path}")

#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage
# draw_connections_from_csv(
#     csv_path=r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\output_new_improved\page_10_junctions_connections_all_connections.csv",
#     image_path=r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\output_new_improved\page_10_junctions_junctions.png",
#     output_path=r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\output_new_improved\page_10_output_connections.png"
# )


# import cv2
# import numpy as np
# from itertools import combinations

# # Your input points
# points = [(833, 1450), (905, 1371), (905, 1605), (1098, 1371), (1275, 1596)]

# # Image paths
# image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\output_new_improved\page_5_junctions_junctions.png"
# output_path = "connections.png"

# # Load the image
# img = cv2.imread(image_path)
# if img is None:
#     raise FileNotFoundError(f"Could not load image: {image_path}")

# # Draw orange dots (BGR format → Orange = (0,140,255))
# for p in points:
#     cv2.circle(img, p, 6, (0, 140, 255), -1)

# # Draw blue lines (BGR format → Blue = (255,0,0))
# for p1, p2 in combinations(points, 2):
#     cv2.line(img, p1, p2, (255, 0, 0), 2)

# # Save and display
# cv2.imwrite(output_path, img)
# cv2.imshow("Connections", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# import cv2
# import numpy as np

# # Your input points
# points = [(3307, 84), (451, 1940), (905, 1680), (954, 1680), (1002, 1680), (1050, 1371), (1146, 1371)]

# # Image paths
# image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\output_new_improved\page_5_junctions_junctions.png"
# output_path = "connections.png"

# # Load the image
# img = cv2.imread(image_path)
# if img is None:
#     raise FileNotFoundError(f"Could not load image: {image_path}")

# # Draw orange dots (BGR format → Orange = (0,140,255))
# for p in points:
#     cv2.circle(img, p, 6, (0, 140, 255), -1)

# # Choose one point as the main source (here: first point)
# source_point = points[0]

# # Draw blue lines (BGR format → Blue = (255,0,0)) from source to all others
# for target_point in points[1:]:
#     cv2.line(img, source_point, target_point, (255, 0, 0), 2)

# # Save and display
# cv2.imwrite(output_path, img)
# cv2.imshow("Connections", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










import cv2
import numpy as np
import pandas as pd
import ast
import os

# Paths
# csv_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\wire_points.csv"  # <-- your CSV
# image_path = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\MDB\output\01) MDB\01) MDB_5_text_redacted.png"
# output_dir = r"C:\Users\Samir.Tak\Desktop\Workspace\Electrical Drawings\line-detection\MDB\output\01) MDB\lines"

# Make output folder
def draw_connections_from_df(wire_df, img):
    images_list = []  # Store all processed images here

    # Process each wire
    for _, row in wire_df.iterrows():
        copy_img = img.copy()  # Copy original image for each wire
        wire_name = row['Wire']
        points = row['Point']

        # Skip empty point lists
        if not points:
            print(f"Skipping {wire_name} (no points)")
            continue

        # Draw orange dots
        for p in points:
            cv2.circle(copy_img, tuple(p), 6, (0, 140, 255), -1)

        # Connect first point to all others in blue
        source_point = tuple(points[0])
        for target_point in points[1:]:
            cv2.line(copy_img, source_point, tuple(target_point), (0, 0, 255), 2)

        # Add to list
        images_list.append((wire_name, copy_img))

    return images_list
