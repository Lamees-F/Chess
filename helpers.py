import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image 

# Create a grid for spaces (reversed order)
def create_grid(grid_rows, grid_cols):
    spaces = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            spaces.append(f"{chr(72 - i)}{j + 1}")  # Start from 'H' (72 in ASCII)
    return spaces

# Function to get the board coordinates
def get_board_coordinates(xyxy):
    min_x = xyxy[:, 0].min().item()
    max_x = xyxy[:, 2].max().item()
    min_y = xyxy[:, 1].min().item()
    max_y = xyxy[:, 3].max().item()
    return min_x - 5, min_y - 5, max_x + 5, max_y + 5

def get_new_shape(xyxy, shape):
    min_x = xyxy[:, 0].min().item()
    max_x = xyxy[:, 2].max().item()
    min_y = xyxy[:, 1].min().item()
    max_y = xyxy[:, 3].max().item()
    
    width, height = shape
    lm = min_x
    rm = width - max_x
    bm = min_y
    tm = height - max_y
    return ((width - (lm + rm)), (height - (tm + bm))), lm, bm

def get_frame_new_shape(xyxy, shape):
    min_x = xyxy[:, 0].min().item()
    max_x = xyxy[:, 2].max().item()
    min_y = xyxy[:, 1].min().item()
    max_y = xyxy[:, 3].max().item()
    
    width, height = shape[0], shape[1]
    lm = min_x
    rm = width - max_x
    bm = min_y
    tm = height - max_y
    return ((width - (lm + rm)), (height - (tm + bm))), lm, bm

# Map detections to cells (reversed grid)
def map_detections_to_spaces(boxes, spaces, classes, frame_shape, grid_rows, grid_cols, lm, bm):
    # Initialize all spaces to "initial"
    occupancy = {space: "initial" for space in spaces}
    grids = {space: "" for space in spaces}
    index = 0
    row_ratio = grid_rows / frame_shape[0]
    col_ratio = grid_cols / frame_shape[1]

    row_size = 1 / row_ratio

    for box in boxes:
        # Calculate the center of the box
        # x_center = (box[0] + box[2]) / 2
        x_center = ((box[0] + box[2]) / 2) + ( 0.2 * row_size) - lm
        y_center = ((box[1] + box[3]) / 2) - bm

        # Calculate row and column indices
        row = int(y_center * row_ratio)
        col = int(x_center * col_ratio)

        # Clamp row and column indices to valid grid bounds
        row = max(0, min(row, grid_rows - 1))  # Ensure 0 <= row < grid_rows
        col = max(0, min(col, grid_cols - 1))  # Ensure 0 <= col < grid_cols

        # Map to a space identifier, reversed for rows
        space = f"{chr(72 - row)}{col + 1}"  # Reverse row index to start from 'H' (72)
        occupancy[space] = classes[index]
        grids[space] += f' box{index}: {classes[index]}'


        index += 1
        
    return occupancy
    

# Create the occupancy grid visualization
def create_occupancy_map(occupancy, grid_rows, grid_cols, map_shape=(480, 640, 3)):
    occ_map = np.full(map_shape, (25, 25, 75), dtype=np.uint8)

    for i, row in enumerate(range(65, 65 + grid_rows)):
        for j in range(grid_cols):
            space = f"{chr(72 - i)}{j + 1}"  # Reverse row to start from 'H' (72)
            x1 = 10 + j * (map_shape[1] - 20) // grid_cols
            y1 = 10 + i * (map_shape[0] - 20) // grid_rows
            x2 = 10 + (j + 1) * (map_shape[1] - 20) // grid_cols
            y2 = 10 + (i + 1) * (map_shape[0] - 20) // grid_rows

            # Define a mapping of class to color
            class_to_color = {
                "empty": (0, 255, 0),    # Green
                "black": (0, 0, 0),      # Black
                "white": (255, 0, 255),  # White
                "initial": (255, 100, 0) # Yellow
            }
            color = class_to_color.get(occupancy[space], (128, 128, 128))  # Default color

            cv2.rectangle(occ_map, (x1, y1), (x2, y2), color, -1)
            cv2.putText(occ_map, space, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return occ_map


# Get the board from the frame
def get_board_from_frame(frame, results):
    if results:
        detection_original = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        min_x, min_y, max_x, max_y = get_board_coordinates(boxes)
        cropped_frame = None
        # Crop the image to focus on the board
        cropped_frame = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        return cropped_frame, detection_original
    else:
        return None
    
# Get the board From image
def get_board(imagePath, model, conf_threshold):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Unable to load image from path: {imagePath}")

    results = model.predict(source=image, conf=conf_threshold)

    min_x, min_y, max_x, max_y = get_board_coordinates(results[0].boxes.xyxy)
    img = Image.open(imagePath)

    # Crop the image
    return img.crop((min_x, min_y, max_x, max_y))

def map_occupancy_to_board_status(occupancy):
    # Mapping rows and columns to a 2D array
    rows = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
    columns = ['1', '2', '3', '4', '5', '6', '7', '8']

    new_board_status = []

    for row in rows:
        row_data = []
        for col in columns:
            key = f"{row}{col}"
            cell_value = occupancy[key]
            row_data.append(cell_value)
        new_board_status.append(row_data)
    
    return new_board_status

def order_detections(boxes, classes):
    board_status = [] # Will be 2-d array that contains the classes
    detections = []

    # assuming the number of boxes is always 64
    index = 0
    for box in boxes:

        x_center = ((box[0] + box[2]) / 2)
        y_center = ((box[1] + box[3]) / 2)
        detections.append({'box': index, 'x_center': x_center, 'y_center': y_center, 'class': classes[index]})

        index += 1
        
    # Step 1: Sort detections by y_center to determine rows
    detections = sorted(detections, key=lambda d: d['y_center'])

    # Step 2: Divide detections into 8 rows
    rows = [detections[i * 8:(i + 1) * 8] for i in range(8)]

    # Step 3: Sort each row by x_center to determine columns
    for row in rows:
        sorted_row = sorted(row, key=lambda d: d['x_center'])
        board_status.append([cell['class'] for cell in sorted_row])

    return board_status