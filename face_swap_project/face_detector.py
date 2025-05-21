# Script to detect faces and landmarks using qualcomm/MediaPipe-Face-Detection

import torch
from PIL import Image, ImageDraw
import numpy as np
# It's good practice to try importing qai_hub_models and Model early
# to catch installation issues sooner.
try:
    from qai_hub_models.models.mediapipe_face import Model
except ImportError:
    print("Error: qai_hub_models or its dependencies not found. Please install them.")
    Model = None # To prevent further errors if import fails

def detect_faces(image_path: str) -> list[dict]:
    """
    Detects faces and 6 facial landmarks in an image.

    Args:
        image_path: Path to the input image.

    Returns:
        A list of dictionaries, where each dictionary contains:
            'box': [x_min, y_min, x_max, y_max] (bounding box coordinates)
            'landmarks': [[lx1,ly1], ..., [lx6,ly6]] (6 facial landmarks)
    """
    if Model is None:
        print("MediaPipe-Face-Detection model could not be loaded. Aborting detection.")
        return []

    try:
        # Load the model
        model = Model.from_pretrained()
        print("MediaPipe-Face-Detection model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return []
    except Exception as e:
        print(f"Error opening image: {e}")
        return []

    # --- Preprocessing ---
    # Get input specification
    try:
        input_spec = model.get_input_spec()
        # Assuming 'image' is the key for image input, this might need adjustment
        # Expected shape is often (batch_size, channels, height, width)
        # Example: input_spec['image'][0] might be (1, 3, 256, 256)
        target_height = input_spec['image'][0][2]
        target_width = input_spec['image'][0][3]
        print(f"Model expects input shape: ({target_height}, {target_width})")
    except Exception as e:
        print(f"Error getting input_spec: {e}. Using default 256x256.")
        target_height, target_width = 256, 256


    # Resize image
    resized_img = img.resize((target_width, target_height))

    # Convert to PyTorch tensor and normalize
    # Normalization typically means scaling to [0, 1] and then possibly standardizing.
    # For many models, scaling to [0,1] is sufficient if it's a float tensor.
    # The model.sample_inputs() might give a clue or documentation is needed.
    # Let's assume normalization to [0,1] for now.
    img_np = np.array(resized_img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC to CHW
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # --- Inference ---
    print("Performing inference...")
    try:
        # Ensure model is in eval mode if it's a PyTorch nn.Module (good practice)
        # For qai_hub_models, this might be handled internally.
        # model.eval() 
        with torch.no_grad(): # Important for inference
            outputs = model(img_tensor)
        print("Inference completed.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # --- Postprocessing ---
    # This is the critical part: interpreting 'outputs'.
    # The structure of 'outputs' needs to be determined.
    # It could be a tensor, a list of tensors, or a dict of tensors.
    # Let's print its type and shape/structure first.
    print(f"Outputs type: {type(outputs)}")
    if isinstance(outputs, torch.Tensor):
        print(f"Outputs shape: {outputs.shape}")
        # Based on MediaPipe, output is often a flat list or a list of detections.
        # Example: NxM where N is number of detections, M is info per detection.
        # Info per detection: [box_coords (4), score (1), landmarks (6*2=12)]
        # Total M = 4 + 1 + 12 = 17 (This is a common pattern)
        # Or it might be a list of tensors, one for boxes, one for scores, one for landmarks.
        # Or a dictionary.
        
        # FOR NOW, let's assume a simple structure based on common MediaPipe patterns
        # and create dummy data. This WILL NEED to be replaced.
        detections_data = outputs 
        
    elif isinstance(outputs, (list, tuple)):
        print(f"Outputs is a list/tuple of length {len(outputs)}")
        for i, item in enumerate(outputs):
            if isinstance(item, torch.Tensor):
                print(f"  Item {i} shape: {item.shape}")
            else:
                print(f"  Item {i} type: {type(item)}")
        # If it's a list, it might be [boxes_tensor, scores_tensor, landmarks_tensor]
        # Or it could be a list of detection objects/dictionaries directly.
        # Based on the task description, we expect a list of detections.
        # The `qualcomm/MediaPipe-Face-Detection` model from QAI Hub usually
        # returns a list of `Detection` objects or a tensor that needs similar parsing.
        # Let's assume `outputs` is a list of tensors, where the first one holds the detections.
        # This is a common pattern for MediaPipe models from QAI Hub.
        # The tensor usually has shape [1, num_detections, 17]
        # 17 = 1 score + 4 bbox_coords + 12 landmark_coords (6 pairs)
        if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor) and outputs[0].ndim == 3:
            detections_data = outputs[0][0] # Assuming batch size 1, take the first element
            print(f"Using detections_data from outputs[0][0] with shape: {detections_data.shape}")
        else:
            print("Unexpected output structure. Cannot extract detections.")
            detections_data = torch.empty(0, 17) # Placeholder for no detections

    elif isinstance(outputs, dict):
        print(f"Outputs is a dict with keys: {outputs.keys()}")
        # If it's a dict, it might be {'boxes': tensor, 'scores': tensor, 'landmarks': tensor}
        # We'd need to combine these.
        # For now, let's assume this isn't the primary structure for this specific model.
        print("Dictionary output structure not yet handled in this template.")
        detections_data = torch.empty(0, 17) # Placeholder

    else:
        print("Unknown output structure.")
        return []

    results = []
    # Assuming detections_data is a tensor of shape [num_detections, 17]
    # where each row is [score, ymin, xmin, ymax, xmax, lx1, ly1, lx2, ly2, ..., lx6, ly6]
    # The order of ymin, xmin, ymax, xmax and landmarks might vary.
    # And score might be at a different position or not present in raw output.
    # Typical MediaPipe output:
    # 0: y_min
    # 1: x_min
    # 2: y_max
    # 3: x_max
    # 4-15: 6 landmarks (x,y pairs, often normalized to detection box or image)
    # Sometimes a confidence score is also included.
    # The QAI Hub model documentation for MediaPipe Face Detection states:
    # "Output is a list of torch Tensors. The first tensor is BxNx17, where B=1 (batch_size),
    #  N is the number of detected faces, and 17 corresponds to
    #  score (1), bounding_box (4: y_min, x_min, y_max, x_max), and keypoints (12: 6 points (x,y))"
    # So, for each detection in detections_data (shape [N, 17]):
    # Index 0: score
    # Index 1: y_min (normalized 0-1)
    # Index 2: x_min (normalized 0-1)
    # Index 3: y_max (normalized 0-1)
    # Index 4: x_max (normalized 0-1)
    # Index 5-16: landmarks (6 pairs, x, y, normalized 0-1 relative to image dimensions)
    
    if detections_data.numel() == 0: # Check if the tensor is empty
        print("No detections found in the output tensor.")
        return []

    for i in range(detections_data.shape[0]):
        detection = detections_data[i]
        score = detection[0].item()
        
        # Assuming a confidence threshold, e.g., 0.5
        if score < 0.5: # This threshold might need tuning
            continue

        # Bounding box (ymin, xmin, ymax, xmax) normalized to [0,1]
        y_min_norm = detection[1].item()
        x_min_norm = detection[2].item()
        y_max_norm = detection[3].item()
        x_max_norm = detection[4].item()

        # Scale box to original image dimensions
        x_min = x_min_norm * original_width
        y_min = y_min_norm * original_height
        x_max = x_max_norm * original_width
        y_max = y_max_norm * original_height

        # Landmarks (6 pairs: x, y) normalized to [0,1] relative to image dimensions
        landmarks_norm = detection[5:].reshape(6, 2) # Reshape to 6x2
        landmarks = []
        for lx_norm, ly_norm in landmarks_norm:
            lx = lx_norm.item() * original_width
            ly = ly_norm.item() * original_height
            landmarks.append([lx, ly])
        
        results.append({
            'box': [x_min, y_min, x_max, y_max],
            'landmarks': landmarks,
            'score': score
        })
    
    print(f"Found {len(results)} faces.")
    return results


def draw_detections_on_image(image_path: str, detections: list[dict], output_path: str = "detected_faces.jpg"):
    """
    Draws bounding boxes and landmarks on an image and saves it.

    Args:
        image_path: Path to the original image.
        detections: A list of detection dictionaries from detect_faces.
        output_path: Path to save the annotated image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path} for drawing.")
        return
    except Exception as e:
        print(f"Error opening image for drawing: {e}")
        return

    for det in detections:
        box = det['box']
        landmarks = det['landmarks']

        # Draw bounding box
        draw.rectangle(box, outline="red", width=2)

        # Draw landmarks (e.g., as small circles)
        for lx, ly in landmarks:
            # Draw a circle of radius 3 around each landmark
            draw.ellipse([(lx - 3, ly - 3), (lx + 3, ly + 3)], fill="blue", outline="blue")
            
    try:
        img.save(output_path)
        print(f"Annotated image saved to {output_path}")
    except Exception as e:
        print(f"Error saving annotated image: {e}")


if __name__ == "__main__":
    print("Face Detector script started.")
    # Create a dummy sample image for testing if one doesn't exist
    # This is important for environments where downloads might be restricted.
    sample_image_path = "sample_image.jpg"
    try:
        # Attempt to open, if it fails, create one
        test_img = Image.open(sample_image_path)
        print(f"Using existing sample image: {sample_image_path}")
    except FileNotFoundError:
        print(f"Sample image not found. Creating a dummy image: {sample_image_path}")
        try:
            dummy_img = Image.new('RGB', (600, 400), color = 'gray')
            draw = ImageDraw.Draw(dummy_img)
            # Draw a couple of simple "faces" for the model to potentially detect
            # A simple large rectangle (body)
            draw.rectangle([100,100, 200,200], fill="lightblue") # Face 1
            draw.ellipse([125,125, 135,135], fill="white") # Eye 1
            draw.ellipse([165,125, 175,135], fill="white") # Eye 2
            
            draw.rectangle([300,150, 450,300], fill="lightgreen") # Face 2
            draw.ellipse([325,175, 335,185], fill="white") # Eye 1
            draw.ellipse([365,175, 375,185], fill="white") # Eye 2
            dummy_img.save(sample_image_path)
            print(f"Dummy sample image saved to {sample_image_path}")
        except Exception as e:
            print(f"Could not create or save dummy image: {e}")
            # If image creation fails, the script might not be testable directly.
            # Consider exiting or using a known existing image path if this is critical.
            sample_image_path = None # Prevent further processing if no image

    if Model is None:
        print("Model was not loaded due to import errors. Cannot run test.")
    elif sample_image_path:
        print(f"\nTesting with image: {sample_image_path}")
        detections = detect_faces(sample_image_path)

        if detections:
            print("\nDetections found:")
            for i, det in enumerate(detections):
                print(f"  Face {i+1}:")
                print(f"    Box: {det['box']}")
                print(f"    Landmarks: {det['landmarks']}")
                if 'score' in det:
                    print(f"    Score: {det['score']:.4f}")
            
            draw_detections_on_image(sample_image_path, detections, "detected_faces_output.jpg")
        else:
            print("No detections were made or an error occurred.")
    else:
        print("No sample image available for testing.")

    print("\nFace Detector script finished.")
