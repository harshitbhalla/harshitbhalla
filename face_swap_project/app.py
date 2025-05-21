# Main application file for the Face Swap project.
# This file will contain the Streamlit application code.

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter # ImageDraw for dummy images, ImageFilter for mask
import os
import numpy as np
import torch
import tempfile
from io import BytesIO

# Attempt to import from face_detector
try:
    from face_swap_project.face_detector import detect_faces, draw_detections_on_image as draw_detections_to_file
    # Check if the Model inside face_detector loaded correctly
    from face_swap_project.face_detector import Model as FaceDetectorModel
except ImportError as e:
    st.error(f"Critical Error: Could not import from face_detector.py: {e}. Application cannot start.")
    # Define dummy functions if import fails, so the rest of the app can be outlined
    def detect_faces(image_path: str) -> list[dict]:
        print("Warning: detect_faces (dummy) called. Face detector module not imported correctly.")
        return []
    def draw_detections_to_file(image_path: str, detections: list[dict], output_path: str):
        print("Warning: draw_detections_to_file (dummy) called. Face detector module not imported correctly.")
    FaceDetectorModel = None # Indicates the actual model from face_detector.py is not available

# Attempt to import diffusers
try:
    from diffusers import StableDiffusionInpaintPipeline
    DiffusersPipelineAvailable = True
except ImportError:
    # This error will be shown in the Streamlit UI if diffusers is needed
    DiffusersPipelineAvailable = False


# --- Existing Core Logic (create_face_mask, extract_source_face, swap_faces) ---
# These functions are assumed to be defined as in the previous version.
# For brevity, they are not repeated here, but they are part of the full app.py.

def create_face_mask(image_size: tuple[int, int], landmarks: list[list[float]]) -> Image.Image:
    mask_image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    polygon_points_indices = [4, 0, 2, 1, 5, 3] 
    if len(landmarks) < 6:
        return mask_image
    try:
        points_for_polygon = [(landmarks[i][0], landmarks[i][1]) for i in polygon_points_indices]
        draw.polygon(points_for_polygon, fill=255)
    except Exception as e:
        print(f"Error drawing polygon for mask: {e}")
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=5))

def extract_source_face(image_path: str) -> tuple[Image.Image | None, dict | None]:
    if FaceDetectorModel is None: return None, None
    detections = detect_faces(image_path)
    if not detections: return None, None
    selected_detection = detections[0]
    if len(detections) > 1:
        largest_area = 0
        for det in detections:
            box = det['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > largest_area:
                largest_area = area
                selected_detection = det
    try:
        image = Image.open(image_path).convert("RGB")
        box = selected_detection['box']
        return image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))), selected_detection
    except Exception as e:
        print(f"Error in extract_source_face: {e}")
        return None, None

def swap_faces(image1_path: str, image2_path: str, selected_face_index: int = 0, output_dir: str = "test_outputs_streamlit") -> Image.Image | None:
    if not DiffusersPipelineAvailable or FaceDetectorModel is None:
        st.error("Models not available for swapping.")
        return None
    
    # Create output_dir for swap_faces debug images if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Warning: Could not create output directory {output_dir} for swap_faces: {e}")
            # Fallback to a default if creation fails, or handle error appropriately
            output_dir = tempfile.gettempdir() 

    model_id = "stabilityai/stable-diffusion-2-inpainting"
    pipeline = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
        else:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(device)
    except Exception as e:
        st.error(f"Error loading inpainting model: {e}")
        return None

    try:
        original_image1 = Image.open(image1_path).convert("RGB")
    except Exception as e:
        st.error(f"Error opening target image: {e}")
        return None

    detections_img1 = detect_faces(image1_path)
    if not detections_img1 or selected_face_index >= len(detections_img1):
        st.error("No faces in target or invalid index.")
        return None
    
    selected_face_info = detections_img1[selected_face_index]
    landmarks_img1 = selected_face_info['landmarks']
    mask_image = create_face_mask(original_image1.size, landmarks_img1)
    
    # Save debug mask
    try:
        mask_save_path = os.path.join(output_dir, "st_generated_mask.png")
        mask_image.save(mask_save_path)
    except Exception as e:
        print(f"Warning: Could not save Streamlit generated mask: {e}")


    source_face_pil_image, _ = extract_source_face(image2_path)
    if not source_face_pil_image:
        st.error("Could not extract source face.")
        return None

    target_bbox = selected_face_info['box']
    target_width = int(target_bbox[2] - target_bbox[0])
    target_height = int(target_bbox[3] - target_bbox[1])
    if target_width <= 0 or target_height <= 0:
        st.error(f"Invalid target bounding box: w={target_width}, h={target_height}")
        return None
    source_face_pil_image = source_face_pil_image.resize((target_width, target_height), Image.LANCZOS)

    image_for_inpainting = original_image1.copy()
    image_for_inpainting.paste(source_face_pil_image, (int(target_bbox[0]), int(target_bbox[1])))
    
    # Save debug intermediate image
    try:
        inp_ready_path = os.path.join(output_dir, "st_image_ready_for_inpainting.png")
        image_for_inpainting.save(inp_ready_path)
    except Exception as e:
        print(f"Warning: Could not save Streamlit image_ready_for_inpainting: {e}")

    try:
        inpainted_image = pipeline(
            prompt="photorealistic face, seamless, high quality, natural skin texture, perfect eyes", 
            image=image_for_inpainting.resize((512, 512), Image.LANCZOS),
            mask_image=mask_image.resize((512, 512), Image.LANCZOS)
        ).images[0]
        return inpainted_image.resize(original_image1.size, Image.LANCZOS)
    except Exception as e:
        st.error(f"Error during inpainting: {e}")
        return None

# --- Streamlit UI Helper ---
def _draw_detections_pil(image_path_or_pil: str | Image.Image, detections: list[dict]) -> Image.Image:
    """Helper to draw detections on a PIL image and return it."""
    try:
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil.convert("RGB")
        else:
            raise ValueError("Input must be a file path or a PIL Image.")
            
        draw = ImageDraw.Draw(img)
        for det in detections:
            box = det['box']
            landmarks = det['landmarks']
            draw.rectangle(box, outline="red", width=3)
            for lx, ly in landmarks:
                draw.ellipse([(lx - 3, ly - 3), (lx + 3, ly + 3)], fill="blue", outline="blue")
        return img
    except Exception as e:
        st.error(f"Error drawing detections: {e}")
        # Return original image if drawing fails
        if isinstance(image_path_or_pil, str):
            return Image.open(image_path_or_pil).convert("RGB")
        return image_path_or_pil


# --- Streamlit App ---
def run_streamlit_app():
    st.title("Face Swap Application")

    # Model Availability Checks
    if FaceDetectorModel is None: # Check if the class itself is None (failed import from face_detector)
        st.error("Face detector model components are missing or failed to load. Please check setup. UI will be limited.")
    if not DiffusersPipelineAvailable:
        st.error("Diffusers library or selected model is not available. Face swapping will not work.")

    # Initialize session state variables
    if 'image1_path' not in st.session_state:
        st.session_state.image1_path = None
    if 'image2_path' not in st.session_state:
        st.session_state.image2_path = None
    if 'detections_img1' not in st.session_state:
        st.session_state.detections_img1 = None
    if 'selected_face_idx' not in st.session_state:
        st.session_state.selected_face_idx = 0 # Default to 0
    if 'swapped_image_bytes' not in st.session_state:
        st.session_state.swapped_image_bytes = None


    # --- Image 1 (Target) Upload ---
    st.header("1. Upload Target Image (Image 1)")
    uploaded_file_1 = st.file_uploader("Choose target image...", type=["jpg", "jpeg", "png"], key="uploader1")

    if uploaded_file_1 is not None:
        try:
            # Save to a temporary file to get a persistent path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file1:
                tmp_file1.write(uploaded_file_1.getvalue())
                st.session_state.image1_path = tmp_file1.name
            
            st.image(st.session_state.image1_path, caption="Target Image")

            if FaceDetectorModel is not None: # Only run detection if model is available
                with st.spinner("Detecting faces in Target Image..."):
                    st.session_state.detections_img1 = detect_faces(st.session_state.image1_path)
                
                if st.session_state.detections_img1:
                    img_with_detections = _draw_detections_pil(st.session_state.image1_path, st.session_state.detections_img1)
                    st.image(img_with_detections, caption=f"Detected {len(st.session_state.detections_img1)} face(s)")

                    if len(st.session_state.detections_img1) > 1:
                        face_options = [f"Face {i+1}" for i in range(len(st.session_state.detections_img1))]
                        selected_option = st.radio(
                            "Select face to replace:", 
                            face_options, 
                            index=st.session_state.selected_face_idx
                        )
                        st.session_state.selected_face_idx = face_options.index(selected_option)
                    elif len(st.session_state.detections_img1) == 1:
                        st.session_state.selected_face_idx = 0
                        st.write("Face 1 selected (only one detected).")
                    # No need for explicit else for no detections, as it's handled by outer if
                else:
                    st.write("No faces detected in target image.")
            else:
                st.warning("Face detector not available. Cannot process target image further.")
        except Exception as e:
            st.error(f"Error processing Target Image: {e}")
            st.session_state.image1_path = None # Reset on error

    # --- Image 2 (Source) Upload ---
    st.header("2. Upload Source Image (Image 2)")
    uploaded_file_2 = st.file_uploader("Choose source image...", type=["jpg", "jpeg", "png"], key="uploader2")

    if uploaded_file_2 is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file2:
                tmp_file2.write(uploaded_file_2.getvalue())
                st.session_state.image2_path = tmp_file2.name
            
            st.image(st.session_state.image2_path, caption="Source Image")

            if FaceDetectorModel is not None: # Only try extraction if model is available
                with st.spinner("Extracting face from Source Image..."):
                    source_face_pil, _ = extract_source_face(st.session_state.image2_path)
                if source_face_pil:
                    st.image(source_face_pil, caption="Extracted Source Face (Largest)")
                else:
                    st.write("No face extracted from source image.")
            else:
                st.warning("Face detector not available. Cannot process source image.")
        except Exception as e:
            st.error(f"Error processing Source Image: {e}")
            st.session_state.image2_path = None # Reset on error

    # --- Face Swap Action ---
    st.header("3. Perform Face Swap")
    if st.button("Swap Faces!", disabled=(FaceDetectorModel is None or not DiffusersPipelineAvailable)):
        if st.session_state.image1_path and st.session_state.image2_path and st.session_state.detections_img1:
            if st.session_state.selected_face_idx < len(st.session_state.detections_img1):
                with st.spinner("Swapping faces... this may take a while."):
                    # Use a different output directory for streamlit related debug images
                    streamlit_debug_output_dir = "test_outputs_streamlit_run" 
                    result_image = swap_faces(
                        st.session_state.image1_path, 
                        st.session_state.image2_path, 
                        st.session_state.selected_face_idx,
                        output_dir=streamlit_debug_output_dir 
                    )
                
                if result_image:
                    st.image(result_image, caption="Swapped Result")
                    buf = BytesIO()
                    result_image.save(buf, format="PNG")
                    st.session_state.swapped_image_bytes = buf.getvalue() # Store for download button
                else:
                    st.error("Face swapping failed. Check console logs or try different images.")
                    st.session_state.swapped_image_bytes = None
            else:
                st.error("Selected face index is invalid. Please re-select the face in Image 1.")
                st.session_state.swapped_image_bytes = None
        else:
            st.error("Please upload both target and source images, and ensure faces are detected in the target image.")
            st.session_state.swapped_image_bytes = None

    if st.session_state.swapped_image_bytes:
        st.download_button(
            label="Download Swapped Image",
            data=st.session_state.swapped_image_bytes,
            file_name="swapped_image.png",
            mime="image/png"
        )


if __name__ == "__main__":
    # --- Comment out or remove the old test block ---
    # print("App.py script started in __main__ block FOR TESTING.")
    # sample_image_path_1 = "sample_image_1.jpg"
    # sample_image_path_2 = "sample_image_2.jpg"
    # output_dir = "test_outputs"
    # try:
    #     if not os.path.exists(output_dir): os.makedirs(output_dir)
    #     # ... (rest of dummy image creation and local tests) ...
    # except Exception as e:
    #     print(f"Error in local test setup: {e}")
    # print("\nApp.py script finished local __main__ block.")
    
    run_streamlit_app()
