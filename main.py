# Import all the required libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO  # object detection
from PIL import Image  # Image class (not image)
import sys
print("Python version:", sys.version)

# -------------------------------------------1) RECOVER THE PATH OF THE MAIN FILE (main.py)-----------------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
# print(f"ROOT = : {ROOT}")      Print to check the correct ROOT 

# --------------------------------------------2) DEFAULT CONFIGURATION-----------------------------------------------------
IMAGE_DIR = ROOT /'images'
#print(f"Images folder path' : {IMAGE_DIR}")         Print to verify the correct path of the folder containing the default

DEFAULT_IMAGE = IMAGE_DIR / 'g_813.png'
#print(f"Path DEFAULT IMAGE : {DEFAULT_IMAGE}")    Print to verify the correct path of the file for the default image

DEFAULT_DETECT_IMAGE = IMAGE_DIR/'g_813_detect.png'
#print(f"Chemin IMAGE DETECTION DEFAUT ' : {DEFAULT_DETECT_IMAGE}")   Print to verify the correct path of the file for the default detection image


# --------------------------------------------3) MODEL CONFIGURATIONS ---------------------------------------------------------

MODEL_DIR = ROOT/'weights'
#print(f"Models Path ' : {MODEL_DIR}")              Print to verifiy the correct path of the folder conatining models 

MODEL_SEG_DIR=MODEL_DIR/'my_model_seg.pt'
#print(f"Chemin DU MODEL SEG ' : {MODEL_SEG_DIR}")   Print to verifiy the correct path of the segmentation model  

MODEL_DETECT_DIR=MODEL_DIR/'my_model_det.pt'
#print(f"Chemin DU MODELE DETECT' : {MODEL_DETECT_DIR}") Print to verifiy the correct path of the detection model


# ------------------------------------------- 4) APPLICATION FORM --------------------------------------------------------- 


# Page Layout
st.set_page_config(
    page_title="YOLO11",
    page_icon="🧠"  
)
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 50px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7U_6yF8HVQjx7h0aYRr-JJoaJv7ye7j05yQ&s\n" alt="Logo 1" style="height:80px;">
        <img src="https://upload.wikimedia.org/wikipedia/id/9/97/I3l-logo.jpg" alt="Logo 2" style="height:80px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<h1 style='text-align: center; color: red; font-size: 48px;'>🧠 BRAIN TUMOR DETECTION AND SEGMENTATION</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center;'>
        <p style='font-size:16px;'>
            <strong>Author :</strong> <em>Ferkioui Yanis</em><br>
            <span style='font-size:12px; color: gray;'>(Industrial student engineer UniLaSalle Amiens | France)</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style='text-align: center;'>
        <p style='font-size:16px;'>
            <strong>Supervisor :</strong> <em>Dr. Muammar Sadrawi </em><br>
            <span style='font-size:12px; color: gray;'>(Researcher in the department of bioinformatics I3L Jakarta | France)</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h6 style='text-align: center; color: blue;'>
        Update : <strong><em> July 2025 </em></strong>
    </h6> 
    """,
    unsafe_allow_html=True
)




# Sidebar
st.sidebar.header("Model Configuration")

# Choose Model Type: Detection or Segmentation
model_type = st.sidebar.radio(
    label="What Type of Model ? ",
    options=["Segmentation", "Detection"],
    captions=["Dimensions and Precision", "Bounding Box Only"]
)

# Select Confidence Value
confidence_value = st.sidebar.slider(
    "Select Model Confidence configuration Value : ",
    min_value=1,
    max_value=100,
    value=40
) / 100.0  # Convert to float between 0.25 and 1.00


#Selecting Detection Segmentation
if model_type=="Detection":
    model_path=Path(MODEL_DETECT_DIR)

elif model_type=="Segmentation":
    model_path=Path(MODEL_SEG_DIR)


#Load the Yolo Model
try :
    model=YOLO(model_path)
except Exception as e :
    st.error(f"Unable to load model. Check the specified path : {model_path}")
    st.error(e)

#Image configuration
st.sidebar.header("Image Configuration ")
source_radio=st.sidebar.radio(
    label="Select Source ",
    options= "IMAGE",
)

source_image=None
if source_radio == "IMAGE":
    source_image = st.sidebar.file_uploader(
        "Choose an Image.....", type = ("jpg","png","jpeg","bmp","webp")
    )

# Uploading 
col1,col2=st.columns(2)
with col1:
    try:
        if source_image is None:
            defautlt_image_path=str(DEFAULT_IMAGE)
            defautlt_image=Image.open(defautlt_image_path)
            st.image(defautlt_image_path, caption="Default Image",use_container_width=True)
        else:
            uploaded_image=Image.open(source_image)
            st.image(source_image, caption="Uploaded Imge",use_container_width=True)
    except Exception as e:
        st.error("Error while Open the Image")
        st.error(e)

with col2:
    try:
        if source_image is None :
            detected_default_image_path=str(DEFAULT_DETECT_IMAGE)
            defautlt_detect_image=Image.open(detected_default_image_path)
            st.image(detected_default_image_path,caption="Defautlt Detected Images",use_container_width=True)
        else:
            if st.sidebar.button("Detect Tumor"):
                status_placeholder = st.empty()  # Espace réservé pour le message
                status_placeholder.info("⏳ Running ...")
                result=model.predict(uploaded_image, conf=confidence_value)
                boxes=result[0].boxes
                result_plotted=result[0].plot()[:,:,::-1]
                st.image(result_plotted, caption="Detected Image", use_container_width=True)
                status_placeholder.success("✅ Detection finished.")
               
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as e:
                    st.error(e)
    except Exception as e:
        st.error("Error Occured While Opening the Image")
        st.error(e)