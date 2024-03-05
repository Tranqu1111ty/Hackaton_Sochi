import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from model_predict import procces
import cv2
import numpy as np


def get_image_as_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def background_color_():
    # Сначала открываем изображение
    image_path = r"image.jpg"
    image = Image.open(image_path)

    # Затем конвертируем его в base64
    img_base64 = get_image_as_base64(image)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1557683304-673a23048d34?q=80&w=1700&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-position: left; 
    background-repeat: repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.header("Микросервис для поиска рыб на изображении")
    # Используйте HTML и CSS для выравнивания текста заголовка по центру, увеличения размера шрифта, изменения шрифта на Montserrat и настройки межбуквенного расстояния
    st.markdown("""
    <head>
    <!-- Подключение шрифта Montserrat с Google Fonts -->
    <link href="https://fonts.googleapis.com/css?family=Montserrat:100" rel="stylesheet">
    </head>
    <style>
    .centered {
        text-align: center;
        font-size: 40px; /* Увеличение размера шрифта */
        font-family: 'Montserrat', sans-serif; /* Изменение шрифта на Montserrat */
        letter-spacing: 10px; /* Настройка межбуквенного расстояния */
    }
    </style>

    """, unsafe_allow_html=True)


def upload_and_display_image():
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    resized = None
    count_zeroes = None
    count_ones = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        rotates = image.rotate(180)
        pil_image_np = np.array(rotates)
        opencv_image = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)
        result, resized = procces(opencv_image)
        color = lambda x: (0, 0, 255) if x == 1 else (0, 255, 0)
        for box, cls in zip(result.get("bbox"), result.get("classes")):
            print(box, cls)
            cv2.rectangle(resized, (int(box[0] * 720), int(box[1] * 480)), (int(box[2] * 720), int(box[3] * 480)), color(cls), 2)
        count_zeroes = np.count_nonzero(result.get("classes") == 0)
        count_ones = np.count_nonzero(result.get("classes") == 1)
        img_base64 = get_image_as_base64(rotates)
        img_html = f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%; height: auto;" />
        </div>
        """
        st.markdown(img_html, unsafe_allow_html=True)
        style = "<style>h2 {text-align: center;}</style>"
        st.markdown(style, unsafe_allow_html=True)
        st.columns(3)[1].header("Загруженное изображение")
    else:
        st.error("Загрузите изображение!")

    return resized, count_zeroes, count_ones


def count_fish(img, count, nerest):
    if img is not None:
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_base64 = get_image_as_base64(pil_image)
        img_html = f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%; height: auto;" />
        </div>
        """
        st.markdown(img_html, unsafe_allow_html=True)

        style = "<style>h2 {text-align: center;}</style>"
        st.markdown(style, unsafe_allow_html=True)
        st.columns(3)[1].header("Обработанное изображение")

        if count != 0:
            st.sidebar.title(f"Общее количество рыб: {count + nerest}")
            st.sidebar.title(f"Количество рыб готовых к нересту: {nerest}")
        else:
            st.sidebar.title("Общее количество рыб и - не определено")


background_color_()
rotated_img, count_0, count_1 = upload_and_display_image()
if rotated_img is not None:
    count_fish(rotated_img, count_0, count_1)
