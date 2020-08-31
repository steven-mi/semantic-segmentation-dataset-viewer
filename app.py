import os

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

BASE_PATH = os.getcwd()
DEFAULT_CLASS_COLOR_PATH = os.path.join(BASE_PATH, "data", "class_specification.csv")
DEFAULT_IMAGE_PATH = os.path.join(BASE_PATH, "data", "images")
DEFAULT_LABEL_PATH = os.path.join(BASE_PATH, "data", "labels")


def rgbstr_to_rgb(rgbstr):
    rgb = [int(color) for color in rgbstr.split(",")]
    return tuple(rgb)


def rgb_to_rgbstr(r, g, b):
    return "{},{},{}".format(r, g, b)


def rgb2hex(rgbstr):
    rgb = [int(color) for color in rgbstr.split(",")]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def get_all_images(path):
    paths = []
    for (dirpath, _, filenames) in os.walk(path, topdown=False):
        for filename in filenames:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                temp = os.path.join(dirpath, filename)
                paths.append(temp)
    paths.sort()
    return paths


@st.cache
def create_data_sidebar():
    # Reading class color CSV files
    st.sidebar.header("Data")
    class_color_path = st.sidebar.text_input('Enter path to class color specification:', value=DEFAULT_CLASS_COLOR_PATH)
    try:
        class_color_df = pd.read_csv(class_color_path)
    except FileNotFoundError as error_msg:
        st.sidebar.error(error_msg)

    try:
        # Check if needed columns are available
        class_colors = class_color_df["ClassColor"]
        class_names = class_color_df["ClassName"]
        class_color_dict = {}
        for class_color, class_name in zip(class_colors, class_names):
            class_color_dict[class_color] = class_name
    except KeyError:
        st.sidebar.error("ClassColor or ClassName is not available in CSV File")

    # Reading SemSeg images and labels
    image_path = st.sidebar.text_input('Enter path to images:', value=DEFAULT_IMAGE_PATH)
    label_path = st.sidebar.text_input('Enter path to labels:', value=DEFAULT_LABEL_PATH)

    # Get all paths
    image_paths = get_all_images(image_path)
    label_paths = get_all_images(label_path)
    st.header("There are {} images and {} labels".format(
        len(image_paths), len(label_paths)))
    if len(image_paths) != len(label_paths):
        st.sidebar.error("There should be the same amount of images as labels".format(
            len(image_paths), len(label_paths)))
    return class_color_dict, (image_paths, label_paths)


@st.cache
def create_class_color_checkboxes(class_color_dict):
    # Read CSV file
    st.sidebar.header("Classes")
    slider = st.sidebar.slider(label="Transparency", min_value=0.0, value=0.4, max_value=1.0, step=0.1)

    # put all needed information in a list
    class_checkboxes = []
    st.sidebar.text('\nWhich classes do you want to see?')
    for class_color in class_color_dict.keys():
        # create checkbox with color
        color = st.sidebar.markdown("""
            <svg width="80" height="20">
            <rect width="80" height="20" style="fill:{};stroke-width:3;stroke:rgb(0,0,0)" />
            </svg>""".format(rgb2hex(class_color)), unsafe_allow_html=True)
        box = st.sidebar.checkbox("{}".format(class_color_dict[class_color]))

        # append everything to list
        class_checkboxes.append((box, class_color, class_color_dict[class_color]))
    return class_checkboxes, slider


@st.cache
def create_images_per_class(paths):
    classes_per_image = []
    for image_path, label_path in zip(*paths):
        label = Image.open(label_path)
        colors = label.convert('RGB').getcolors()

        class_per_image = {}
        for _, color in colors:
            key = rgb_to_rgbstr(*color)
            class_per_image[key] = 1
        class_per_image["ImagePath"] = image_path
        class_per_image["LabelPath"] = label_path
        classes_per_image.append(class_per_image)

    classes_per_image_df = pd.DataFrame(classes_per_image)
    return classes_per_image_df


def create_overlay_image(image_path, label_path, alpha):
    image = Image.open(image_path)
    label = Image.open(label_path)

    image = image.convert("RGBA")
    label = label.convert("RGBA")

    overlay = Image.blend(image, label, alpha)
    st.image(overlay, use_column_width=True)
    return overlay


def main():
    class_color_dict, paths = create_data_sidebar()
    class_checkboxes, slider = create_class_color_checkboxes(class_color_dict)
    classes_per_image_df = create_images_per_class(paths)

    # get all checked boxes
    marked_boxes = []
    for box, class_color, class_name in class_checkboxes:
        if box:
            marked_boxes.append(class_color)
    marked = classes_per_image_df[marked_boxes + ["ImagePath", "LabelPath"]].dropna()

    image_list = marked["ImagePath"].tolist()
    label_list = marked["LabelPath"].tolist()
    selected_image = st.selectbox(
        "Which image do you want to see?",
        image_list
    )
    create_overlay_image(selected_image, label_list[image_list.index(selected_image)], slider)


if __name__ == '__main__':
    # Set title of app
    st.title('Semantic Segmentation Dataset Viewer')
    # Set title for sidebar
    st.sidebar.title("Settings")
    # run main
    main()
