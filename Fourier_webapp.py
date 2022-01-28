import streamlit as st
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from math import sqrt
from streamlit_drawable_canvas import st_canvas

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Fourier Transforms on Images', 'Different Filters on Images', 'DIY Masking Tool')
    )

    if selected_box == 'Fourier Transforms on Images':
        welcome()
    if selected_box == 'Different Filters on Images':
        filter_img()
    if selected_box == 'DIY Masking Tool':
        masking_img()

def welcome():
    st.title('Fourier Transform on Images!')
    st.subheader('This is an app built to help the user understand the various use-cases and effects of 2D Fourier Transforms on images')
    st.write('The Fourier Transform of a 2D image helps us represent the image in the *frequency* or Fourier domain by decomposing'+
    ' the image into its sine and cosine components.')
    st.write('Let us see what happens when we apply a 2D Fourier Transform to the following image....')
    st.image('puppy2.jpg',use_column_width=True)

    img_0 = cv2.imread('puppy2.jpg')
    img = rgb2gray(img_0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    magnitude_spectrum  = magnitude_spectrum  / np.max(magnitude_spectrum )
    magnitude_spectrum  = (magnitude_spectrum*255).astype('uint8')
    st.write('The 2D Fourier Transform of the above image gives us the corresponding frequency domain representation shown below:')
    st.image(magnitude_spectrum,use_column_width=True,clamp=True)
    st.write('The whiter regions of the frequency domain image indicate low frequency and the darker colors indicate higher frequency')
    # add RGB 3 channel
    st.write('The 2D Fourier Transform of the above image gives us the corresponding frequency domain in RGB channel representation shown below:')
    fig, ax = plt.subplots(1, img_0.shape[2])
    subtitle = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i in range(img_0.shape[2]):
        f = np.fft.fft2(img_0[:,:,i])
        fshift = np.fft.fftshift(f)
        ax[i].imshow(np.log(abs(fshift)), cmap='gray')
        ax[i].set_title(subtitle[i], fontsize=5)
        ax[i].tick_params(labelsize=5)
    st.pyplot(fig)

    st.write('TO-DO: Fill more info here ?')
    st.subheader('Try it out!')
    # Asking user to upload their own image for calculating 2D fft
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        up_img_0 = np.array(original)
        st.image(up_img_0, use_column_width=True)
        up_img = rgb2gray(up_img_0)
        f_up = np.fft.fft2(up_img)
        fshift_up = np.fft.fftshift(f_up)
        magnitude_spectrum1 = np.log(np.abs(fshift_up))
        magnitude_spectrum1  = magnitude_spectrum1  / np.max(magnitude_spectrum1 )
        magnitude_spectrum1  = (magnitude_spectrum1*255).astype('uint8')
        st.write('Your image in frequency domain:')
        st.image(magnitude_spectrum1,use_column_width=True,clamp=True)
        # add RGB 3 channel
        st.write('Your image in frequency domain in RGB channel:')
        fig, ax = plt.subplots(1, up_img_0.shape[2])
        subtitle = ['Red Channel', 'Green Channel', 'Blue Channel']
        for i in range(up_img_0.shape[2]):
            f = np.fft.fft2(up_img_0[:, :, i])
            fshift = np.fft.fftshift(f)
            ax[i].imshow(np.log(abs(fshift)), cmap='gray')
            ax[i].set_title(subtitle[i], fontsize=5)
            ax[i].tick_params(labelsize=5)
        st.pyplot(fig)


# my code start from here

def distance(a,b):
    return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def fliter(D,image,type):
    base = np.zeros(image.shape[:2])
    row, col = image.shape[:2]
    center = (row/2,col/2)
    for i in range(row):
        for j in range(col):
            if type == 'Lowpass':
                if distance((i,j),center) < D:
                    base[i,j] = 1
            elif type == 'highpass':
                if distance((i,j),center) > D:
                    base[i,j] = 1
            else:
                if distance((i, j), center) > D[0] and distance((i, j), center) < D[1]:
                    base[i, j] = 1
    return base

def fourier(image):
    image_fourier = []
    fig, ax = plt.subplots(1, image.shape[2])
    subtitle = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i in range(image.shape[2]+1):
        if i == 0:
            st.write('The size of the image is ', image.shape)
            st.write('Your image in frequency domain of three color channel (RGB)')
        else:
            fft = np.fft.fftshift(np.fft.fft2((image[:, :, i - 1])))
            image_fourier.append(fft)
            ax[i-1].imshow(np.log(abs(image_fourier[i - 1])), cmap='gray')
            ax[i-1].set_title(subtitle[i-1], fontsize=5)
            ax[i-1].tick_params(labelsize=5)

    st.pyplot(fig)
    return image_fourier

def filter_img():

    st.title('Different Filter on Images')
    st.subheader('You can see your image change by different filter')

    file = st.file_uploader("Choose an image...", type=["jpeg", "png", "jpg"])
    if file is not None:

        original = Image.open(file)
        image = np.array(original)
        st.image(image, use_column_width=True)

        image_fourier = fourier(image)

        type_filter = st.radio(
            "Type of filter",
            ('Lowpass', 'highpass', 'bandpass'))
        if type_filter == 'Lowpass':
            st.latex(r'''H(x,y) =
                \begin{cases}
                        1       & \quad \text{if } D(x,y) \leq d\\
                        0  & \quad \text{if } D(x,y) > d
                \end{cases}''')
            st.write(r'''Formula for low pass filter where $d$ is the positive constant and $D(x,y)$ is the
            distance between a point $(x,y)$ in the frequency domain and the center of the frequency rectangle''')
            D = st.number_input('Input d', min_value=0.0)
        elif type_filter == 'highpass':
            st.latex(r'''H(x,y) =
                \begin{cases}
                        1       & \quad \text{if } D(x,y) \geq d\\
                        0  & \quad \text{if } D(x,y) < d
                \end{cases}''')
            st.write(r'''Formula for high pass filter where $d$ is the positive constant and $D(x,y)$ is the
            distance between a point $(x,y)$ in the frequency domain and the center of the frequency rectangle''')
            D = st.number_input('Input d', min_value=0.0)
        else:
            st.latex(r'''H(x,y) =
                            \begin{cases}
                                    1       & \quad \text{if } d_0 \leq D(x,y) \leq d_1\\
                                    0  & \quad \text{else }
                            \end{cases}''')
            st.write(r'''Formula for band pass filter where $d_0$ and $d_1$ is the positive constant and $D(x,y)$ is the
                        distance between a point $(x,y)$ in the frequency domain and the center of the frequency rectangle''')
            st.write(r'''Input $d_0$ and $d_1$''')
            a = st.number_input('Input d_0', min_value=0.0)
            b = st.number_input('Input d_1', min_value=0.0)
            D = [a,b]

        if st.button('Get result'):

            st.write('Thus the new image through filter in fourier transform will be')
            st.latex(r'''F \times H''')
            st.write(r'''where $F,$ $H$ is the fourier transform of the original image and filter''')

            st.write('There is the picture of the new image through filter in fourier transform of three channel')
            fig, ax = plt.subplots(1, image.shape[2])
            subtitle = ['Red Channel', 'Green Channel', 'Blue Channel']

            inverse_image = []

            for i in range(image.shape[2]):
                image_filter = image_fourier[i] * fliter(D, image, type_filter)
                ax[i].imshow(np.log(1+abs(image_filter)), cmap='gray')
                ax[i].set_title(subtitle[i - 1], fontsize=5)
                ax[i].tick_params(labelsize=5)
                inverse_image.append(abs(np.fft.ifft2(image_filter)))
                if np.max(inverse_image[i]) != 0:
                    inverse_image[i] = inverse_image[i] / np.max(inverse_image[i])

            st.pyplot(fig)

            st.write('That is the result of the input image through filter and inverse fourier transform')
            final_image = np.dstack([(inverse_image[0]*255).astype(int),
                                     (inverse_image[1]*255).astype(int),
                                     (inverse_image[2]*255).astype(int)])

            st.image(final_image, use_column_width=True)

# my code end


# Specify canvas parameters in application


def get_masked_image(image, canvas_image):
    mask = canvas_image[:,:,3]
    mask_inv = cv2.bitwise_not(mask)
    mask_inv3 = cv2.merge((mask_inv,mask_inv,mask_inv))
    return cv2.bitwise_and(image, mask_inv3)


def inverse_furiour(image):
    final_image = []
    for c in image:
        channel = abs(np.fft.ifft2(c))
#         plt.imshow(channel)
#         plt.show()
        final_image.append(channel)
    final_image_assebled = np.dstack([final_image[0].astype('int'),
                                     final_image[1].astype('int'),
                                     final_image[2].astype('int')])
    return final_image_assebled

def create_canvas_draw_instance(stroke_width, stroke_color, bg_color, drawing_mode, realtime_update, background_image, key, height, width):

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(background_image),
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        height = height,
        width = width,
        key=key,
    )

    return canvas_result

def get_mask_from_canvas(canvas_images):
    list_mask = []
    for image in canvas_images:
        list_mask.append(image[:,:,3])

    return list_mask

def normalize_image(img):
    img = img / np.max(img)
    return (img*255).astype('uint8')

def write_background_images(images, names):
    for image, name in zip(images, names):
        image3 = cv2.merge((image,image,image))
        image_3_nor = normalize_image(image3)
        cv2.imwrite(name, image_3_nor)

def write_canvas_images(images, names):
    for image, name in zip(images, names):
        cv2.imwrite(name, image)

def rgb_fft(image):
    f_size = 25
    fft_images=[]
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(abs(rgb_fft)))

    return fft_images, fft_images_log


def apply_mask(input_image, mask):
    _, mask_thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    mask_bool = mask_thresh.astype('bool')
    input_image[mask_bool] = 1

    return input_image


def apply_mask_all(list_images, list_mask):
    final_result = []

    for (i,mask) in zip(list_images, list_mask):
        result = apply_mask(i,mask)
        final_result.append(result)
    return final_result

def masking_img():
    st.title('A do-it-yourself Masking Tool')
    st.subheader('Play around and create your own filter in the frequency domain and visualize the results')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#FFFFFF")
    drawing_mode = st.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.checkbox("Update in realtime", True)
    if uploaded_file is not None:

        original = Image.open(uploaded_file)
        img = np.array(original)
        st.image(img, use_column_width=True)

        fft_images, fft_images_log = rgb_fft(img)

        for temp in fft_images_log:
            st.text(temp.shape)

        names = ["bg_image_r.png", "bg_image_g.png", "bg_image_b.png"]

        write_background_images(fft_images_log, names)

        st.text("Red Channel in frequency domain - ")
        canvas_r = create_canvas_draw_instance(stroke_width, stroke_color, bg_color, drawing_mode, realtime_update, names[0], key="red", height=img.shape[0], width=img.shape[1])
        st.text("Green Channel in frequency domain - ")
        canvas_g = create_canvas_draw_instance(stroke_width, stroke_color, bg_color, drawing_mode, realtime_update, names[1], key="green",height=img.shape[0], width=img.shape[1])
        st.text("Blue channel in frequency domain - ")
        canvas_b = create_canvas_draw_instance(stroke_width, stroke_color, bg_color, drawing_mode, realtime_update, names[2], key="blue", height=img.shape[0], width=img.shape[1])

        if st.button('Get Result: - '):

            canvas_image_data = [canvas_r.image_data, canvas_g.image_data, canvas_b.image_data]

            names_canvas_images = ["canvas_image_r.png","canvas_image_g.png","canvas_image_b.png"]
            write_canvas_images(canvas_image_data, names_canvas_images)

            # appending the images which are saved earlier
            canvas_images = []

            for name in names_canvas_images:
                canvas_images.append(cv2.imread(name,-1))

            list_mask = get_mask_from_canvas(canvas_images)

            # reading canvas images
            result = apply_mask_all(fft_images, list_mask)

            transformed = inverse_furiour(result)

            # result_clipped = np.clip(result, 0 ,255)

            # result_formatted = np.dstack([result_clipped[0],
            #                             result_clipped[1],
            #                             result_clipped[2]])

            transformed_clipped = np.clip(transformed, 0, 255)
            st.text("Image Returned by Inverse Fourier Transform - ")
            st.image(transformed_clipped, use_column_width=True)


if __name__ == "__main__":
    main()
