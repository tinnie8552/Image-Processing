import streamlit as st
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from math import sqrt

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Fourier Transforms on Images', 'Different Filter on Images', 'Masking using Fourier Transforms', 'Video', 'Face Detection', 'Feature Detection', 'Object Detection')
    )

    if selected_box == 'Fourier Transforms on Images':
        welcome()
    if selected_box == 'Different Filter on Images':
        Filter()
    if selected_box == 'Masking using Fourier Transforms':
        photo()
    if selected_box == 'Video':
        video()
    if selected_box == 'Face Detection':
        face_detection()
    if selected_box == 'Feature Detection':
        feature_detection()
    if selected_box == 'Object Detection':
        object_detection()

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

def Filter():

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


if __name__ == "__main__":
    main()
