import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, label
from matplotlib.backends.backend_pdf import PdfPages

# Load the image
image_path = r"C:\Users\tomde\Downloads\frame_0018.png"  # Replace with your image path
image = imread(image_path, as_gray=True)
image = image / image.max()  # Normalize the image intensity

# Threshold the image to isolate bright spots
threshold = np.percentile(image, 99)  # Keep the top 1% intensity values
binary_mask = image > threshold

# Find the largest bright spot
labeled_mask, num_features = label(binary_mask)
spot_area = np.bincount(labeled_mask.ravel())
spot_area[0] = 0  # Ignore the background
largest_spot_label = np.argmax(spot_area)
largest_spot_mask = labeled_mask == largest_spot_label

# Get the center of the largest bright spot
spot_center = center_of_mass(largest_spot_mask)

# Crop the region around the bright spot
crop_size = 50  # Adjust this size as needed
x_min = int(max(spot_center[0] - crop_size, 0))
x_max = int(min(spot_center[0] + crop_size, image.shape[0]))
y_min = int(max(spot_center[1] - crop_size, 0))
y_max = int(min(spot_center[1] + crop_size, image.shape[1]))
cropped_image = image[x_min:x_max, y_min:y_max]

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()

# Generate a grid for Gaussian fitting
x = np.arange(cropped_image.shape[0])
y = np.arange(cropped_image.shape[1])
x, y = np.meshgrid(x, y)

# Improved initial guess for Gaussian parameters
amplitude_guess = cropped_image.max() - cropped_image.min()
x_center_guess, y_center_guess = crop_size, crop_size
sigma_guess = np.sqrt(np.sum(cropped_image > (cropped_image.max() / 2))) / 2
offset_guess = cropped_image.min()

initial_guess = (
    amplitude_guess,  # amplitude
    x_center_guess,   # x center
    y_center_guess,   # y center
    sigma_guess,      # sigma_x
    sigma_guess,      # sigma_y
    offset_guess,     # offset
)

# Refined bounds for Gaussian parameters
bounds = (
    [0, 0, 0, 0, 0, 0],  # Lower bounds
    [
        1.5 * amplitude_guess,
        cropped_image.shape[0],
        cropped_image.shape[1],
        cropped_image.shape[0] / 2,
        cropped_image.shape[1] / 2,
        cropped_image.max(),
    ],  # Upper bounds
)

# Perform the Gaussian fit
popt, _ = curve_fit(
    gaussian_2d,
    (x, y),
    cropped_image.ravel(),
    p0=initial_guess,
    bounds=bounds,
)

# Extract fitted parameters
amplitude, xo, yo, sigma_x, sigma_y, offset = popt

# Define the radius and other calculations
fraction = 0.1
fwhm_x = 2 * np.sqrt(2 * np.log(1 / fraction)) * sigma_x
fwhm_y = 2 * np.sqrt(2 * np.log(1 / fraction)) * sigma_y
diameter = (fwhm_x + fwhm_y) / 2
radius = diameter * 0.52 / 2

# Generate a 2D cut of the Gaussian along x-axis
x_cut = np.arange(cropped_image.shape[0])
y_cut = gaussian_2d((x_cut, np.ones_like(x_cut) * yo), amplitude, xo, yo, sigma_x, sigma_y, offset)

# Save the plots in a PDF
output_pdf_path = r"C:\Users\tomde\Desktop\Gaussian_fit_1.pdf"
with PdfPages(output_pdf_path) as pdf:
    # 2D cut along x-axis
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(x_cut, y_cut, label=f"Radius: {radius:.2f} microns")
    ax.set_title("2D Gaussian Profile Along X-axis")
    ax.set_xlabel("X-axis (pixels)")
    ax.set_ylabel("Intensity")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF with Gaussian fit and 2D profile saved to: {output_pdf_path}")
