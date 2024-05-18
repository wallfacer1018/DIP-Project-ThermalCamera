import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom

# Initialize I2C bus and MLX90640 sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)

print("MLX90640 addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

# Create a frame buffer
frame = [0] * 768

# Setup the plot
fig, ax = plt.subplots()
# Create an initial empty image with higher resolution for interpolation
therm1 = ax.imshow(np.zeros((240, 320)), interpolation='none', cmap='inferno')
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [C]')

frame_counter = 0  # Counter to keep track of frames


# Function to update the plot
def update_fig(*args):
    global frame_counter
    try:
        mlx.getFrame(frame)
    except ValueError:
        # Retry on ValueError
        return
    # Reshape the frame to 24x32
    data = np.reshape(frame, (24, 32))
    # Interpolate the data to 10x the original resolution
    data_interpolated = zoom(data, (10, 10), order=1)
    therm1.set_array(data_interpolated)
    therm1.set_clim(vmin=np.min(data_interpolated),
                    vmax=np.max(data_interpolated))  # Scale the colors to the min/max of the current frame

    # Save the current frame as a JPG image
    plt.savefig(f'frame_{frame_counter:04d}.jpg')
    frame_counter += 1

    return therm1,


# Create an animation
ani = animation.FuncAnimation(fig, update_fig, interval=500)

# Show the plot
plt.show()
