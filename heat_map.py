import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
therm1 = ax.imshow(np.zeros((24, 32)), interpolation='none', cmap='inferno')
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [C]')

# Function to update the plot
def update_fig(*args):
    try:
        mlx.getFrame(frame)
    except ValueError:
        # Retry on ValueError
        return
    data = np.reshape(frame, (24, 32))
    therm1.set_array(data)
    therm1.set_clim(vmin=np.min(data), vmax=np.max(data))  # Scale the colors to the min/max of the current frame
    return therm1,

# Create an animation
ani = animation.FuncAnimation(fig, update_fig, interval=500)

# Show the plot
plt.show()
