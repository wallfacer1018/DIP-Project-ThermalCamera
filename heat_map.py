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
frame = np.zeros((24, 32))

# Setup the plot
fig, ax = plt.subplots()
therm1 = ax.imshow(frame, interpolation='none', cmap='gray', vmin=0, vmax=60)
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [C]')

# Function to update the plot
def update_fig(*args):
    try:
        mlx.getFrame(frame)
    except ValueError:
        # Retry on ValueError
        return
    therm1.set_array(frame)
    return therm1,

# Create an animation
ani = animation.FuncAnimation(fig, update_fig, interval=500)

# Show the plot
plt.show()
