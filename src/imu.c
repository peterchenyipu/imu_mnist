#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <imu.h>
#include <zephyr/drivers/uart.h>

FusionEuler euler;

FusionEuler getEuler()
{
	return euler;
}

#define SAMPLE_PERIOD (0.01f) // replace this with actual sample period

#define LOG_LEVEL CONFIG_LOG_DEFAULT_LEVEL
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(imu);

static inline float out_ev(struct sensor_value *val)
{
	return (val->val1 + (float)val->val2 / 1000000);
}

static int print_samples;
static int lsm6dsl_trig_cnt;

static struct sensor_value accel_x_out, accel_y_out, accel_z_out;
static struct sensor_value gyro_x_out, gyro_y_out, gyro_z_out;

#ifdef CONFIG_LSM6DSL_TRIGGER
static void lsm6dsl_trigger_handler(const struct device *dev,
				    const struct sensor_trigger *trig)
{
	static struct sensor_value accel_x, accel_y, accel_z;
	static struct sensor_value gyro_x, gyro_y, gyro_z;

	lsm6dsl_trig_cnt++;

	sensor_sample_fetch_chan(dev, SENSOR_CHAN_ACCEL_XYZ);
	sensor_channel_get(dev, SENSOR_CHAN_ACCEL_X, &accel_x);
	sensor_channel_get(dev, SENSOR_CHAN_ACCEL_Y, &accel_y);
	sensor_channel_get(dev, SENSOR_CHAN_ACCEL_Z, &accel_z);

	/* lsm6dsl gyro */
	sensor_sample_fetch_chan(dev, SENSOR_CHAN_GYRO_XYZ);
	sensor_channel_get(dev, SENSOR_CHAN_GYRO_X, &gyro_x);
	sensor_channel_get(dev, SENSOR_CHAN_GYRO_Y, &gyro_y);
	sensor_channel_get(dev, SENSOR_CHAN_GYRO_Z, &gyro_z);

	accel_x_out = accel_x;
	accel_y_out = accel_y;
	accel_z_out = accel_z;

	gyro_x_out = gyro_x;
	gyro_y_out = gyro_y;
	gyro_z_out = gyro_z;

}
#endif

#define ATL_LOCAL_G (9.80665f)

#define UART_DEVICE_NODE DT_NODELABEL(xiao_serial)
static const struct device *const uart_dev = DEVICE_DT_GET(UART_DEVICE_NODE);

bool uartSending = false;
void uart_dma_cb(const struct device *dev, struct uart_event *evt, void *user_data)
{
	if (evt->type == UART_TX_DONE) {
		uartSending = false;
	} else if (evt->type == UART_TX_ABORTED) {
		uartSending = false;
	}
}

IMU imu = {0};

#define CH_COUNT 7
struct Frame {
	float fdata[CH_COUNT];
	unsigned char tail[4];
} __attribute__((packed));

struct Frame frame = {
	.tail = {0x00, 0x00, 0x80, 0x7f}
};


int imu_task(void)
{
    int cnt = 0;
	char out_str[64];
	struct sensor_value odr_attr;
	const struct device *const lsm6dsl_dev = DEVICE_DT_GET_ONE(st_lsm6dsl);

	if (!device_is_ready(lsm6dsl_dev)) {
		printk("sensor: device not ready.\n");
		return 0;
	}

	/* set accel/gyro sampling frequency to 104 Hz */
	odr_attr.val1 = 104;
	odr_attr.val2 = 0;

	if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_ACCEL_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
		printk("Cannot set sampling frequency for accelerometer.\n");
		return 0;
	}

	if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_GYRO_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
		printk("Cannot set sampling frequency for gyro.\n");
		return 0;
	}

#ifdef CONFIG_LSM6DSL_TRIGGER
	struct sensor_trigger trig;

	trig.type = SENSOR_TRIG_DATA_READY;
	trig.chan = SENSOR_CHAN_ACCEL_XYZ;

	if (sensor_trigger_set(lsm6dsl_dev, &trig, lsm6dsl_trigger_handler) != 0) {
		printk("Could not set sensor type and channel\n");
		return 0;
	}
#endif

	if (sensor_sample_fetch(lsm6dsl_dev) < 0) {
		printk("Sensor sample update error\n");
		return 0;
	}

	printk("Sampling accelerometer/gyro... (press button to print data)\n");

    uint32_t count = 0;

	FusionAhrs ahrs;
	// TODO add calibration here
    FusionAhrsInitialise(&ahrs);

	// prepare the UART
	if (!device_is_ready(uart_dev)) {
		printk("UART device not ready\n");
		while (1)
			;
	}

	int ret = uart_callback_set(uart_dev, uart_dma_cb, NULL);
	if (ret < 0) {
		if (ret == -ENOTSUP) {
			printk("Async UART API support not enabled\n");
		} else if (ret == -ENOSYS) {
			printk("UART device does not support Async API\n");
		} else {
			printk("Error setting UART callback: %d\n", ret);
		}
		while (1) {
			printk("Error setting UART callback\n");
			k_msleep(1000);
		}
	}

	sprintf(out_str, "hello world\n");
	uart_tx(uart_dev, out_str, strlen(out_str), SYS_FOREVER_US);

    while (1)
    {
        // if ((count % 10) == 0U) {
		// 	sprintf(out_str, "a x:%.1f y:%.1f z:%.1f\ng x:%.1f y:%.1f z:%.1f\n",
		// 					  out_ev(&accel_x_out),
		// 					  out_ev(&accel_y_out),
		// 					  out_ev(&accel_z_out),
		// 					  out_ev(&gyro_x_out),
		// 					  out_ev(&gyro_y_out),
		// 					  out_ev(&gyro_z_out));
        //     // log out_str
        //     printk("%s", out_str);
    	// }
		
		// Update IMU
		// accel in m/s^2, gyro in rad/s
		imu.raw.ax = out_ev(&accel_x_out);
		imu.raw.ay = out_ev(&accel_y_out);
		imu.raw.az = out_ev(&accel_z_out);
		imu.raw.gx = out_ev(&gyro_x_out);
		imu.raw.gy = out_ev(&gyro_y_out);
		imu.raw.gz = out_ev(&gyro_z_out);


		const FusionVector accelerometer = {
			.array = {
				imu.raw.ax / ATL_LOCAL_G,
				imu.raw.ay / ATL_LOCAL_G,
				imu.raw.az / ATL_LOCAL_G
			}
		};
        const FusionVector gyroscope = {
			.array = {
				imu.raw.gx * 180.0f / M_PI,
				imu.raw.gy * 180.0f / M_PI,
				imu.raw.gz * 180.0f / M_PI
			}
		};

        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

		const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&ahrs);
		euler = FusionQuaternionToEuler(quaternion);

		// update quaternion
		imu.state.qw = quaternion.element.w;
		imu.state.qx = quaternion.element.x;
		imu.state.qy = quaternion.element.y;
		imu.state.qz = quaternion.element.z;

		// perform dead reckoning
		FusionVector accel = FusionAhrsGetEarthAcceleration(&ahrs);
		// convert to m/s^2
		imu.state.ax = accel.axis.x * ATL_LOCAL_G;
		imu.state.ay = accel.axis.y * ATL_LOCAL_G;
		imu.state.az = accel.axis.z * ATL_LOCAL_G;

		if (fabs(imu.state.ax*imu.state.ax + imu.state.ay*imu.state.ay + imu.state.az*imu.state.az) < 1) {
			imu.state.ax = 0;
			imu.state.ay = 0;
			imu.state.az = 0;
			imu.state.vx *= 0.9;
			imu.state.vy *= 0.9;
			imu.state.vz *= 0.9;
		}

		imu.state.vx += imu.state.ax * SAMPLE_PERIOD;
		imu.state.vy += imu.state.ay * SAMPLE_PERIOD;
		imu.state.vz += imu.state.az * SAMPLE_PERIOD;

		imu.state.px += imu.state.vx * SAMPLE_PERIOD;
		imu.state.py += imu.state.vy * SAMPLE_PERIOD;
		imu.state.pz += imu.state.vz * SAMPLE_PERIOD;

		// send position over UART
		// sprintf(out_str, "p x:%.1f y:%.1f z:%.1f\n", imu.state.px, imu.state.py, imu.state.pz);
		if (!uartSending) {
			uartSending = true;
			frame.fdata[0] = imu.state.qw;
			frame.fdata[1] = imu.state.qx;
			frame.fdata[2] = imu.state.qy;
			frame.fdata[3] = imu.state.qz;
			// sprintf(out_str, "px:%.6f,%.6f,%.6f,%.6f\n\0", imu.state.qw, imu.state.qx, imu.state.qy, imu.state.qz);
			sprintf(out_str, "px:%.6f,%.6f,%.6f\n\0", imu.state.px, imu.state.py, imu.state.pz);
			uart_tx(uart_dev, out_str, strlen(out_str), SYS_FOREVER_US);
			// uart_tx(uart_dev, (const char *)&frame, sizeof(frame), SYS_FOREVER_US);
		}

        count++;
        k_msleep(1000 * SAMPLE_PERIOD);
    }
}

K_THREAD_DEFINE(imu_tid, 128 * 20,
                imu_task, NULL, NULL, NULL,
                5, 0, 0);