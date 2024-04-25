#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <imu.h>
#include <zephyr/drivers/uart.h>
#include <state_machine.h>

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
	printk("lsm6dsl_trig_cnt: %d\n", lsm6dsl_trig_cnt);

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

IMU imu = {0};

#define CH_COUNT 7
struct Frame {
	float fdata[CH_COUNT];
	unsigned char tail[4];
} __attribute__((packed));

struct Frame frame = {
	.tail = {0x00, 0x00, 0x80, 0x7f}
};


float features[1800] = {0};
bool features_ready = false;
bool collecting = false;

static int features_counter = 0;

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
	// odr_attr.val1 = 104;
	// odr_attr.val2 = 0;

	// if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_ACCEL_XYZ,
	// 		    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
	// 	printk("Cannot set sampling frequency for accelerometer.\n");
	// 	return 0;
	// }

	// if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_GYRO_XYZ,
	// 		    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
	// 	printk("Cannot set sampling frequency for gyro.\n");
	// 	return 0;
	// }

#ifdef CONFIG_LSM6DSL_TRIGGER
	struct sensor_trigger trig;

	trig.type = SENSOR_TRIG_DATA_READY;
	trig.chan = SENSOR_CHAN_ACCEL_XYZ;

	if (sensor_trigger_set(lsm6dsl_dev, &trig, lsm6dsl_trigger_handler) != 0) {
		printk("Could not set sensor type and channel\n");
		while (1)
			;
	}
#endif

	if (sensor_sample_fetch(lsm6dsl_dev) < 0) {
		printk("Sensor sample update error\n");
		while (1)
			;
	}
	
    uint32_t count = 0;

	FusionAhrs ahrs;
	// TODO add calibration here
    FusionAhrsInitialise(&ahrs);

	

    while (1)
    {
		if (state == COLLECT_DATA)
		{
			if (!collecting)
			{
				printk("Start Collecting data\n");
				collecting = true;
				features_counter = 0;
				features_ready = false;
			}
			uint32_t start = k_cycle_get_32();
			
			// Update IMU
			// accel in m/s^2, gyro in rad/s
			imu.raw.ax = out_ev(&accel_x_out);
			imu.raw.ay = out_ev(&accel_y_out);
			imu.raw.az = out_ev(&accel_z_out);
			imu.raw.gx = out_ev(&gyro_x_out);
			imu.raw.gy = out_ev(&gyro_y_out);
			imu.raw.gz = out_ev(&gyro_z_out);

			// fill the features array
			features[features_counter*6] = imu.raw.ax;
			features[features_counter*6+1] = imu.raw.ay;
			features[features_counter*6+2] = imu.raw.az;
			features[features_counter*6+3] = imu.raw.gx;
			features[features_counter*6+4] = imu.raw.gy;
			features[features_counter*6+5] = imu.raw.gz;
			features_counter++;

			if (features_counter >= 300) {
				features_ready = true;
				features_counter = 0;
				collecting = false;
				printk("Features ready\n");
			}


			// const FusionVector accelerometer = {
			// 	.array = {
			// 		imu.raw.ax / ATL_LOCAL_G,
			// 		imu.raw.ay / ATL_LOCAL_G,
			// 		imu.raw.az / ATL_LOCAL_G
			// 	}
			// };
			// const FusionVector gyroscope = {
			// 	.array = {
			// 		imu.raw.gx * 180.0f / M_PI,
			// 		imu.raw.gy * 180.0f / M_PI,
			// 		imu.raw.gz * 180.0f / M_PI
			// 	}
			// };

			// FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

			// const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&ahrs);
			// euler = FusionQuaternionToEuler(quaternion);

			// // update quaternion
			// imu.state.qw = quaternion.element.w;
			// imu.state.qx = quaternion.element.x;
			// imu.state.qy = quaternion.element.y;
			// imu.state.qz = quaternion.element.z;

			// // perform dead reckoning
			// FusionVector accel = FusionAhrsGetEarthAcceleration(&ahrs);
			// // convert to m/s^2
			// imu.state.ax = accel.axis.x * ATL_LOCAL_G;
			// imu.state.ay = accel.axis.y * ATL_LOCAL_G;
			// imu.state.az = accel.axis.z * ATL_LOCAL_G;

			// if (fabs(imu.state.ax*imu.state.ax + imu.state.ay*imu.state.ay + imu.state.az*imu.state.az) < 1) {
			// 	imu.state.ax = 0;
			// 	imu.state.ay = 0;
			// 	imu.state.az = 0;
			// 	imu.state.vx *= 0.9;
			// 	imu.state.vy *= 0.9;
			// 	imu.state.vz *= 0.9;
			// }

			// imu.state.vx += imu.state.ax * SAMPLE_PERIOD;
			// imu.state.vy += imu.state.ay * SAMPLE_PERIOD;
			// imu.state.vz += imu.state.az * SAMPLE_PERIOD;

			// imu.state.px += imu.state.vx * SAMPLE_PERIOD;
			// imu.state.py += imu.state.vy * SAMPLE_PERIOD;
			// imu.state.pz += imu.state.vz * SAMPLE_PERIOD;
			uint32_t end = k_cycle_get_32();
			int64_t elapsed_time = end - start;
			elapsed_time = k_cyc_to_us_floor64(elapsed_time);
			// k_msleep(1000 * SAMPLE_PERIOD - elapsed_time);
			k_usleep(1000000 * SAMPLE_PERIOD - elapsed_time);
		}
		else
		{
			k_msleep(100); // wait for inference
		}
    }
}

K_THREAD_DEFINE(imu_tid, 128 * 10,
                imu_task, NULL, NULL, NULL,
                5, 0, 0);