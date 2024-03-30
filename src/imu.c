#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <imu.h>

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
#if defined(CONFIG_LSM6DSL_EXT0_LIS2MDL)
static struct sensor_value magn_x_out, magn_y_out, magn_z_out;
#endif
#if defined(CONFIG_LSM6DSL_EXT0_LPS22HB)
static struct sensor_value press_out, temp_out;
#endif

#ifdef CONFIG_LSM6DSL_TRIGGER
static void lsm6dsl_trigger_handler(const struct device *dev,
				    const struct sensor_trigger *trig)
{
	static struct sensor_value accel_x, accel_y, accel_z;
	static struct sensor_value gyro_x, gyro_y, gyro_z;
#if defined(CONFIG_LSM6DSL_EXT0_LIS2MDL)
	static struct sensor_value magn_x, magn_y, magn_z;
#endif
#if defined(CONFIG_LSM6DSL_EXT0_LPS22HB)
	static struct sensor_value press, temp;
#endif
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

#if defined(CONFIG_LSM6DSL_EXT0_LIS2MDL)
	/* lsm6dsl external magn */
	sensor_sample_fetch_chan(dev, SENSOR_CHAN_MAGN_XYZ);
	sensor_channel_get(dev, SENSOR_CHAN_MAGN_X, &magn_x);
	sensor_channel_get(dev, SENSOR_CHAN_MAGN_Y, &magn_y);
	sensor_channel_get(dev, SENSOR_CHAN_MAGN_Z, &magn_z);
#endif

#if defined(CONFIG_LSM6DSL_EXT0_LPS22HB)
	/* lsm6dsl external press/temp */
	sensor_sample_fetch_chan(dev, SENSOR_CHAN_PRESS);
	sensor_channel_get(dev, SENSOR_CHAN_PRESS, &press);

	sensor_sample_fetch_chan(dev, SENSOR_CHAN_AMBIENT_TEMP);
	sensor_channel_get(dev, SENSOR_CHAN_AMBIENT_TEMP, &temp);
#endif

	// if (print_samples) {
	// 	print_samples = 0;

		accel_x_out = accel_x;
		accel_y_out = accel_y;
		accel_z_out = accel_z;

		gyro_x_out = gyro_x;
		gyro_y_out = gyro_y;
		gyro_z_out = gyro_z;

#if defined(CONFIG_LSM6DSL_EXT0_LIS2MDL)
		magn_x_out = magn_x;
		magn_y_out = magn_y;
		magn_z_out = magn_z;
#endif

#if defined(CONFIG_LSM6DSL_EXT0_LPS22HB)
		press_out = press;
		temp_out = temp;
#endif
	// }

}
#endif

#define ATL_LOCAL_G (9.80665f)


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
    FusionAhrsInitialise(&ahrs);

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
		const FusionVector accelerometer = {
			.array = {
				out_ev(&accel_x_out)/ATL_LOCAL_G,
				out_ev(&accel_y_out)/ATL_LOCAL_G,
				out_ev(&accel_z_out)/ATL_LOCAL_G
			}
		};
        const FusionVector gyroscope = {
			.array = {
				out_ev(&gyro_x_out) * 180.0f / M_PI,
				out_ev(&gyro_y_out) * 180.0f / M_PI,
				out_ev(&gyro_z_out) * 180.0f / M_PI
			}
		};

        FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, SAMPLE_PERIOD);

		const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&ahrs);
		euler = FusionQuaternionToEuler(quaternion);
        count++;
        k_msleep(1000 * SAMPLE_PERIOD);
    }
}

K_THREAD_DEFINE(imu_tid, 8192*10,
                imu_task, NULL, NULL, NULL,
                5, 0, 0);