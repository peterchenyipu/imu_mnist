#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

#define MY_STACK_SIZE 500
#define MY_PRIORITY 5
#define SLEEP_TIME_MS 1000

#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

#define LED1_NODE DT_ALIAS(led1)
static const struct gpio_dt_spec led1 = GPIO_DT_SPEC_GET(LED1_NODE, gpios);

int my_entry_point(void *, void *, void *)
{
    int ret;

	if (!gpio_is_ready_dt(&led)) {
		return 0;
	}

	ret = gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);

	if (ret < 0) {
		return 0;
	}

	while (1) {
		ret = gpio_pin_toggle_dt(&led);
		if (ret < 0) {
			return 0;
		}
		k_msleep(SLEEP_TIME_MS);
	}
	return 0;
}

K_THREAD_DEFINE(my_tid, MY_STACK_SIZE,
                my_entry_point, NULL, NULL, NULL,
                MY_PRIORITY, 0, 0);