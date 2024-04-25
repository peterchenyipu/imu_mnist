#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <state_machine.h>

#define MY_STACK_SIZE 500
#define MY_PRIORITY 5
#define SLEEP_TIME_MS 10


#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec red_led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

#define LED1_NODE DT_ALIAS(led1)
static const struct gpio_dt_spec green_led = GPIO_DT_SPEC_GET(LED1_NODE, gpios);

#define LED2_NODE DT_ALIAS(led2)
static const struct gpio_dt_spec blue_led = GPIO_DT_SPEC_GET(LED2_NODE, gpios);

#define RUN_LED_PERIOD 1000
#define PAIR_LED_PERIOD 100
#define CONNECTED_LED_PERIOD 2000
#define COLLECT_LED_PERIOD 100

bool run_blink = false;
int my_entry_point(void *, void *, void *)
{
    int ret;
	int counter = 0;

	gpio_pin_configure_dt(&red_led, GPIO_OUTPUT_ACTIVE);
	gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
	gpio_pin_configure_dt(&blue_led, GPIO_OUTPUT_ACTIVE);
	gpio_pin_set_dt(&red_led, 0);
	gpio_pin_set_dt(&green_led, 0);
	gpio_pin_set_dt(&blue_led, 0);

	while (1) {

		if (run_blink && counter % RUN_LED_PERIOD == 0)
		{
			ret = gpio_pin_toggle_dt(&red_led);
		}

		// blue led states
		if (state == PAIR)
		{
			if (counter % PAIR_LED_PERIOD == 0)
			{
				ret = gpio_pin_toggle_dt(&blue_led);
			}
		} else if (state != START && state != DISCONNECT_DISPLAY)
		{
			if (counter % CONNECTED_LED_PERIOD > 50)
			{
				ret = gpio_pin_set_dt(&blue_led, 0);
			} else 
			{
				ret = gpio_pin_set_dt(&blue_led, 1);
			}
		}

		// green led states
		if (state == IDLE)
		{
			ret = gpio_pin_set_dt(&green_led, 1);
		} else if (state == COLLECT_DATA)
		{
			if (counter % COLLECT_LED_PERIOD == 0)
			{
				ret = gpio_pin_toggle_dt(&green_led);
			}
		} else
		{
			ret = gpio_pin_set_dt(&green_led, 0);
		}

		counter += SLEEP_TIME_MS;
		k_msleep(SLEEP_TIME_MS);
	}
	return 0;
}

K_THREAD_DEFINE(my_tid, MY_STACK_SIZE,
                my_entry_point, NULL, NULL, NULL,
                MY_PRIORITY, 0, 0);