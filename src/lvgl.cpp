/*
 * Copyright (c) 2018 Jan Van Winkel <jan.van_winkel@dxplore.eu>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/display.h>
#include <zephyr/drivers/gpio.h>
#include <lvgl.h>
#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <lvgl_input_device.h>
#include <imu.h>
#include <state_machine.h>
#include <hog.h>
#include <main.h>

State state = START;

#define LOG_LEVEL CONFIG_LOG_DEFAULT_LEVEL
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(app);

static uint32_t count;

#ifdef CONFIG_GPIO
static struct gpio_dt_spec button_gpio = GPIO_DT_SPEC_GET_OR(
		DT_ALIAS(sw0), gpios, {0});
static struct gpio_callback button_callback;

static void button_isr_callback(const struct device *port,
				struct gpio_callback *cb,
				uint32_t pins)
{
	ARG_UNUSED(port);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);

	count = 0;
}
#endif /* CONFIG_GPIO */

#ifdef CONFIG_LV_Z_ENCODER_INPUT
static const struct device *lvgl_encoder =
	DEVICE_DT_GET(DT_COMPAT_GET_ANY_STATUS_OKAY(zephyr_lvgl_encoder_input));
#endif /* CONFIG_LV_Z_ENCODER_INPUT */

static void lv_btn_click_callback(lv_event_t *e)
{
	ARG_UNUSED(e);

	count = 0;
}

LV_FONT_DECLARE(jb_mono_bold);
// extern lv_font_t jb_mono_bold;
static lv_style_t style;
#define RIGHT_DOWN_ARROW "\xE2\x86\x98"
#define BLUETOOTH_ICON "\xEF\x8A\x94"
bool timer_submitted = false;
bool timer_expired = false;

#define SW0_NODE DT_ALIAS(sw0)

void my_timer_handler(struct k_timer *dummy)
{
	timer_expired = true;
}
K_TIMER_DEFINE(my_timer, my_timer_handler, NULL);

int lvgl_task(void)
{
	char count_str[11] = {0};
	const struct device *display_dev;
	lv_obj_t *middle_label;
	lv_obj_t *hint_label;

	display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
	if (!device_is_ready(display_dev)) {
		LOG_ERR("Device not ready, aborting test");
		return 0;
	}

#ifdef CONFIG_GPIO
	if (gpio_is_ready_dt(&button_gpio)) {
		int err;

		err = gpio_pin_configure_dt(&button_gpio, GPIO_INPUT);
		if (err) {
			LOG_ERR("failed to configure button gpio: %d", err);
			return 0;
		}

		gpio_init_callback(&button_callback, button_isr_callback,
				   BIT(button_gpio.pin));

		err = gpio_add_callback(button_gpio.port, &button_callback);
		if (err) {
			LOG_ERR("failed to add button callback: %d", err);
			return 0;
		}

		err = gpio_pin_interrupt_configure_dt(&button_gpio,
						      GPIO_INT_EDGE_TO_ACTIVE);
		if (err) {
			LOG_ERR("failed to enable button callback: %d", err);
			return 0;
		}
	}
#endif /* CONFIG_GPIO */

	lv_style_init(&style);
	lv_style_set_text_font(&style, &jb_mono_bold);
	lv_obj_add_style(lv_scr_act(), &style, 0);


	middle_label = lv_label_create(lv_scr_act());

	lv_label_set_text(middle_label, "Welcome to IMU Keyboard!");
	
	lv_label_set_long_mode(middle_label, LV_LABEL_LONG_WRAP);
	lv_obj_set_width(middle_label, 128);
	lv_obj_align(middle_label, LV_ALIGN_TOP_MID, 0, 0);

	
	
	hint_label = lv_label_create(lv_scr_act());
	
	lv_obj_set_width(hint_label, 128);
	lv_label_set_long_mode(hint_label, LV_LABEL_LONG_SCROLL_CIRCULAR);
	lv_obj_align(hint_label, LV_ALIGN_BOTTOM_MID, 0, 0);

	lv_task_handler();

	display_blanking_off(display_dev);
	char out_str[64];

	const struct gpio_dt_spec sw0 = GPIO_DT_SPEC_GET(SW0_NODE, gpios);
	gpio_pin_configure_dt(&sw0, GPIO_INPUT);
	int last_button_state = 0;

	while (1) {
		int this_button_state = gpio_pin_get_dt(&sw0);


		if (state == START)
		{
			state = PAIR;
		}
		else if (state == PAIR)
		{
			lv_label_set_text_fmt(hint_label, "Waiting for Bluetooth connection %s. Device name: %s.", BLUETOOTH_ICON, CONFIG_BT_DEVICE_NAME);
			if (ble_connected)
			{
				state = PAIR_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s Connected to %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(8), K_NO_WAIT);
			}
		}
		else if (state == PAIR_DISPLAY)
		{
			if (timer_expired)
			{
				timer_expired = false;
				state = IDLE;
			}
			if (!ble_connected)
			{
				state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s Bluetooth disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(8), K_NO_WAIT);
			}
		}
		else if (state == DISCONNECT_DISPLAY)
		{
			if (timer_expired)
			{
				timer_expired = false;
				state = PAIR;
			}
		}
		else if (state == IDLE)
		{
			lv_label_set_text_fmt(hint_label, "Press lower right button to start writing %s,%s.", RIGHT_DOWN_ARROW, BLUETOOTH_ICON);
			
			if (last_button_state == 1 && this_button_state == 0) // transition to collect data
			{
				state = COLLECT_DATA;
				lv_label_set_text(hint_label, "Collecting data.");
			}
			
			
			if (!ble_connected)
			{
				state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s Bluetooth disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(8), K_NO_WAIT);
			}
		
		} else if (state == COLLECT_DATA)
		{
			if (features_ready)
			{
				state = INFERENCE;
				lv_label_set_text(hint_label, "Inferencing...");
			}
		} else if (state == INFERENCE)
		{
			if (inference_done)
			{
				state = DISPLAY_INFERENCE;
				lv_label_set_text_fmt(hint_label, "Inference Result: %c.", inference_result + '0');
				k_timer_start(&my_timer, K_SECONDS(1), K_NO_WAIT);
			}
		} else if (state == DISPLAY_INFERENCE)
		{
			if (timer_expired)
			{
				timer_expired = false;
				state = IDLE;
			}
		}


		last_button_state = this_button_state;

		lv_task_handler();
		++count;
		k_sleep(K_MSEC(10));
	}
}



#define LVGL_STACK_SIZE 4096
#define LVGL_PRIORITY 5

K_THREAD_DEFINE(lvgl_tid, LVGL_STACK_SIZE,
                lvgl_task, NULL, NULL, NULL,
                LVGL_PRIORITY, 0, 0);