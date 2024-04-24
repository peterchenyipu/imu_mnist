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
	lv_label_set_text_fmt(hint_label, "Press lower right button to start writing %s,%s.", RIGHT_DOWN_ARROW, BLUETOOTH_ICON);
	lv_obj_set_width(hint_label, 128);
	lv_label_set_long_mode(hint_label, LV_LABEL_LONG_SCROLL_CIRCULAR);
	lv_obj_align(hint_label, LV_ALIGN_BOTTOM_MID, 0, 0);

	lv_task_handler();

	display_blanking_off(display_dev);
	char out_str[64];

	while (1) {

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