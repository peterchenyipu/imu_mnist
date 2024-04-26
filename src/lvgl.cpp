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

volatile State state = START;

#define LOG_LEVEL CONFIG_LOG_DEFAULT_LEVEL
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(app);


LV_FONT_DECLARE(jb_mono_bold);

#define RIGHT_DOWN_ARROW "\xE2\x86\x98"
#define BLUETOOTH_ICON "\xEF\x8A\x94"
volatile bool timer_submitted = false;
volatile bool timer_expired = false;

#define SW0_NODE DT_ALIAS(sw0)

void my_timer_handler(struct k_timer *dummy)
{
	timer_expired = true;
}
K_TIMER_DEFINE(my_timer, my_timer_handler, NULL);

static lv_obj_t *bar;
static lv_obj_t *middle_label;
static lv_obj_t *hint_label;

void create_labels(void)
{
	static lv_style_t style;
	lv_style_init(&style);
	lv_style_set_text_font(&style, &jb_mono_bold);
	lv_obj_add_style(lv_scr_act(), &style, 0);


	middle_label = lv_label_create(lv_scr_act());

	lv_label_set_text(middle_label, "Welcome to IMU Keyboard!");
	
	lv_label_set_long_mode(middle_label, LV_LABEL_LONG_WRAP);
	lv_obj_set_width(middle_label, 128);
	lv_obj_set_style_text_align(middle_label, LV_TEXT_ALIGN_CENTER, 0);
	lv_obj_align(middle_label, LV_ALIGN_TOP_MID, 0, 0);

	
	
	hint_label = lv_label_create(lv_scr_act());
	
	lv_obj_set_width(hint_label, 128);
	lv_label_set_long_mode(hint_label, LV_LABEL_LONG_SCROLL_CIRCULAR);
	lv_obj_align(hint_label, LV_ALIGN_BOTTOM_MID, 0, 0);
}

void create_progress_bar(void)
{
		// progress bar
	static lv_style_t style_bg;
    static lv_style_t style_indic;

    /* Initialize and configure the background style for the bar */
    lv_style_init(&style_bg);
    lv_style_set_bg_color(&style_bg, lv_color_white()); // Background color to white
    lv_style_set_border_color(&style_bg, lv_color_black()); // Black border for visibility
    lv_style_set_border_width(&style_bg, 2);
    lv_style_set_pad_all(&style_bg, 6); /* To make the indicator smaller and create padding inside the bar */
    lv_style_set_radius(&style_bg, 6); // Rounded corners for aesthetic
    lv_style_set_anim_time(&style_bg, 1000); // Animation time for value changes

    /* Initialize and configure the indicator style */
    lv_style_init(&style_indic);
    lv_style_set_bg_opa(&style_indic, LV_OPA_COVER);
    lv_style_set_bg_color(&style_indic, lv_color_black()); // Indicator color to black for contrast
    lv_style_set_radius(&style_indic, 3); // Slightly rounded corners for the indicator

	/* Create a container or use the main screen object */
    lv_obj_t * scr = lv_scr_act(); // Get the active screen object

    /* Create a progress bar */
    bar = lv_bar_create(scr);

    /* Set the size of the progress bar */
    lv_obj_set_size(bar, 128, 14);

    /* Get the display size */
    lv_coord_t disp_width = lv_obj_get_width(scr);
    lv_coord_t disp_height = lv_obj_get_height(scr);

    /* Calculate the y position for the bar to be at the bottom of the screen */
    lv_coord_t bar_y = disp_height - 28;

    /* Set the position of the progress bar */
    lv_obj_set_pos(bar, (disp_width - 128) / 2, bar_y); // Center the bar horizontally

    /* Optional: Configure the range and the initial value */
    lv_bar_set_range(bar, 0, 300);
    lv_bar_set_value(bar, 0, LV_ANIM_ON); // Set initial value to 0, animation off
	/* Apply the background style */
    lv_obj_add_style(bar, &style_bg, 0);

    /* Apply the indicator style to the indicator part of the bar */
    lv_obj_add_style(bar, &style_indic, LV_PART_INDICATOR);

}


static void hide_progress_bar(void) {
    lv_obj_add_flag(bar, LV_OBJ_FLAG_HIDDEN);
}

static void show_progress_bar(void) {
    lv_obj_clear_flag(bar, LV_OBJ_FLAG_HIDDEN);
}


int lvgl_task(void)
{
	const struct device *display_dev;
	
	display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
	if (!device_is_ready(display_dev)) {
		LOG_ERR("Device not ready, aborting test");
		return 0;
	}

	// create LVGL elements
	create_progress_bar();
	create_labels();
	hide_progress_bar();

	lv_task_handler();

	display_blanking_off(display_dev);

	// configure button
	const struct gpio_dt_spec sw0 = GPIO_DT_SPEC_GET(SW0_NODE, gpios);
	gpio_pin_configure_dt(&sw0, GPIO_INPUT);
	int last_button_state = 0;

	while (1) {
		int this_button_state = gpio_pin_get_dt(&sw0);
		State next_state = state;

		if (state == START)
		{
			next_state = PAIR;
		}
		else if (state == PAIR)
		{
			lv_label_set_text_fmt(hint_label, "Waiting for %s. Device name: %s.", BLUETOOTH_ICON, CONFIG_BT_DEVICE_NAME);
			if (ble_connected)
			{
				next_state = PAIR_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s Connected to %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}
		}
		else if (state == PAIR_DISPLAY)
		{
			if (timer_expired)
			{
				timer_expired = false;
				next_state = IDLE;
			}
			if (!ble_connected)
			{
				next_state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}
		}
		else if (state == DISCONNECT_DISPLAY)
		{
			if (timer_expired)
			{
				timer_expired = false;
				next_state = PAIR;
			}
		}
		else if (state == IDLE)
		{
			lv_label_set_text_fmt(hint_label, "Press button to start writing %s.", RIGHT_DOWN_ARROW);
			
			if (last_button_state == 1 && this_button_state == 0) // transition to collect data
			{
				next_state = COLLECT_DATA;
				show_progress_bar();
				lv_label_set_text(hint_label, "Collecting data...");
			}
			
			
			if (!ble_connected)
			{
				next_state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}
		
		} else if (state == COLLECT_DATA)
		{
			if (features_ready)
			{
				next_state = INFERENCE;
				hide_progress_bar();
				lv_label_set_text(hint_label, "Inferencing...");
			}

			lv_bar_set_value(bar, features_counter, LV_ANIM_ON);

			if (!ble_connected)
			{
				next_state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}

		} else if (state == INFERENCE)
		{
			if (inference_done)
			{
				next_state = DISPLAY_INFERENCE;
				features_ready = false;
				lv_label_set_text_fmt(hint_label, "Result: %c.", inference_result + '0');
				k_timer_start(&my_timer, K_SECONDS(3), K_NO_WAIT);
			}

			if (!ble_connected)
			{
				next_state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}
		} else if (state == DISPLAY_INFERENCE)
		{
			if (timer_expired)
			{
				timer_expired = false;
				next_state = IDLE;
			}

			if (last_button_state == 1 && this_button_state == 0) // transition to collect data
			{
				next_state = COLLECT_DATA;
				k_timer_stop(&my_timer);
				show_progress_bar();
				lv_label_set_text(hint_label, "Collecting data...");
			}

			if (!ble_connected)
			{
				next_state = DISCONNECT_DISPLAY;
				lv_label_set_text_fmt(hint_label, "%s disconnected from %s.", BLUETOOTH_ICON, ble_master_name);
				timer_expired = false;
				k_timer_start(&my_timer, K_SECONDS(5), K_NO_WAIT);
			}
		}


		last_button_state = this_button_state;
		state = next_state;
		lv_task_handler();
		k_sleep(K_MSEC(5));
	}
}



#define LVGL_STACK_SIZE 4096
#define LVGL_PRIORITY 2

K_THREAD_DEFINE(lvgl_tid, LVGL_STACK_SIZE,
                lvgl_task, NULL, NULL, NULL,
                LVGL_PRIORITY, 0, 0);