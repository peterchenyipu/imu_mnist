/** @file
 *  @brief HoG Service sample
 */

/*
 * Copyright (c) 2016 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

void hog_init(void);

void hog_button_loop(void);

typedef uint8_t zmk_mod_flags_t;
typedef uint8_t zmk_mouse_button_flags_t;
struct zmk_hid_keyboard_report_body {
    zmk_mod_flags_t modifiers;
    uint8_t _reserved;
    uint8_t keys[6]; // HKRO
} __packed;
typedef struct zmk_hid_keyboard_report_body zmk_hid_keyboard_report_body_t;

struct zmk_hid_keyboard_report {
    uint8_t report_id;
    struct zmk_hid_keyboard_report_body body;
} __packed;


struct zmk_hid_consumer_report_body {
    uint16_t keys[6];
} __packed;

struct zmk_hid_consumer_report {
    uint8_t report_id;
    struct zmk_hid_consumer_report_body body;
} __packed;

struct zmk_hid_mouse_report_body {
    zmk_mouse_button_flags_t buttons;
    int8_t d_x;
    int8_t d_y;
    int8_t d_wheel;
} __packed;
typedef struct zmk_hid_mouse_report_body zmk_hid_mouse_report_body_t;

struct zmk_hid_mouse_report {
    uint8_t report_id;
    struct zmk_hid_mouse_report_body body;
} __packed;


/**
 * @brief simulate a key press, modify the report
 * 
 * @param k key code
 * @param report pointer to the report
 * @return uint8_t 0 for fail, 1 for success
 */
uint8_t press(uint8_t k, zmk_hid_keyboard_report_body_t *report);

/**
 * @brief simulate a key release, modify the report
 * 
 * @param k key code
 * @param report pointer to the report
 * @return uint8_t 0 for fail, 1 for success
 */
uint8_t release(uint8_t k, zmk_hid_keyboard_report_body_t *report);

/**
 * @brief write a key to the host
 * 
 * @param k key code
 */
void write(uint8_t k);

/**
 * @brief write a string to the host
 * 
 * @param str string
 */
void write_string(const char *str);

#ifdef __cplusplus
}
#endif
