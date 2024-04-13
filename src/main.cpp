/*
 * Copyright (c) 2021 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>


int main(void)
{
	

	printk("Main launched\n");

	while (1) {
		k_sleep(K_FOREVER);
	}

	return 0;
}
