// Zpehyr 3.1.x and newer uses different include scheme
#include <version.h>
#if (KERNEL_VERSION_MAJOR > 3) || ((KERNEL_VERSION_MAJOR == 3) && (KERNEL_VERSION_MINOR >= 1))
#include <zephyr/kernel.h>
#else
#include <zephyr.h>
#endif
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include <nrfx_clock.h>
#include <imu.h>

// static const float features[] = {
//     // copy raw features here (for example from the 'Live classification' page)
//     // see https://docs.edgeimpulse.com/docs/running-your-impulse-locally-zephyr
//     -4.1990, -2.1660, -8.3510, -0.0180, -0.0320, -0.0200, -4.1660, -2.2100, -8.4350, -0.0320, -0.0360, -0.0170, -4.1520, -2.2480, -8.5480, -0.0330, -0.0430, -0.0120, -4.1280, -2.2290, -8.5620, -0.0330, -0.0500, -0.0100, -4.1070, -2.1140, -8.4460, -0.0440, -0.0550, -0.0020, -4.1570, -2.1120, -8.4980, -0.0640, -0.0570, -0.0120, -4.2190, -2.2790, -8.4950, -0.0410, -0.0500, -0.0300, -4.1360, -2.4160, -8.4430, 0.0070, -0.0430, -0.0230, -4.2010, -2.1450, -8.5250, 0.0150, -0.0290, 0.0160, -4.2250, -2.1280, -8.5440, 0.0140, -0.0270, 0.0110, -4.1590, -2.1080, -8.5370, 0.0210, -0.0230, 0.0030, -4.1600, -2.0780, -8.4950, 0.0390, -0.0190, 0.0000, -4.0810, -2.1600, -8.4860, 0.0300, -0.0250, -0.0010, -4.1090, -2.1870, -8.6160, 0.0200, -0.0310, -0.0020, -4.1070, -2.1920, -8.6530, 0.0280, -0.0350, -0.0130, -4.1330, -2.1050, -8.5860, 0.0220, -0.0390, -0.0040, -4.1330, -2.1180, -8.5510, 0.0050, -0.0440, -0.0020, -4.1050, -2.1340, -8.5510, -0.0010, -0.0490, -0.0060, -4.0970, -2.2330, -8.5560, 0.0130, -0.0470, -0.0080, -4.0590, -2.2030, -8.5060, 0.0280, -0.0490, -0.0010, -4.0760, -2.1850, -8.4840, 0.0060, -0.0550, 0.0060, -4.1100, -2.2090, -8.4890, 0.0030, -0.0560, 0.0110, -4.1130, -2.1530, -8.4740, 0.0290, -0.0590, 0.0080, -4.1390, -2.0660, -8.4400, 0.0380, -0.0560, 0.0100, -4.2100, -2.0230, -8.3740, 0.0160, -0.0600, 0.0020, -4.2020, -2.0530, -8.4110, -0.0080, -0.0570, -0.0100, -4.2650, -2.3180, -8.5040, -0.0190, -0.0540, -0.0240, -4.1700, -2.4630, -8.4980, 0.0230, -0.0520, -0.0180, -4.1440, -2.2890, -8.4230, 0.0530, -0.0530, 0.0210, -4.1810, -2.0760, -8.4280, 0.0410, -0.0530, 0.0360, -4.1530, -2.0740, -8.4650, 0.0140, -0.0510, -0.0040, -4.1790, -2.3370, -8.5140, 0.0400, -0.0500, -0.0090, -4.1890, -2.3250, -8.3920, 0.0410, -0.0420, 0.0020, -4.1620, -2.3210, -8.4690, 0.0290, -0.0430, 0.0140, -4.1770, -2.2770, -8.4320, 0.0320, -0.0440, 0.0210, -4.1580, -2.2520, -8.5050, 0.0440, -0.0420, 0.0340, -4.0970, -2.2160, -8.5250, 0.0510, -0.0470, 0.0340, -4.0770, -2.2220, -8.4910, 0.0470, -0.0490, 0.0300, -4.0910, -2.1170, -8.4870, 0.0310, -0.0470, 0.0300, -4.1470, -2.0470, -8.4570, -0.0010, -0.0570, 0.0250, -4.2300, -2.1450, -8.5680, -0.0110, -0.0570, 0.0020, -4.2150, -2.1960, -8.4350, 0.0100, -0.0540, -0.0060, -4.1860, -2.2090, -8.3760, 0.0210, -0.0460, 0.0040, -4.1590, -2.2700, -8.4480, 0.0290, -0.0460, 0.0090, -4.1300, -2.2330, -8.5010, 0.0440, -0.0440, 0.0190, -4.1150, -2.1680, -8.4670, 0.0440, -0.0480, 0.0270, -4.1290, -2.1520, -8.5230, 0.0350, -0.0500, 0.0220, -4.1260, -2.2340, -8.5780, 0.0250, -0.0530, 0.0120, -4.1090, -2.3070, -8.5320, 0.0420, -0.0590, 0.0080, -4.1190, -2.2540, -8.5090, 0.0510, -0.0600, 0.0150, -4.1380, -2.1710, -8.4680, 0.0380, -0.0690, 0.0190, -4.2170, -2.1700, -8.4620, 0.0190, -0.0750, 0.0120, -4.2350, -2.1480, -8.4200, 0.0300, -0.0730, -0.0030, -4.2720, -2.3220, -8.2960, 0.0060, -0.0660, -0.0080, -4.2300, -2.4460, -8.3590, 0.0170, -0.0560, 0.0000, -4.1330, -2.4580, -8.4440, 0.0500, -0.0520, 0.0230, -4.1560, -2.3000, -8.3930, 0.0640, -0.0540, 0.0500, -4.2340, -2.1250, -8.3550, 0.0550, -0.0550, 0.0490, -4.2620, -2.0910, -8.2910, 0.0350, -0.0580, 0.0230, -4.2410, -2.1150, -8.2720, 0.0130, -0.0520, 0.0070, -4.2720, -2.1060, -8.2840, -0.0160, -0.0470, -0.0050, -4.2850, -2.2140, -8.3530, -0.0340, -0.0370, -0.0170, -4.1730, -2.3360, -8.3550, -0.0210, -0.0320, -0.0140, -4.1930, -2.4080, -8.4040, -0.0130, -0.0370, -0.0010, -4.1850, -2.3150, -8.3940, -0.0110, -0.0470, 0.0170, -4.2940, -2.1600, -8.3060, -0.0180, -0.0610, 0.0060, -4.2630, -2.2230, -8.1620, -0.0150, -0.0530, 0.0000, -4.2980, -2.2500, -8.1430, -0.0210, -0.0410, 0.0000, -4.2530, -2.2980, -8.1600, -0.0230, -0.0370, 0.0010, -4.2210, -2.3240, -8.2370, -0.0210, -0.0370, 0.0070, -4.2750, -2.2600, -8.2350, -0.0120, -0.0300, 0.0190, -4.2930, -2.1360, -8.2570, -0.0200, -0.0300, 0.0190, -4.2540, -2.2160, -8.2490, -0.0450, -0.0350, 0.0050, -4.2880, -2.3110, -8.2490, -0.0480, -0.0360, 0.0060, -4.2590, -2.2070, -8.2090, -0.0350, -0.0200, 0.0140, -4.2320, -2.1550, -8.2070, -0.0370, -0.0170, 0.0170, -4.2200, -2.1610, -8.2450, -0.0420, -0.0140, 0.0160, -4.2760, -2.2600, -8.2470, -0.0460, -0.0090, 0.0040, -4.2780, -2.3820, -8.1920, -0.0380, -0.0030, 0.0280, -4.2920, -2.2980, -8.2640, -0.0320, 0.0050, 0.0410, -4.3150, -2.2670, -8.3260, -0.0160, 0.0140, 0.0510, -4.2750, -2.2940, -8.4350, -0.0030, 0.0210, 0.0540, -4.2380, -2.2670, -8.4620, -0.0080, 0.0200, 0.0570, -4.2670, -2.1990, -8.4860, -0.0290, 0.0210, 0.0550, -4.1910, -2.2660, -8.5090, -0.0400, 0.0210, 0.0370, -4.1790, -2.3440, -8.5420, -0.0320, 0.0320, 0.0310, -4.1270, -2.4020, -8.5070, -0.0250, 0.0360, 0.0360, -4.0900, -2.3380, -8.5840, -0.0280, 0.0280, 0.0480, -4.0610, -2.2190, -8.5070, -0.0270, 0.0210, 0.0500, -4.1490, -2.0200, -8.3610, -0.0600, 0.0120, 0.0330, -4.1230, -2.0930, -8.4240, -0.0820, 0.0100, 0.0150, -4.0820, -2.2580, -8.3890, -0.0830, 0.0110, 0.0130, -4.0120, -2.3290, -8.3570, -0.0760, 0.0170, 0.0280, -4.0070, -2.4080, -8.3970, -0.0650, 0.0130, 0.0410, -3.9300, -2.3930, -8.2370, -0.0610, 0.0080, 0.0600, -3.9200, -2.2450, -8.0770, -0.0620, 0.0080, 0.0800, -3.9110, -2.0690, -8.0500, -0.0880, 0.0070, 0.0790, -3.8910, -2.0610, -7.9200, -0.1160, 0.0070, 0.0580, -3.8330, -2.0760, -7.7360, -0.1430, 0.0060, 0.0400, -3.7590, -2.0660, -7.5580, -0.1660, 0.0030, 0.0360, -3.7200, -2.0470, -7.3140, -0.1810, 0.0000, 0.0330, -3.5610, -2.1710, -6.6230, -0.2000, 0.0050, 0.0410, -3.4500, -2.1870, -6.3040, -0.2200, 0.0220, 0.0520, -3.3050, -2.1290, -5.8940, -0.2570, 0.0290, 0.0530, -3.1320, -2.1270, -5.4910, -0.3240, 0.0400, 0.0620, -3.0280, -2.0040, -5.0510, -0.3960, 0.0520, 0.0600, -2.9270, -1.9050, -4.6800, -0.4620, 0.0540, 0.0510, -2.8910, -1.9220, -4.2950, -0.5100, 0.0590, 0.0390, -2.9610, -1.7200, -3.9870, -0.5830, 0.0490, 0.0360, -3.1810, -1.3860, -3.7750, -0.7110, 0.0310, 0.0490, -3.5010, -0.8390, -3.7120, -0.8600, 0.0020, 0.0360, -3.8430, -0.4000, -3.6110, -0.9780, -0.0440, -0.0170, -4.3080, -0.5070, -3.4240, -1.0460, -0.0050, -0.2070, -4.2100, -1.0790, -3.4840, -0.9340, 0.0320, -0.2920, -4.0070, -1.6030, -3.4360, -0.8090, 0.0730, -0.3140, -3.8750, -2.0880, -3.6290, -0.6970, 0.1100, -0.2440, -3.6440, -2.3910, -3.7890, -0.6090, 0.1290, -0.1340, -3.6500, -2.4780, -3.8050, -0.5510, 0.1710, -0.0260, -3.6310, -2.3090, -4.0100, -0.5450, 0.2100, 0.0500, -3.4880, -2.0820, -4.2750, -0.5640, 0.2390, 0.0960, -3.3500, -1.7610, -4.2940, -0.5970, 0.2610, 0.0860, -3.1710, -1.3090, -4.4300, -0.6620, 0.2480, 0.0320, -3.1980, -1.0600, -4.4090, -0.7420, 0.2240, -0.0560, -2.9030, -1.0970, -4.2850, -0.9760, 0.2850, -0.2680, -2.7150, -0.8880, -4.6670, -1.0790, 0.3230, -0.3060, -2.4520, -0.9970, -4.6530, -1.1640, 0.3690, -0.3490, -2.3870, -1.4370, -4.3690, -1.2530, 0.3940, -0.3830, -2.2220, -1.7070, -3.9490, -1.3440, 0.4780, -0.3340, -2.2100, -1.3850, -3.6840, -1.4940, 0.6400, -0.2060, -2.2830, -0.8110, -3.8010, -1.7100, 0.8290, -0.0840, -2.4070, -0.1250, -4.3250, -1.9280, 1.0230, 0.0210, -2.0810, 0.8750, -5.0230, -2.1020, 1.1830, 0.1240, -1.9750, 1.7030, -5.5090, -2.2540, 1.2690, 0.1560, -2.0390, 1.6920, -5.6440, -2.4170, 1.2510, 0.0870, -2.0110, 1.2900, -5.4580, -2.4910, 1.1640, -0.0100, -1.7840, -0.6430, -4.6890, -2.4820, 1.0850, 0.0530, -1.7710, -0.7770, -4.2180, -2.3080, 1.0840, 0.3010, -1.6220, -0.3130, -3.9400, -2.0300, 1.1920, 0.6670, -1.4050, 0.8470, -4.0760, -1.7590, 1.4250, 1.0440, -0.7940, 2.7720, -4.8230, -1.5610, 1.6930, 1.3000, 0.2200, 4.6770, -5.9390, -1.5670, 1.9560, 1.3430, 0.8060, 5.5540, -7.2140, -1.8160, 2.1400, 1.1770, 1.3560, 5.3810, -8.0010, -2.1910, 2.1960, 0.9170, 1.7240, 4.0800, -8.6850, -2.4280, 2.1370, 0.7280, 2.0150, 3.1800, -8.5270, -2.3870, 2.0520, 0.7510, 1.9890, 2.8930, -8.2240, -2.1380, 2.0660, 1.0100, 2.8480, 5.9130, -8.4070, -1.3240, 2.4510, 1.8500, 4.1400, 9.0340, -9.7680, -1.3570, 2.5010, 2.1520, 5.0620, 12.1350, -12.2390, -1.4970, 2.5010, 2.2730, 5.6480, 13.8290, -14.7690, -1.4680, 2.5010, 2.1640, 6.4420, 14.1140, -16.3560, -1.3860, 2.5010, 1.7460, 6.6990, 12.2210, -17.2830, -1.3430, 2.4770, 1.2610, 5.8010, 9.8920, -15.9200, -1.1690, 1.9080, 0.9220, 4.6960, 7.2060, -13.6430, -0.8850, 1.4340, 0.7210, 4.1660, 5.5220, -11.7200, -0.6340, 1.3140, 0.7680, 3.4950, 5.6900, -10.2750, -0.4720, 1.3780, 0.9610, 3.4880, 6.3080, -10.1930, -0.3910, 1.5620, 1.1220, 4.3180, 6.5510, -10.9340, -0.4220, 1.8150, 1.1160, 4.8960, 6.5520, -11.1520, -0.5170, 1.8380, 1.0570, 5.5500, 7.0270, -11.4550, -0.6700, 1.7970, 1.0040, 5.8220, 7.5150, -11.6150, -0.8580, 1.6760, 0.8900, 5.8860, 7.0250, -11.0230, -1.1260, 1.5290, 0.7340, 5.8100, 6.5040, -10.1000, -1.3950, 1.4200, 0.6290, 5.7310, 6.3500, -9.3900, -1.6120, 1.3840, 0.6120, 5.7900, 6.7800, -9.0140, -1.7000, 1.3940, 0.6170, 5.9890, 7.1840, -9.3110, -1.6710, 1.4420, 0.6040, 6.2390, 6.9180, -9.6830, -1.5520, 1.4710, 0.6040, 6.2820, 6.9370, -9.9230, -1.3320, 1.4750, 0.6420, 6.5910, 7.3500, -10.9050, -0.6210, 1.5420, 0.7430, 6.9320, 7.1940, -11.3920, -0.1890, 1.5550, 0.7870, 7.2360, 7.6800, -11.9650, 0.2210, 1.5290, 0.8190, 7.4400, 8.2510, -12.4570, 0.5040, 1.4330, 0.7830, 7.3990, 7.9100, -12.5270, 0.6190, 1.2920, 0.6420, 7.1670, 7.1190, -12.3390, 0.6470, 1.1280, 0.4900, 6.8060, 6.7380, -11.9370, 0.6310, 0.9740, 0.3900, 6.5630, 6.5740, -11.6910, 0.5310, 0.8730, 0.3070, 6.5470, 6.2830, -11.7850, 0.3470, 0.7820, 0.2190, 6.6770, 6.0920, -12.0310, 0.1250, 0.6650, 0.1580, 6.7320, 5.9140, -12.0770, -0.0930, 0.5160, 0.1140, 6.5270, 5.7490, -11.8760, -0.2760, 0.3450, 0.0820, 5.6500, 5.3410, -9.9370, -0.6820, -0.1040, 0.0360, 5.4100, 5.2640, -9.6680, -0.7030, -0.2030, 0.0470, 5.3450, 5.3580, -9.1870, -0.6750, -0.3010, 0.0600, 5.0540, 5.4170, -8.5380, -0.6550, -0.3710, 0.0710, 4.8240, 5.4290, -8.0680, -0.6270, -0.4160, 0.0770, 4.7980, 5.5800, -7.6330, -0.5880, -0.4570, 0.0780, 4.5990, 5.6990, -6.7340, -0.6350, -0.5100, 0.0440, 4.5060, 5.6250, -6.3380, -0.6710, -0.5120, 0.0310, 4.4710, 5.6440, -6.0460, -0.6700, -0.5000, 0.0250, 4.4160, 5.7510, -5.6730, -0.6300, -0.4680, 0.0140, 4.3030, 5.7330, -5.3560, -0.6130, -0.4250, 0.0000, 4.3660, 5.7130, -5.3090, -0.5770, -0.3630, -0.0110, 4.6100, 5.7600, -5.7150, -0.4640, -0.3140, -0.0200, 4.7640, 5.6890, -6.0520, -0.3200, -0.3160, -0.0400, 4.8140, 5.5120, -5.9680, -0.1650, -0.3360, -0.0560, 4.7380, 5.4430, -5.6910, -0.0530, -0.3470, -0.0460, 4.6280, 5.1900, -5.4650, -0.0140, -0.3440, -0.0370, 4.6010, 4.9460, -5.2580, -0.0190, -0.3280, -0.0280, 4.6870, 5.1640, -5.3150, -0.0870, -0.2710, 0.0200, 4.7610, 5.2050, -5.3980, -0.1430, -0.2550, 0.0260, 4.7250, 5.0750, -5.6440, -0.1380, -0.2330, 0.0320, 4.7700, 5.1640, -5.7070, -0.0570, -0.2050, 0.0340, 4.8320, 5.2260, -5.7560, 0.0050, -0.1810, 0.0550, 4.8280, 5.2220, -5.8040, 0.0570, -0.1480, 0.0700, 4.9560, 5.3710, -5.9930, 0.1050, -0.1200, 0.0740, 5.1000, 5.5190, -6.1660, 0.1440, -0.1240, 0.0690, 5.1600, 5.5010, -6.2470, 0.1450, -0.1400, 0.0400, 5.0930, 5.2890, -6.2720, 0.1360, -0.1630, 0.0170, 5.0130, 5.1460, -6.2000, 0.1340, -0.1770, 0.0200, 4.6940, 5.2740, -5.8930, 0.0690, -0.1700, 0.0150, 4.7580, 5.3400, -5.8030, 0.0100, -0.1450, 0.0000, 4.6950, 5.2320, -5.8770, -0.0190, -0.1290, -0.0040, 4.7000, 5.1690, -5.8180, -0.0050, -0.1030, -0.0060, 4.6780, 5.0650, -5.8430, 0.0280, -0.0790, 0.0080, 4.7430, 5.2110, -6.0400, 0.0760, -0.0570, 0.0380, 4.7140, 5.4130, -6.1520, 0.1240, -0.0410, 0.0410, 4.7520, 5.3100, -6.2300, 0.1430, -0.0330, 0.0220, 4.7730, 5.1620, -6.3340, 0.1480, -0.0370, 0.0230, 4.8610, 4.9950, -6.4020, 0.1530, -0.0500, 0.0500, 4.7570, 5.1630, -6.3370, 0.1370, -0.0610, 0.0730, 4.6150, 5.3700, -6.1950, 0.0170, -0.0660, 0.0480, 4.6170, 5.1820, -6.1050, -0.0400, -0.0540, 0.0290, 4.6280, 5.2140, -6.0590, -0.0790, -0.0360, 0.0360, 4.8190, 5.4400, -6.1850, -0.0860, -0.0280, 0.0510, 4.7680, 5.6050, -6.3130, -0.0760, -0.0180, 0.0460, 4.7560, 5.5610, -6.3180, -0.0600, -0.0190, 0.0250, 4.6940, 5.4160, -6.3250, -0.0500, -0.0160, 0.0210, 4.6720, 5.4360, -6.2840, -0.0420, 0.0050, 0.0300, 4.6320, 5.4460, -6.2200, -0.0490, 0.0230, 0.0260, 4.7120, 5.4520, -6.1550, -0.0700, 0.0350, 0.0250, 4.7400, 5.2910, -6.1980, -0.0800, 0.0510, 0.0200, 4.8070, 5.3990, -6.2780, -0.0190, 0.0660, 0.0520, 4.7840, 5.4020, -6.3750, -0.0030, 0.0710, 0.0590, 4.6950, 5.3030, -6.3260, 0.0160, 0.0670, 0.0630, 4.7160, 5.2720, -6.3320, 0.0500, 0.0560, 0.0630, 4.7780, 5.3640, -6.3360, 0.0460, 0.0530, 0.0660, 4.8010, 5.4560, -6.4540, 0.0180, 0.0560, 0.0580, 4.8670, 5.4220, -6.5120, -0.0050, 0.0510, 0.0310, 4.9120, 5.2870, -6.5000, -0.0280, 0.0280, 0.0060, 4.8440, 5.1200, -6.4390, -0.0350, 0.0100, 0.0050, 4.7910, 5.0940, -6.3520, -0.0290, 0.0020, 0.0200, 4.7730, 5.1910, -6.3190, -0.0280, 0.0090, 0.0310, 4.8050, 5.2660, -6.3940, -0.0270, 0.0210, 0.0320, 5.0140, 5.2500, -6.5310, -0.0250, 0.0260, 0.0330, 4.9890, 5.3470, -6.6740, 0.0040, 0.0160, 0.0460, 5.0140, 5.4080, -6.6140, 0.0330, 0.0020, 0.0490, 4.9480, 5.3560, -6.6190, 0.0490, -0.0140, 0.0390, 5.0180, 5.3540, -6.5550, 0.0570, -0.0380, 0.0430, 4.9380, 5.2650, -6.4910, 0.0540, -0.0460, 0.0410, 4.8350, 5.2100, -6.3880, 0.0500, -0.0480, 0.0350, 4.9090, 5.2740, -6.2730, 0.0340, -0.0510, 0.0430, 4.8640, 5.3040, -6.2540, -0.0060, -0.0490, 0.0490, 4.8070, 5.2480, -6.1900, -0.0330, -0.0330, 0.0370, 4.7740, 5.1930, -6.1140, -0.0440, -0.0210, 0.0420, 4.7780, 5.2710, -6.1620, -0.0130, 0.0200, 0.0710, 4.9280, 5.3700, -6.2500, 0.0180, 0.0320, 0.0790, 4.9920, 5.5840, -6.4540, 0.0290, 0.0290, 0.0700, 5.0080, 5.6560, -6.5860, 0.0300, 0.0130, 0.0440, 5.0090, 5.4360, -6.5530, 0.0460, -0.0140, 0.0130, 4.9560, 5.2600, -6.3700, 0.0470, -0.0350, -0.0160, 4.8920, 5.1700, -6.2850, 0.0360, -0.0410, -0.0210, 4.9090, 5.1160, -6.2030, 0.0240, -0.0310, -0.0100, 4.8870, 5.0670, -6.2510, 0.0100, -0.0230, 0.0000, 4.9140, 5.1080, -6.1630, -0.0040, -0.0250, 0.0020, 4.8750, 5.1950, -6.1050, -0.0320, -0.0160, 0.0110, 5.0200, 5.4380, -6.3550, -0.0330, -0.0130, 0.0350, 5.0210, 5.5670, -6.3530, -0.0030, -0.0200, 0.0280, 5.1110, 5.4960, -6.4260, 0.0240, -0.0330, 0.0210, 5.0300, 5.5230, -6.4890, 0.0570, -0.0570, 0.0170, 4.9750, 5.5250, -6.4050, 0.0900, -0.0730, 0.0080, 4.9500, 5.4430, -6.2300, 0.0880, -0.0780, -0.0020, 4.9280, 5.2120, -6.3020, 0.0640, -0.0830, -0.0110, 4.8890, 5.1260, -6.2960, 0.0880, -0.0850, -0.0030, 4.8360, 5.2170, -6.1390, 0.1080, -0.0740, 0.0050, 4.7940, 5.2800, -6.0550, 0.0820, -0.0570, 0.0160, 4.8470, 5.2060, -6.1770, 0.0440, -0.0440, 0.0130, 4.8900, 5.2100, -6.3520, 0.0530, -0.0390, 0.0160, 4.9380, 5.4050, -6.3450, 0.0430, -0.0500, 0.0200, 4.9440, 5.3840, -6.3760, 0.0330, -0.0630, 0.0080, 4.9390, 5.2860, -6.3210, 0.0380, -0.0730, -0.0050, 4.8640, 5.1830, -6.2020, 0.0320, -0.0720, -0.0100, 4.7850, 5.1020, -6.1320, 0.0180, -0.0620, -0.0020, 4.7800, 5.0650, -6.1020, 0.0090, -0.0440, 0.0010, 4.7680, 4.8650, -6.2270, 0.0250, -0.0260, 0.0040, 4.8420, 5.0270, -6.0740, 0.0410, -0.0100, 0.0160, 4.8120, 5.2440, -6.0560, 0.0060, 0.0030, 0.0310, 5.0050, 5.0960, -6.2640, -0.0220, 0.0130, 0.0290, 5.0000, 5.3600, -6.3010, -0.0120, 0.0020, 0.0360, 5.0590, 5.5350, -6.4150, -0.0260, -0.0150, 0.0120, 5.0780, 5.2720, -6.5580, -0.0140, -0.0310, -0.0170, 5.1230, 5.2170, -6.4870, 0.0440, -0.0510, -0.0230, 4.9290, 5.2610, -6.2170, 0.0540, -0.0600, -0.0030, 4.9010, 5.2210, -6.1650, 0.0240, -0.0500, -0.0070, 4.8530, 5.1010, -6.2350, -0.0030, -0.0410, -0.0040, 4.9840, 5.1630, -6.2780, -0.0110, -0.0310, 0.0000, 4.9710, 5.3040, -6.2480, -0.0170, -0.0250, 0.0060, 4.9450, 5.3380, -6.2900, -0.0290, -0.0220, 0.0010, 5.0760, 5.2990, -6.4350, -0.0240, -0.0290, -0.0060, 5.1030, 5.3330, -6.6360, -0.0080, -0.0440, -0.0020, 5.0510, 5.3890, -6.6180, 0.0220, -0.0710, -0.0120
// };

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

int main() {
    // This is needed so that output of printf is output immediately without buffering
    setvbuf(stdout, NULL, _IONBF, 0);

#ifdef CONFIG_SOC_NRF5340_CPUAPP
    // Switch CPU core clock to 128 MHz
    nrfx_clock_divider_set(NRF_CLOCK_DOMAIN_HFCLK, NRF_CLOCK_HFCLK_DIV_1);
#endif

    printk("Edge Impulse standalone inferencing (Zephyr)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printk("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    ei_impulse_result_t result = { 0 };

    while (1) {
        if (!features_ready)
        {
            printk("Waiting for features...\n");
            k_msleep(100);
            continue;
        }
        printk("Running inferencing...\n");
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, true);
        printk("run_classifier returned: %d\n", res);

        if (res != 0) return 1;

        printk("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
        bool bb_found = result.bounding_boxes[0].value > 0;
        for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
            auto bb = result.bounding_boxes[ix];
            if (bb.value == 0) {
                continue;
            }
            printk("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
        }
        if (!bb_found) {
            printk("    No objects found\n");
        }
#else
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            printk("    %s: %.5f\n", result.classification[ix].label,
                                    result.classification[ix].value);
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printk("    anomaly score: %.3f\n", result.anomaly);
#endif
#endif
        features_ready = false;
        // k_msleep(2000);
    }
}
