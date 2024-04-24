/*******************************************************************************
 * Size: 14 px
 * Bpp: 1
 * Opts: --bpp 1 --size 14 --no-compress --font JetBrainsMonoNerdFontMono-Bold.ttf --symbols `1234567890-=qwertyuiop[]\asdfghjkl;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?\ --range 8600,62100 --format lvgl -o jb_mono_bold.c
 ******************************************************************************/

// #ifdef LV_LVGL_H_INCLUDE_SIMPLE
#include "lvgl.h"
// #else
// #include "lvgl/lvgl.h"
// #endif

#ifndef JB_MONO_BOLD
#define JB_MONO_BOLD 1
#endif

#if JB_MONO_BOLD

/*-----------------
 *    BITMAPS
 *----------------*/

/*Store the image of the glyphs*/
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap[] = {
    /* U+0021 "!" */
    0xff, 0xfc, 0x30,

    /* U+0022 "\"" */
    0xde, 0xf7, 0xb0,

    /* U+0023 "#" */
    0x26, 0x4f, 0xfb, 0x24, 0x49, 0xbf, 0xa4, 0xc9,
    0x90,

    /* U+0024 "$" */
    0x10, 0x21, 0xf7, 0xfd, 0x7a, 0x1e, 0x1e, 0x17,
    0xaf, 0xfb, 0xe1, 0x2, 0x0,

    /* U+0025 "%" */
    0xf1, 0xd2, 0xd6, 0xf4, 0x8, 0x1e, 0x39, 0x29,
    0x69, 0xc6,

    /* U+0026 "&" */
    0x3c, 0x32, 0x18, 0x6, 0x7, 0x7, 0xdb, 0x39,
    0x8c, 0xce, 0x3d, 0x80,

    /* U+0027 "'" */
    0xff,

    /* U+0028 "(" */
    0x19, 0xd9, 0x8c, 0x63, 0x18, 0xc6, 0x38, 0xe3,
    0x84,

    /* U+0029 ")" */
    0xc7, 0xc, 0x31, 0x8c, 0x63, 0x18, 0xce, 0xee,
    0x40,

    /* U+002A "*" */
    0x18, 0x18, 0xdb, 0x7e, 0x3c, 0x3c, 0x66,

    /* U+002B "+" */
    0x30, 0x60, 0xc7, 0xf3, 0x6, 0xc, 0x0,

    /* U+002C "," */
    0xfe,

    /* U+002D "-" */
    0xf0,

    /* U+002E "." */
    0xc0,

    /* U+002F "/" */
    0x6, 0x18, 0x30, 0x41, 0x83, 0x4, 0x18, 0x30,
    0xc1, 0x83, 0xc, 0x18, 0x0,

    /* U+0030 "0" */
    0x7d, 0x8f, 0x1e, 0x3f, 0x78, 0xf1, 0xe3, 0xc6,
    0xf8,

    /* U+0031 "1" */
    0x39, 0xf3, 0x60, 0xc1, 0x83, 0x6, 0xc, 0x19,
    0xfc,

    /* U+0032 "2" */
    0x7b, 0x3c, 0xc3, 0x1c, 0x63, 0x9c, 0xe3, 0xf0,

    /* U+0033 "3" */
    0xfc, 0x18, 0x61, 0x83, 0xc0, 0xc1, 0xe3, 0xc6,
    0xf8,

    /* U+0034 "4" */
    0x18, 0x63, 0x1c, 0x6f, 0x3c, 0xff, 0xc, 0x30,

    /* U+0035 "5" */
    0xff, 0x83, 0x6, 0xf, 0xc0, 0xc1, 0xe3, 0xc6,
    0xf8,

    /* U+0036 "6" */
    0x18, 0x60, 0xc3, 0x7, 0xd8, 0xf1, 0xe3, 0xc6,
    0xf8,

    /* U+0037 "7" */
    0xff, 0x8f, 0x10, 0x60, 0xc3, 0x6, 0x18, 0x30,
    0x60,

    /* U+0038 "8" */
    0x7d, 0x8f, 0x1e, 0x33, 0x9d, 0xf1, 0xe3, 0xc6,
    0xf8,

    /* U+0039 "9" */
    0x7d, 0x8f, 0x1e, 0x3c, 0x6f, 0x83, 0xc, 0x18,
    0x60,

    /* U+003A ":" */
    0xc0, 0x3,

    /* U+003B ";" */
    0xc0, 0x3, 0xe8,

    /* U+003C "<" */
    0x2, 0x1c, 0xe7, 0xc, 0xe, 0x7, 0x3,

    /* U+003D "=" */
    0xfc, 0x0, 0x0, 0xfc,

    /* U+003E ">" */
    0x1, 0x81, 0xe0, 0xf0, 0x67, 0xbc, 0x60,

    /* U+003F "?" */
    0xf8, 0x30, 0xc3, 0x79, 0x80, 0x0, 0x1, 0x80,

    /* U+0040 "@" */
    0x3c, 0x62, 0xc3, 0xdf, 0xf3, 0xf3, 0xf3, 0xf3,
    0xf3, 0xde, 0xc0, 0x60, 0x3c,

    /* U+0041 "A" */
    0x18, 0x1c, 0x3c, 0x34, 0x34, 0x26, 0x7e, 0x62,
    0x63, 0x43,

    /* U+0042 "B" */
    0xfb, 0x3c, 0xf3, 0xf3, 0x3c, 0xf3, 0xcf, 0xe0,

    /* U+0043 "C" */
    0x7d, 0x8f, 0x1e, 0xc, 0x18, 0x30, 0x63, 0xc6,
    0xf8,

    /* U+0044 "D" */
    0xf9, 0x8f, 0x1e, 0x3c, 0x78, 0xf1, 0xe3, 0xc7,
    0xf0,

    /* U+0045 "E" */
    0xff, 0xc, 0x30, 0xff, 0xc, 0x30, 0xc3, 0xf0,

    /* U+0046 "F" */
    0xff, 0x83, 0x6, 0xf, 0xd8, 0x30, 0x60, 0xc1,
    0x80,

    /* U+0047 "G" */
    0x7d, 0x8f, 0x6, 0xd, 0xf8, 0xf1, 0xe3, 0xce,
    0xf8,

    /* U+0048 "H" */
    0xcf, 0x3c, 0xf3, 0xff, 0x3c, 0xf3, 0xcf, 0x30,

    /* U+0049 "I" */
    0xfc, 0xc3, 0xc, 0x30, 0xc3, 0xc, 0x33, 0xf0,

    /* U+004A "J" */
    0x6, 0xc, 0x18, 0x30, 0x60, 0xc1, 0xe3, 0xc6,
    0xf8,

    /* U+004B "K" */
    0xc7, 0x9b, 0x36, 0xcf, 0x9b, 0x33, 0x66, 0xc7,
    0x8c,

    /* U+004C "L" */
    0xc3, 0xc, 0x30, 0xc3, 0xc, 0x30, 0xc3, 0xf0,

    /* U+004D "M" */
    0xef, 0xdf, 0xbf, 0x7d, 0x7a, 0xf1, 0xe3, 0xc7,
    0x8c,

    /* U+004E "N" */
    0xe7, 0xcf, 0x9f, 0xbd, 0x7a, 0xf7, 0xe7, 0xcf,
    0x9c,

    /* U+004F "O" */
    0x7b, 0x3c, 0xf3, 0xcf, 0x3c, 0xf3, 0xcd, 0xe0,

    /* U+0050 "P" */
    0xfd, 0x8f, 0x1e, 0x3c, 0x7f, 0xb0, 0x60, 0xc1,
    0x80,

    /* U+0051 "Q" */
    0x7d, 0x8f, 0x1e, 0x3c, 0x78, 0xf1, 0xe3, 0xc6,
    0xf8, 0x30, 0x60, 0x60,

    /* U+0052 "R" */
    0xfd, 0x8f, 0x1e, 0x3c, 0x7f, 0xb6, 0x66, 0xcd,
    0x8c,

    /* U+0053 "S" */
    0x7d, 0x9b, 0x7, 0x7, 0xc7, 0xc1, 0xe3, 0xc6,
    0xf8,

    /* U+0054 "T" */
    0xfe, 0x30, 0x60, 0xc1, 0x83, 0x6, 0xc, 0x18,
    0x30,

    /* U+0055 "U" */
    0xcf, 0x3c, 0xf3, 0xcf, 0x3c, 0xf3, 0xcd, 0xe0,

    /* U+0056 "V" */
    0x43, 0x63, 0x62, 0x66, 0x26, 0x36, 0x34, 0x3c,
    0x1c, 0x18,

    /* U+0057 "W" */
    0x9b, 0x9a, 0xda, 0xda, 0xea, 0xea, 0x6a, 0x6e,
    0x6e, 0x66,

    /* U+0058 "X" */
    0x63, 0x66, 0x34, 0x3c, 0x18, 0x1c, 0x3c, 0x36,
    0x66, 0x63,

    /* U+0059 "Y" */
    0xc3, 0x66, 0x66, 0x34, 0x3c, 0x18, 0x18, 0x18,
    0x18, 0x18,

    /* U+005A "Z" */
    0xfc, 0x31, 0x86, 0x30, 0xc6, 0x18, 0xc3, 0xf0,

    /* U+005B "[" */
    0xfc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xf0,

    /* U+005C "\\" */
    0xc1, 0x81, 0x3, 0x6, 0x4, 0xc, 0x18, 0x18,
    0x30, 0x60, 0x60, 0xc0, 0x80,

    /* U+005D "]" */
    0xf3, 0x33, 0x33, 0x33, 0x33, 0x33, 0xf0,

    /* U+005E "^" */
    0x30, 0xc7, 0x9a, 0x4f, 0x10,

    /* U+005F "_" */
    0xfe,

    /* U+0060 "`" */
    0x60,

    /* U+0061 "a" */
    0x7c, 0xcc, 0x1b, 0xfc, 0x78, 0xf1, 0xbf,

    /* U+0062 "b" */
    0xc3, 0xf, 0xb3, 0xcf, 0x3c, 0xf3, 0xcf, 0xe0,

    /* U+0063 "c" */
    0x79, 0x8b, 0x16, 0xc, 0x18, 0xf3, 0x3c,

    /* U+0064 "d" */
    0xc, 0x37, 0xf3, 0xcf, 0x3c, 0xf3, 0xcd, 0xf0,

    /* U+0065 "e" */
    0x7d, 0x8f, 0x1f, 0xfc, 0x18, 0x31, 0xbe,

    /* U+0066 "f" */
    0x1e, 0x60, 0xc1, 0x8f, 0xe6, 0xc, 0x18, 0x30,
    0x60,

    /* U+0067 "g" */
    0x7f, 0x3c, 0xf3, 0xcf, 0x3c, 0xdf, 0xc, 0x37,
    0x80,

    /* U+0068 "h" */
    0xc3, 0xf, 0xb3, 0xcf, 0x3c, 0xf3, 0xcf, 0x30,

    /* U+0069 "i" */
    0x30, 0x0, 0x7, 0x83, 0x6, 0xc, 0x18, 0x30,
    0x63, 0xf8,

    /* U+006A "j" */
    0x18, 0x3e, 0x31, 0x8c, 0x63, 0x18, 0xc6, 0x3f,
    0x0,

    /* U+006B "k" */
    0xc1, 0x83, 0x1e, 0x6c, 0x9f, 0x36, 0x66, 0xcd,
    0x8c,

    /* U+006C "l" */
    0xf8, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0xf,

    /* U+006D "m" */
    0xff, 0xbf, 0x7e, 0xfd, 0xfb, 0xf7, 0xef,

    /* U+006E "n" */
    0xfb, 0x3c, 0xf3, 0xcf, 0x3c, 0xf3,

    /* U+006F "o" */
    0x7d, 0x8f, 0x1e, 0x3c, 0x78, 0xf1, 0xbe,

    /* U+0070 "p" */
    0xfb, 0x3c, 0xf3, 0xcf, 0x3c, 0xfe, 0xc3, 0xc,
    0x0,

    /* U+0071 "q" */
    0x7f, 0x3c, 0xf3, 0xcf, 0x3c, 0xdf, 0xc, 0x30,
    0xc0,

    /* U+0072 "r" */
    0xfb, 0x3c, 0xf0, 0xc3, 0xc, 0x30,

    /* U+0073 "s" */
    0x7d, 0x8f, 0x7, 0xe7, 0xe0, 0xf1, 0xbe,

    /* U+0074 "t" */
    0x30, 0x63, 0xf9, 0x83, 0x6, 0xc, 0x18, 0x30,
    0x3c,

    /* U+0075 "u" */
    0xcf, 0x3c, 0xf3, 0xcf, 0x3c, 0xde,

    /* U+0076 "v" */
    0x63, 0x62, 0x66, 0x26, 0x34, 0x3c, 0x1c, 0x18,

    /* U+0077 "w" */
    0xd9, 0xd9, 0x5b, 0x5b, 0x56, 0x56, 0x66, 0x66,

    /* U+0078 "x" */
    0x66, 0x36, 0x3c, 0x18, 0x1c, 0x3c, 0x66, 0x63,

    /* U+0079 "y" */
    0x63, 0x62, 0x66, 0x36, 0x34, 0x1c, 0x1c, 0x18,
    0x18, 0x18, 0x30,

    /* U+007A "z" */
    0xfc, 0x31, 0x8c, 0x31, 0x8c, 0x3f,

    /* U+007B "{" */
    0x1c, 0x61, 0x86, 0x18, 0x6e, 0x4, 0x18, 0x41,
    0x4, 0x1c,

    /* U+007C "|" */
    0xff, 0xff, 0xff, 0xc0,

    /* U+007D "}" */
    0xe0, 0x82, 0x8, 0x60, 0x81, 0xc8, 0x60, 0x82,
    0x8, 0xe0,

    /* U+007E "~" */
    0xe7, 0xaf, 0x38,

    /* U+2198 "↘" */
    0x0, 0x62, 0x73, 0x3b, 0x1f, 0xf, 0x7f,

    /* U+F294 "" */
    0x0, 0x8, 0xc, 0xe, 0x4b, 0x6b, 0x3e, 0x1c,
    0x1c, 0x3e, 0x6b, 0x4b, 0xe, 0xc, 0x8, 0x0
};


/*---------------------
 *  GLYPH DESCRIPTION
 *--------------------*/

static const lv_font_fmt_txt_glyph_dsc_t glyph_dsc[] = {
    {.bitmap_index = 0, .adv_w = 0, .box_w = 0, .box_h = 0, .ofs_x = 0, .ofs_y = 0} /* id = 0 reserved */,
    {.bitmap_index = 0, .adv_w = 134, .box_w = 2, .box_h = 10, .ofs_x = 3, .ofs_y = 0},
    {.bitmap_index = 3, .adv_w = 134, .box_w = 5, .box_h = 4, .ofs_x = 2, .ofs_y = 6},
    {.bitmap_index = 6, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 15, .adv_w = 134, .box_w = 7, .box_h = 14, .ofs_x = 1, .ofs_y = -2},
    {.bitmap_index = 28, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 38, .adv_w = 134, .box_w = 9, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 50, .adv_w = 134, .box_w = 2, .box_h = 4, .ofs_x = 3, .ofs_y = 6},
    {.bitmap_index = 51, .adv_w = 134, .box_w = 5, .box_h = 14, .ofs_x = 2, .ofs_y = -2},
    {.bitmap_index = 60, .adv_w = 134, .box_w = 5, .box_h = 14, .ofs_x = 1, .ofs_y = -2},
    {.bitmap_index = 69, .adv_w = 134, .box_w = 8, .box_h = 7, .ofs_x = 0, .ofs_y = 2},
    {.bitmap_index = 76, .adv_w = 134, .box_w = 7, .box_h = 7, .ofs_x = 1, .ofs_y = 1},
    {.bitmap_index = 83, .adv_w = 134, .box_w = 2, .box_h = 4, .ofs_x = 3, .ofs_y = -3},
    {.bitmap_index = 84, .adv_w = 134, .box_w = 4, .box_h = 1, .ofs_x = 2, .ofs_y = 4},
    {.bitmap_index = 85, .adv_w = 134, .box_w = 2, .box_h = 1, .ofs_x = 3, .ofs_y = 0},
    {.bitmap_index = 86, .adv_w = 134, .box_w = 7, .box_h = 14, .ofs_x = 1, .ofs_y = -2},
    {.bitmap_index = 99, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 108, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 117, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 125, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 134, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 142, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 151, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 160, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 169, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 178, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 187, .adv_w = 134, .box_w = 2, .box_h = 8, .ofs_x = 3, .ofs_y = 0},
    {.bitmap_index = 189, .adv_w = 134, .box_w = 2, .box_h = 11, .ofs_x = 3, .ofs_y = -3},
    {.bitmap_index = 192, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 1},
    {.bitmap_index = 199, .adv_w = 134, .box_w = 6, .box_h = 5, .ofs_x = 1, .ofs_y = 2},
    {.bitmap_index = 203, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 1},
    {.bitmap_index = 210, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 218, .adv_w = 134, .box_w = 8, .box_h = 13, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 231, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 241, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 249, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 258, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 267, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 275, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 284, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 293, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 301, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 309, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 318, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 327, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 2, .ofs_y = 0},
    {.bitmap_index = 335, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 344, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 353, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 361, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 370, .adv_w = 134, .box_w = 7, .box_h = 13, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 382, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 391, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 400, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 409, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 417, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 427, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 437, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 447, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 457, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 465, .adv_w = 134, .box_w = 4, .box_h = 13, .ofs_x = 3, .ofs_y = -1},
    {.bitmap_index = 472, .adv_w = 134, .box_w = 7, .box_h = 14, .ofs_x = 1, .ofs_y = -2},
    {.bitmap_index = 485, .adv_w = 134, .box_w = 4, .box_h = 13, .ofs_x = 2, .ofs_y = -1},
    {.bitmap_index = 492, .adv_w = 134, .box_w = 6, .box_h = 6, .ofs_x = 1, .ofs_y = 5},
    {.bitmap_index = 497, .adv_w = 134, .box_w = 7, .box_h = 1, .ofs_x = 1, .ofs_y = -2},
    {.bitmap_index = 498, .adv_w = 134, .box_w = 3, .box_h = 1, .ofs_x = 2, .ofs_y = 10},
    {.bitmap_index = 499, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 506, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 514, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 521, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 529, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 536, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 545, .adv_w = 134, .box_w = 6, .box_h = 11, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 554, .adv_w = 134, .box_w = 6, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 562, .adv_w = 134, .box_w = 7, .box_h = 11, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 572, .adv_w = 134, .box_w = 5, .box_h = 13, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 581, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 590, .adv_w = 134, .box_w = 8, .box_h = 10, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 600, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 607, .adv_w = 134, .box_w = 6, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 613, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 620, .adv_w = 134, .box_w = 6, .box_h = 11, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 629, .adv_w = 134, .box_w = 6, .box_h = 11, .ofs_x = 1, .ofs_y = -3},
    {.bitmap_index = 638, .adv_w = 134, .box_w = 6, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 644, .adv_w = 134, .box_w = 7, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 651, .adv_w = 134, .box_w = 7, .box_h = 10, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 660, .adv_w = 134, .box_w = 6, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 666, .adv_w = 134, .box_w = 8, .box_h = 8, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 674, .adv_w = 134, .box_w = 8, .box_h = 8, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 682, .adv_w = 134, .box_w = 8, .box_h = 8, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 690, .adv_w = 134, .box_w = 8, .box_h = 11, .ofs_x = 0, .ofs_y = -3},
    {.bitmap_index = 701, .adv_w = 134, .box_w = 6, .box_h = 8, .ofs_x = 1, .ofs_y = 0},
    {.bitmap_index = 707, .adv_w = 134, .box_w = 6, .box_h = 13, .ofs_x = 1, .ofs_y = -1},
    {.bitmap_index = 717, .adv_w = 134, .box_w = 2, .box_h = 13, .ofs_x = 3, .ofs_y = -2},
    {.bitmap_index = 721, .adv_w = 134, .box_w = 6, .box_h = 13, .ofs_x = 1, .ofs_y = -1},
    {.bitmap_index = 731, .adv_w = 134, .box_w = 7, .box_h = 3, .ofs_x = 1, .ofs_y = 4},
    {.bitmap_index = 734, .adv_w = 134, .box_w = 8, .box_h = 7, .ofs_x = 0, .ofs_y = 0},
    {.bitmap_index = 741, .adv_w = 134, .box_w = 8, .box_h = 16, .ofs_x = 0, .ofs_y = -3}
};

/*---------------------
 *  CHARACTER MAPPING
 *--------------------*/

static const uint16_t unicode_list_1[] = {
    0x0, 0xd0fc
};

/*Collect the unicode lists and glyph_id offsets*/
static const lv_font_fmt_txt_cmap_t cmaps[] =
{
    {
        .range_start = 33, .range_length = 94, .glyph_id_start = 1,
        .unicode_list = NULL, .glyph_id_ofs_list = NULL, .list_length = 0, .type = LV_FONT_FMT_TXT_CMAP_FORMAT0_TINY
    },
    {
        .range_start = 8600, .range_length = 53501, .glyph_id_start = 95,
        .unicode_list = unicode_list_1, .glyph_id_ofs_list = NULL, .list_length = 2, .type = LV_FONT_FMT_TXT_CMAP_SPARSE_TINY
    }
};



/*--------------------
 *  ALL CUSTOM DATA
 *--------------------*/

#if LVGL_VERSION_MAJOR == 8
/*Store all the custom data of the font*/
static  lv_font_fmt_txt_glyph_cache_t cache;
#endif

#if LVGL_VERSION_MAJOR >= 8
static const lv_font_fmt_txt_dsc_t font_dsc = {
#else
static lv_font_fmt_txt_dsc_t font_dsc = {
#endif
    .glyph_bitmap = glyph_bitmap,
    .glyph_dsc = glyph_dsc,
    .cmaps = cmaps,
    .kern_dsc = NULL,
    .kern_scale = 0,
    .cmap_num = 2,
    .bpp = 1,
    .kern_classes = 0,
    .bitmap_format = 0,
#if LVGL_VERSION_MAJOR == 8
    .cache = &cache
#endif
};

extern const lv_font_t lv_font_montserrat_14;


/*-----------------
 *  PUBLIC FONT
 *----------------*/

/*Initialize a public general font descriptor*/
#if LVGL_VERSION_MAJOR >= 8
const lv_font_t jb_mono_bold = {
#else
lv_font_t jb_mono_bold = {
#endif
    .get_glyph_dsc = lv_font_get_glyph_dsc_fmt_txt,    /*Function pointer to get glyph's data*/
    .get_glyph_bitmap = lv_font_get_bitmap_fmt_txt,    /*Function pointer to get glyph's bitmap*/
    .line_height = 16,          /*The maximum line height required by the font*/
    .base_line = 3,             /*Baseline measured from the bottom of the line*/
#if !(LVGL_VERSION_MAJOR == 6 && LVGL_VERSION_MINOR == 0)
    .subpx = LV_FONT_SUBPX_NONE,
#endif
#if LV_VERSION_CHECK(7, 4, 0) || LVGL_VERSION_MAJOR >= 8
    .underline_position = -2,
    .underline_thickness = 1,
#endif
    .dsc = &font_dsc,          /*The custom font data. Will be accessed by `get_glyph_bitmap/dsc` */
#if LV_VERSION_CHECK(8, 2, 0) || LVGL_VERSION_MAJOR >= 9
    .fallback = &lv_font_montserrat_14,
#endif
    .user_data = NULL,
};



#endif /*#if JB_MONO_BOLD*/

