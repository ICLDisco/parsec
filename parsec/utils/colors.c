/*
 * Copyright (c) 2010-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/colors.h"

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

static unsigned int color_seed = 1789;

/**
 * A simple solution to generate different color tables for each rank. For a
 * more detailed and visualy appealing solution take a look at
 * http://phrogz.net/css/distinct-colors.html
 * and http://en.wikipedia.org/wiki/HSV_color_space
 */
static void HSVtoRGB( double *r, double *g, double *b, double h, double s, double v )
{
    int i;
    double c, x, m;

    c = v * s;
    h /= 60.0;
    i = (int)( h );
    x = c * (1 - abs(i % 2 - 1));
    m = v - c;

    switch( i ) {
    case 0:
        *r = c;
        *g = x;
        *b = 0;
        break;
    case 1:
        *r = x;
        *g = c;
        *b = 0;
        break;
    case 2:
        *r = 0;
        *g = c;
        *b = x;
        break;
    case 3:
        *r = 0;
        *g = x;
        *b = c;
        break;
    case 4:
        *r = x;
        *g = 0;
        *b = c;
        break;
    default: // case 5:
        *r = c;
        *g = 0;
        *b = x;
        break;
    }
    *r += m;
    *g += m;
    *b += m;
}

static inline double get_rand_in_range(int m, int M)
{
#if defined(__WINDOWS__)
    rand_s(&color_seed);
    return (double)m + (double)color_seed / ((double)UINT_MAX / (M - m + 1) + 1);
#elif defined(PARSEC_HAVE_RAND_R)
    return (double)m + (double)rand_r(&color_seed) / ((double)RAND_MAX / (M - m + 1) + 1);
#else
#error Missing support for the platform random number generator similar to POSIX rand_r
#endif
}

char *parsec_unique_color(int index, int colorspace)
{
    char color[8];
    double r, g, b;

    double hue = get_rand_in_range(0, 360);  //  0.0 to 360.0
    double saturation = get_rand_in_range(180, 360) / 360.0;  //  0.5 to 1.0, away from white
    double brightness = get_rand_in_range(180, 360) / 360.0;  //  0.5 to 1.0, away from black
    HSVtoRGB(&r, &g, &b, hue, saturation, brightness);
    (void)index; (void)colorspace;
    snprintf(color, 8, "#%02x%02x%02x", (int)(255.0*r), (int)(255.0*g), (int)(255.0*b));
    return strdup(color);
}

