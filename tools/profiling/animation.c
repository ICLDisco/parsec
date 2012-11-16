#include <stdio.h>
#include <gd.h>
#include <assert.h>

static FILE *out = NULL;
static gdImagePtr firstim = NULL;
//static int trans;

/* Input: an image with all the colors needed, or the expected size */
void startAnimation(char *filename, char *pseudo, unsigned int length)
{
    out = fopen(filename, "wb");
    firstim = gdImageCreateFromPngPtr(length, pseudo);
    gdImageTrueColorToPalette(firstim, 1, 256);
    gdImageGifAnimBegin(firstim, out, 1, 1);
}

/* Input: an image to add to the animation */
void addAnimation(char *data, int length, int delay)
{
    gdImagePtr im;
    
    im = gdImageCreateFromPngPtr(length, data);
    assert( im->sx == firstim->sx &&
            im->sy == firstim->sy );
    gdImageTrueColorToPalette(im, 1, 256);
    gdImagePaletteCopy(im, firstim);

    gdImageGifAnimAdd(im, out, 0, 0, 0, delay, gdDisposalNone, NULL);
}

void endAnimation(void)
{
    firstim = NULL;
    gdImageGifAnimEnd(out);
    fclose(out);
    out = NULL;
}
