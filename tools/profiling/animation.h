#ifndef animation_h
#define animation_h

void startAnimation(char *filename, char *data, unsigned int length);
void addAnimation(char *data, unsigned int length, int delay);
void endAnimation(void);

#endif
