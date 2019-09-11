"""=============================================================================
Utility script to convert multi-page tiffs (.svs files) to .pngs.
============================================================================="""

from   openslide import OpenSlide
import os
import numpy as np
import random
import sys

# ------------------------------------------------------------------------------

FINAL_SIZE = 1000

# ------------------------------------------------------------------------------

def main():
    tissue = sys.argv[1]
    print('=' * 80)
    print('Processing %s' % tissue)
    process_tissue(tissue)

# ------------------------------------------------------------------------------

def process_tissue(tissue):
    directory = '/tigress/gwg3/Tissues/%s' % tissue
    with open('/tigress/gwg3/new_GTEX/%s/out.txt' % tissue, 'w+') as f:
        n = len(os.listdir(directory))
        for i, fname in enumerate(os.listdir(directory)):

            fname = fname.strip()  # Some files have a trailing carriage return.
            if not fname.endswith('.svs'):
                raise AttributeError('Misunderstood file: %s' % fname)

            print('-' * 80)
            print('%s / %s' % (i+1, n))
            print(fname)

            fpath  = os.path.join(directory, fname)
            slide  = OpenSlide(fpath)
            strict = True
            img    = get_good_crop(slide, strict)

            if not img:
                # Relax the constraint that the image should be surrounded by
                # non-white crops and search again.
                strict = False
                img    = get_good_crop(slide, strict)
            print('Strict: %s' % strict)

            if img:
                print('SUCCEEDED.')
                f.write('%s\tSucceeded, strict=%s\n' % (fname, strict))
                new_fname = fname.replace('.svs', '.png')
                new_fpath = '/tigress/gwg3/new_GTEX/%s/%s' % (tissue, new_fname)
                img.save(new_fpath)
            else:
                print('FAILED.')
                f.write('%s\tFailed: could not crop\n' % fname)
                new_fname = fname.replace('.svs', '.png')
                new_fname = 'failed_' + new_fname
                new_fpath = '/tigress/gwg3/new_GTEX/%s/%s' % (tissue, new_fname)
                save_as_png(slide, new_fpath)

# ------------------------------------------------------------------------------

def get_good_crop(slide, strict):
    """
    Images were split into 1000x1000 pixel tiles. A tile was considered for
    selection if the mean gray values of itself and the tiles above, below,
    left, and right of it were each below (darker than) 180 out of 255. Because
    of the intractable file size of the full-resolution images, tile selection
    was actually performed on the 16x lower resolution version of the image, and
    the region in the full-resolution image corresponding to the selected tile
    was extracted.
    """
    level     = slide.level_count - 1
    ds_factor = slide.level_downsamples[level]
    if strict:
        print('Scaled by: %s' % ds_factor)
    crop_dim  = int(FINAL_SIZE / ds_factor)
    ds_dims   = slide.level_dimensions[level]
    ds_slide  = slide.read_region(location=(0, 0), level=level, size=ds_dims)
    width, height = ds_dims
    good_crops    = []

    # Search increments of the low-resolution crop.
    if strict:
        inc = int(FINAL_SIZE / ds_factor)
    else:
        inc = int(FINAL_SIZE / ds_factor / 2)

    for x in range(0, width-crop_dim, inc):
        for y in range(0, height-crop_dim, inc):
            if is_good_crop(ds_slide, x, y, crop_dim, strict):
                good_crops.append((x, y))

    n_crops = len(good_crops)
    if n_crops > 0 or not strict:
        print('# dark crops: %s' % n_crops)

    if n_crops > 0:
        choice = choose_crop(good_crops, ds_slide, crop_dim)
        if not choice:
            return None
        else:
            x, y = choice

        # We found (x, y) coordinates on images that are smaller by 
        # `ds_factor`.
        x = int(x * ds_factor)
        y = int(y * ds_factor)
        HIGHEST_RES_LEVEL = 0
        # According to documentation, the `location` argument requires a "tuple
        # giving the top left pixel in the level 0 reference frame." The size,
        # then, is measured from the top left corner.
        img = slide.read_region(location=(x, y), level=HIGHEST_RES_LEVEL,
                                size=(FINAL_SIZE, FINAL_SIZE))
        return img
    return None

# ------------------------------------------------------------------------------

def choose_crop(good_crops, ds_slide, crop_dim):
    n_tries = 100
    n_crops = len(good_crops)
    while n_tries > 0:
        choice = random.randint(0, n_crops-1)
        x, y = good_crops[choice]
        crop = ds_slide.crop((x, y, x + crop_dim, y + crop_dim))
        # This sanity checks that the chosen crop is not just a gray image, such
        # as the edge of the slide.
        pct = _pct_gray(crop)
        if pct > 0.5:
            #print('Crop too gray: %s' % pct)
            n_tries -= 1
        else:
            return (x, y)
    return None

# ------------------------------------------------------------------------------

def is_good_crop(image, x, y, crop_dim, strict):
    if not _is_good_crop(image, x, y, crop_dim):
        return False
    # Search on strict, first, and only skip this if we can't find a strictly
    # good sample.
    if strict:
        if not _is_good_crop(image, x, y - crop_dim, crop_dim):
            return False
        if not _is_good_crop(image, x, y + crop_dim, crop_dim):
            return False
        if not _is_good_crop(image, x - crop_dim, y, crop_dim):
            return False
        if not _is_good_crop(image, x + crop_dim, y, crop_dim):
            return False
    return True

# ------------------------------------------------------------------------------

def _is_good_crop(image, x, y, crop_dim):
    THRESHOLD = 180

    # Mean Gray Value - Average gray value within the selection. This is the
    # sum of the gray values of all the pixels in the selection divided by the 
    # number of pixels.
    #
    # For RGB images, the mean is calulated by converting each pixel to
    # grayscale using the formula:
    # 
    #     gray = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    #
    crop  = image.crop((x, y, x + crop_dim, y + crop_dim))
    crop  = np.asarray(crop)

    red   = crop[..., 0]
    green = crop[..., 1]
    blue  = crop[..., 2]

    gray_pixels = (red * 0.3) + (green * 0.55) + (blue * 0.15)
    
    if gray_pixels.mean() < THRESHOLD:
        return True
    return False

# ------------------------------------------------------------------------------

def _pct_gray(image):
    image       = np.asarray(image.copy())
    w, h, _     = image.shape
    pixels      = 1000
    gray_pixels = 0
    TOLERANCE   = 10.0
    for _ in range(pixels):
        ri = np.random.randint(0, w)
        rj = np.random.randint(0, h)
        rgb = image[ri, rj, :][:3]  # A column of RGB values (one pixel).
        med = np.ones(3) * np.median(rgb)
        if (np.abs(rgb - med) < TOLERANCE).all():
            gray_pixels += 1
    pct = gray_pixels / float(pixels)
    return pct

# ------------------------------------------------------------------------------

def save_as_png(slide, fname):
    level    = slide.level_count - 1
    ds_dims  = slide.level_dimensions[level]
    img      = slide.read_region(location=(0, 0), level=level,
                                 size=ds_dims)
    img.save(fname)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
