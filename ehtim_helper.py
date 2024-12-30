import matplotlib.pyplot as plt
import ehtim as eh
from PIL import Image
import numpy as np
from pyproj import Proj, Transformer

geodetic = Proj(proj='latlong', datum='WGS84')
ecef = Proj(proj='latlong', datum='WGS84',
            lat_0=0, lon_0=0, x_0=0, y_0=0, z_0=0)

transformer = Transformer.from_proj(geodetic, ecef)


def latlon_to_ecef(lat, lon, alt=0):
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z


def modify_telescope_positions(eht, new_positions):
    for telescope_name, lat, lon, alt in new_positions:
        x, y, z = latlon_to_ecef(lat, lon, alt)

        idx = np.where(eht.tarr['site'] == telescope_name)[0]
        if len(idx) > 0:
            eht.tarr[idx[0]]['x'] = x
            eht.tarr[idx[0]]['y'] = y
            eht.tarr[idx[0]]['z'] = z
        else:
            print(f"Warning: Telescope '{
                  telescope_name}' not found in the array.")
    return eht


def load_image(image_path):
    if image_path.lower().endswith('.fits'):
        im = eh.image.load_image(image_path)
    elif image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        normalized_img = img_array / 255.0
        total_flux = 1.0
        eht_fov = 200 * eh.RADPERUAS
        eht_size = img_array.shape[0]

        im_array = normalized_img * total_flux / normalized_img.sum()

        im = eh.image.Image(
            im_array,
            psize=eht_fov / eht_size,
            ra=0.0,
            dec=0.0,
            rf=230e9,
            source='SyntheticImage'
        )
    else:
        raise ValueError(f"Unsupported file format: {image_path}")

    return im


def generate_image(im, eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, npix=100,
                   ttype='fast'):

    obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=True, ttype=ttype)

    fov = 200 * eh.RADPERUAS

    dim = obs.dirtyimage(npix, fov)
    dbeam = obs.dirtybeam(npix, fov)
    cbeam = obs.cleanbeam(npix, fov)

    dim_array = dim.imarr()
    dbeam_array = dbeam.imarr()
    cbeam_array = cbeam.imarr()

    zbl = im.total_flux()

    prior_fwhm = 100 * eh.RADPERUAS
    emptyprior = eh.image.make_square(obs, npix, fov)
    gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

    data_term = {'vis': 1}
    reg_term = {'tv2': 1, 'l1': 0.1}

    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, flux=zbl,
                            data_term=data_term, reg_term=reg_term,
                            norm_reg=True,
                            epsilon_tv=1.e-10,
                            maxit=250, ttype=ttype)
    imgr.make_image_I(show_updates=False)

    out = imgr.out_last()
    out = out.imarr()

    return out, dim_array, dbeam_array, cbeam_array


if __name__ == "__main__":
    image_path = 'fits/Mean GRMHD.fits'
    eht = eh.array.load_txt('arrays/EHT2017.txt')

    new_positions = []
    eht = modify_telescope_positions(eht, new_positions)

    tint_sec = 60
    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 4.e9

    im = load_image(image_path)

    out, dim, dbeam, cbeam = generate_image(im, eht, tint_sec, tadv_sec,
                                            tstart_hr, tstop_hr, bw_hz)
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)

    axes[0, 0].imshow(dbeam, cmap='hot')
    axes[0, 0].title.set_text('Dirty Beam')

    axes[0, 1].imshow(dim, cmap='hot')
    axes[0, 1].title.set_text('Dirty Image')

    axes[1, 0].imshow(cbeam, cmap='hot')
    axes[1, 0].title.set_text('Clean Beam')

    axes[1, 1].imshow(out, cmap='hot')
    axes[1, 1].title.set_text('Output')

    plt.show()
