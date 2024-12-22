import matplotlib.pyplot as plt
import ehtim as eh


def generate_image(im, eht, tint_sec, tadv_sec, tstart_hr, stop_hr, bw_hz, ttype='direct'):

    obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=True, ttype=ttype)

    npix = 100
    fov = 200*eh.RADPERUAS

    dim = obs.dirtyimage(npix, fov)

    dbeam = obs.dirtybeam(npix, fov)

    cbeam = obs.cleanbeam(npix, fov)

    zbl = im.total_flux()

    prior_fwhm = 100*eh.RADPERUAS
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
    return out, dim, dbeam, cbeam


if __name__ == "__main__":
    im = eh.image.load_image('fits/Mean GRMHD.fits')
    eht = eh.array.load_txt('arrays/EHT2017.txt')

    tint_sec = 60
    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 4.e9

    out, dim, dbeam, cbeam = generate_image(im, eht, tint_sec, tadv_sec,
                                            tstart_hr, tstop_hr, bw_hz)
    dim.display()
    out.display()
    plt.show()
