#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import shutil
import warnings

import numpy as np
import pylab as plt
import pandas as pd
import fitsio

import astropy.io.fits
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"

# Max rows returned by Gaia
LIMIT = 10000
# Buffered radius around DECam CCD
SEARCH_RADIUS = 1.5 * 4096//2 * 0.2634 / 3600. # deg
# Catalog match radius
MATCH_RADIUS = 1.0 # arcsec

QUERY = """
select top {limit:d}
source_id, ra, dec, 
phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
astrometric_excess_noise
from gaiadr3.gaia_source
WHERE 1 = CONTAINS(
   POINT({ra}, {dec}),
   CIRCLE(ra, dec, {radius}))
and in_qso_candidates = 'False'
and in_galaxy_candidates = 'False'
and non_single_star = 0
order by source_id
"""

def calc_nmad(values):
    """ Calculate the median absolute deviation (MAD) and distance from
    the median in units of the MAD for each value.
    
    Parameters
    ----------
    values : array of values
    
    Returns
    -------
    median, mad, nmad : median, MAD, and NMAD
    """
    median = np.median(values)
    mad = np.median(np.fabs(median - values))
    nmad = (values-median)/mad
    return median, mad, nmad

def read_catalog_header(hdulist):
    """ Read image header from SourceExtractor catalog

    Parameters
    ----------
    hdulist : HDU list
    
    Returns
    -------
    header : FITS header
    """
    header_data = hdulist['LDAC_IMHEAD'].data 
    header_string = ''.join(header_data[0][0].tolist())
    header = astropy.io.fits.Header.fromstring(header_string)  
    return header

def get_centroid_from_header(hdulist):
    """ Grab the image centroid from the image header 

    Parameters
    ----------
    hdulist : HDU list
    
    Returns
    -------
    ra,dec : centroid (deg)
    """
    header = read_catalog_header(hdulist)
    ra = header['RA_CENT']
    dec = header['DEC_CENT']
    return ra,dec

def get_centroid_from_catalog(ra,dec):
    """ Grab the image centroid from the median of the catalog.

    Parameters
    ----------
    ra  : Right ascension of catalog sources (deg)
    dec : Declination of catalog sources (deg)
    
    Returns
    -------
    ra,dec : centroid (deg)
    """
    # Stupidly simple. Will fail when crossing RA = 0.
    ra = np.median(ra)
    dec = np.median(dec)
    return ra,dec

def get_gaia_catalog(ra,dec,radius):
    """ Grab a Gaia DR3 catalog using TAP for a conical region centered on
    ra,dec with an opening angle of radius.

    Parameters
    ----------
    ra     : Right ascension (deg)
    dec    : Declination (deg)
    radius : Cone search radius (deg)

    Returns
    -------
    tab    : astropy table

    """
    query = QUERY.format(ra=ra,dec=dec,radius=radius,limit=LIMIT)
    job = Gaia.launch_job(query)
    tab = job.get_results()
    return tab

def flag_gaia_stars(gaia):
    """ Flag likely Gaia stars 

    Parameters
    ----------
    gaia : Gaia catalog
    
    Returns
    -------
    stars : Boolean array for whether or not an object is a star.
    """
    # Gaia star/galaxy selection From Ting:
    # https://delve-survey.slack.com/archives/CFC30CXDW/p1670277441985609?thread_ts=1670277065.740789&cid=CFC30CXDW
    stars = (np.log10(gaia['astrometric_excess_noise']) < np.maximum((gaia['phot_g_mean_mag']-18.2)*.3+.2,.3))
    return stars

def calc_flux_radius_nmad(flux_radius, sel=slice(None)):
    median, mad, nmad = calc_nmad(cat['FLUX_RADIUS'][sel])
    return (cat['FLUX_RADIUS'] - median)/mad

def match_catalogs(ra1,dec1,ra2,dec2):
    """
    Parameters
    ----------
    ra1, dec1 : ra,dec pair from first catalog
    ra2, dec2 : ra,dec pair from second catalog
    sep : maximum separation for the match (arcsec)
    
    Returns
    idx1, idx2 : indices into the input catalogs of matched objects
    """
    coord1 = SkyCoord(ra1*u.deg,dec1*u.deg)
    coord2 = SkyCoord(ra2*u.deg,dec2*u.deg)

    idx, d2d, d3d = match_coordinates_sky(coord1, coord2)
    return idx, d2d

def create_diagnostics(cat, gaia):
    # Diagnostics
    columns = [desc[0] for desc in cat.dtype.descr if len(desc)==2]
    df = pd.DataFrame()
    for column in columns:
        df[column] = cat[column].byteswap().newbyteorder()
    df['FLUX_APER_8'] = cat['FLUX_APER'][:,8].byteswap().newbyteorder()
    df['FLUXERR_APER_8'] = cat['FLUXERR_APER'][:,8].byteswap().newbyteorder()
    df['MAG_APER_8'] = cat['MAG_APER'][:,8].byteswap().newbyteorder()
    df['MAGERR_APER_8'] = cat['MAGERR_APER'][:,8].byteswap().newbyteorder()


    gaia_columns = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                    'astrometric_excess_noise']
    for column in gaia_columns:
        df[column] = np.nan
        df.loc[sel,column] = gaia[idx[sel]][column]

    return df

def plot_snr_radius(df):
    plt.figure()
    plt.scatter(df['FLUX_APER_8']/df['FLUXERR_APER_8'],df['FLUX_RADIUS'],
                c='k',label='All DECam')

    good = (df['IMAFLAGS_ISO'] == 0)
    plt.scatter(df[good]['FLUX_APER_8']/df[good]['FLUXERR_APER_8'],df[good]['FLUX_RADIUS'],
                c='navy',label='Good DECam')

    match = good & df['GAIA_MATCH'] > 0
    sel = match
    plt.scatter(df[sel]['FLUX_APER_8']/df[sel]['FLUXERR_APER_8'],df[sel]['FLUX_RADIUS'],
                c='r',label='Gaia Match')
    sel = match & (df['GAIA_STAR'] > 0)
    plt.scatter(df[sel]['FLUX_APER_8']/df[sel]['FLUXERR_APER_8'],df[sel]['FLUX_RADIUS'],
                s=100,marker='*',ec='k', c='gold',label='Gaia Stars') 

    sel = match &  (np.abs(df['FLUX_RADIUS_NMAD']) > 5)
    plt.scatter(df[sel]['FLUX_APER_8']/df[sel]['FLUXERR_APER_8'],df[sel]['FLUX_RADIUS'],
                s=100,marker='*',ec='r', c='gold',label='Gaia Star Outliers') 

    plt.legend()
    plt.ylabel('FLUX_RADIUS')
    plt.xlabel('FLUX_APER_8/FLUXERR_APER_8') 
    plt.gca().set_xscale('log')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', help="input catalog file")
    parser.add_argument('-o','--outfile', default='out.fits', help="output catalog file")
    parser.add_argument('-d','--debug', action='store_true', help="debug output")
    args = parser.parse_args()

    hdulist = astropy.io.fits.open(args.filename)
    cat = hdulist['LDAC_OBJECTS'].data

    # Get the ra,dec centroid from the image header stored in the catalog
    ra,dec = get_centroid_from_header(hdulist)
    # Alternatively could get from the catalog
    #ra,dec = get_centroid_from_catalog(cat)

    # A bit larger than the size of a chip
    search_radius = SEARCH_RADIUS # deg

    # Get the Gaia catalog
    gaia = get_gaia_catalog(ra,dec,search_radius)

    # Match the two catalogs
    idx, d2d = match_catalogs(cat['ALPHAWIN_J2000'],cat['DELTAWIN_J2000'], 
                              gaia['ra'].data, gaia['dec'].data)

    # Create the new column we will add
    star_flag = np.zeros(len(cat),dtype=bool)

    # Select matches within some radius
    match_radius = MATCH_RADIUS # arcsec
    sel = d2d.arcsec < match_radius
    # Set matched object star flag to true if they pass our star flagging procedure
    star_flag[sel] = flag_gaia_stars(gaia[idx[sel]])
    
    flux_radius_nmad = calc_flux_radius_nmad(cat, star_flag)
    star_flag &= np.abs(flux_radius_nmad) < 5

    print(f"Writing {args.outfile}...")
    shutil.copy(args.filename,args.outfile)
    with fitsio.FITS(args.outfile,'rw') as fits:
        fits[-1].insert_column('GAIA_MATCH', sel)
        fits[-1].insert_column('GAIA_STAR', star_flag)
        fits[-1].insert_column('FLUX_RADIUS_NMAD', flux_radius_nmad)

    if args.debug:
        dbgfile = args.outfile.replace('.fits','_debug.csv')
        print(f"Writing debug output: {dbgfile}...")
        out = fitsio.read(args.outfile, ext='LDAC_OBJECTS')
        df = create_diagnostics(out, gaia)
        df.to_csv(dbgfile, index=False)
        plot_snr_radius(df)
        pngfile = dbgfile.replace('.csv','.png')
        print(f"Writing debug image: {pngfile}...")
        plt.savefig(pngfile)
