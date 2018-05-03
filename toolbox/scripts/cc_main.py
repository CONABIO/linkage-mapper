#!/usr/bin/env python2.6
# Authors: Darren Kavanagh and Brad McRae

"""Climate Linkage Mapper

Reguired Software:
ArcGIS 10.x with Spatial Analyst extension
Python 2.6
Numpy 1.3

"""

# $Revision$

import os
import sys
import csv
import itertools
import traceback
from datetime import datetime

# import arcinfo  # Import arcinfo license. Needed before arcpy import.
# import arcpy
# import arcpy.sa as sa

from cc_config import cc_env
import cc_util
import lm_master
from lm_config import tool_env as lm_env
import lm_util

# ----
import rasterio
from rasterio.warp import (calculate_default_transform,
                             reproject,
                             Resampling)
from rasterio import features
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import logging
from rasterstats import zonal_stats
from rasterstats.utils import VALID_STATS
import pandas as pd
# ----

_SCRIPT_NAME = "cc_main.py"

TFORMAT = "%m/%d/%y %H:%M:%S"

FR_COL = "From_Core"
TO_COL = "To_Core"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(argv=None):
    """Main function for Climate Linkage Mapper tool"""
    start_time = datetime.now()
    # print "Start time: %s" % start_time.strftime(TFORMAT)

    if argv is None:
        argv = sys.argv
    try:
        cc_env.configure(argv)
        cc_util.check_cc_project_dir()

        grass_dir_setup()

        check_out_sa_license()
        arc_wksp_setup()
        config_lm()
        log_setup()

        run_analysis()

    except arcpy.ExecuteError:
        msg = arcpy.GetMessages(2)
        arcpy.AddError(arcpy.GetMessages(2))
        lm_util.write_log(msg)
        exc_traceback = sys.exc_info()[2]
        lm_util.gprint("Traceback (most recent call last):\n" +
                       "".join(traceback.format_tb(exc_traceback)[:-1]))

    except Exception:
        exc_value, exc_traceback = sys.exc_info()[1:]
        arcpy.AddError(exc_value)
        lm_util.gprint("Traceback (most recent call last):\n" +
                       "".join(traceback.format_tb(exc_traceback)))
    finally:
        delete_proj_files()
        arcpy.CheckInExtension("Spatial")
        print_runtime(start_time)


def grass_dir_setup():
    """Create new folder for GRASS workspace"""
    gisdbase = os.path.join(cc_env.proj_dir, "gwksp")

    # Remove GRASS directory if it exists
    if not cc_util.remove_grass_wkspc(gisdbase):
        lm_util.warn("\nCannot delete GRASS workspace from earlier"
                         " run."
                         "\nPlease choose a new project directory.")
        raise Exception("Cannot delete GRASS workspace: " + gisdbase)


def check_out_sa_license():
    """Check out the ArcGIS Spatial Analyst extension license"""
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
    else:
        raise


def arc_wksp_setup():
    """Setup ArcPy workspace"""
    arcpy.env.overwriteOutput = True
    arcpy.env.cellSize = "MAXOF"  # Setting to default. For batch runs.
    if arcpy.Exists(cc_env.out_dir):
        cc_util.delete_features(cc_env.out_dir)
    arcpy.env.workspace = cc_util.mk_proj_dir(cc_env.out_dir)
    arcpy.env.scratchWorkspace = cc_util.mk_proj_dir(cc_env.tmp_dir)


def config_lm():
    """Configure Linkage Mapper"""
    lm_arg = [_SCRIPT_NAME, cc_env.proj_dir, cc_env.prj_core_fc,
              cc_env.core_fld, cc_env.prj_resist_rast, "false", "false", "#",
              "#", "true", "false", cc_env.prune_network, cc_env.max_nn,
              cc_env.nn_unit, cc_env.keep_constelations, "true",
              cc_env.WRITETRUNCRASTER, cc_env.CWDTHRESH, "#", "#", "#",
              cc_env.OUTPUTFORMODELBUILDER, cc_env.LMCUSTSETTINGS]
    lm_env.configure(lm_env.TOOL_CC, lm_arg)
    lm_util.create_dir(lm_env.DATAPASSDIR)
    lm_util.gprint('\nClimate Linkage Mapper Version ' + lm_env.releaseNum)
    lm_util.gprint('NOTE: This tool runs best with BACKGROUND '
                   'PROCESSING (see user guide).')


def log_setup():
    """Set up Linkage Mapper logging"""
    lm_util.create_dir(lm_env.LOGDIR)
    lm_util.create_dir(lm_env.MESSAGEDIR)
    lm_env.logFilePath = lm_util.create_log_file(lm_env.MESSAGEDIR,
                                                 lm_env.TOOL,
                                                 lm_env.PARAMS)


def run_analysis():
    """Run Climate Linkage Mapper analysis"""
    import cc_grass_cwd  # Cannot import until configured

    zonal_tbl = "zstats.dbf"

    cc_copy_inputs()  # Clip inputs and create project area raster

    # Get zonal statistics for cores and climate
    logger.info("\nCALCULATING ZONAL STATISTICS FROM CLIMATE RASTER")

    # http://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/zonal-statistics-as-table.htm
    # Toma el raster clima saca las estadisticas en cada parche:
    # - count
    # - area
    # - media
    # - sd
    # - min
    # - max
    # - rango = max-min
    # - suma = suma de pixeles
    #  FID | media | sd | min | max | ...
    climate_stats = pd.DataFrame(zonal_stats(
        cc_env.prj_core_fc,
        cc_env.prj_climate_rast,
        stats=VALID_STATS
    ))


    # Create core pairings table and limit based upon climate threshold
    core_pairings = create_pair_tbl(climate_stats)

    # Generate link table, calculate CWD and run Linkage Mapper
    if int(arcpy.GetCount_management(core_pairings).getOutput(0)) == 0:
        lm_util.warn("\nNo core pairs within climate threshold. "
                         "Program will end")
    else:
        # Process pairings and generate link table
        grass_cores = process_pairings(core_pairings)
        if not grass_cores:
            lm_util.warn("\nNo core pairs within Euclidean distances. "
                             "Progam will end")
        else:
            # Create CWD using Grass
            cc_grass_cwd.grass_cwd(grass_cores)
            # Run Linkage Mapper
            lm_util.gprint("\nRUNNING LINKAGE MAPPER "
                           "TO CREATE CLIMATE CORRIDORS")
            lm_master.lm_master()


def cc_copy_inputs():
    """Clip Climate Linkage Mapper inputs to smallest extent

    Copy and transform every input data to a common extent and  resolution,
    this includes climate raster, resistance ratser and cores features.
    """
    ext_poly = os.path.join(cc_env.out_dir, "ext_poly.shp")  # Extent polygon
    try:
        logger.info("\nCOPYING LAYERS AND, IF NECESSARY, REDUCING EXTENT")

        cc_climate = rasterio.open(cc_env.climate_ras)
        cc_climate_meta = cc_climate.meta.copy()
        climate_extent = cc_climate.bounds

        if cc_env.resist_rast is not None:
            cc_resist = rasterio.open(cc_env.resist_rast)
            resist_extent = cc_resist.bounds

            xmin = max(climate_extent.left, resist_extent.left)
            ymin = max(climate_extent.bottom, resist_extent.bottom)
            xmax = min(climate_extent.right, resist_extent.right)
            ymax = min(climate_extent.top, resist_extent.top)

            # Set to minimum extent if resistance raster was given
            minimum_extent = rasterio.coords.BoundingBox(xmin,
                ymin,
                xmax,
                ymax)

            # Want climate and resistance rasters in same spatial ref
            # with same nodata cells
            nodata_climate_rast = cc_climate.nodata

            dst_crs = cc_climate.crs
            profile = cc_resist.profile
            dst_affine, dst_width, dst_height = calculate_default_transform(
                cc_resist.crs, dst_crs, cc_resist.width, cc_resist.height,
                *cc_resist.bounds)
            profile.update({
                'crs': dst_crs,
                'transform': dst_affine,
                'affine': dst_affine,
                'width': dst_width,
                'height': dst_height,
                'nodata': nodata_climate_rast
                })

            with rasterio.open(cc_env.prj_resist_rast, 'w', **profile) as dst:
                src_array = cc_resist.read(1)
                dst_array = np.empty((dst_height, dst_width), dtype='float32')

                reproject(
                    # Source parameters
                    source=src_array,
                    src_crs=cc_resist.crs,
                    src_transform=cc_resist.affine,
                    # Destination paramaters
                    destination=dst_array,
                    dst_transform=dst_affine,
                    dst_crs=dst_crs,
                    # Configuration
                    resampling=Resampling.nearest,
                    num_threads=2)

                dst.write(dst_array, 1)
        else:
            xmin = climate_extent.left
            ymin = climate_extent.bottom
            xmax = climate_extent.right
            ymax = climate_extent.top

            ones_resist_array = np.ones((cc_climate.height,
                                         cc_climate.width), dtype='float32')
            cc_climate_mask = cc_climate.read_masks(1)

            profile = cc_climate.profile

            with rasterio.open(cc_env.prj_resist_rast, 'w', **profile) as dst:
                dst.write(ones_resist_array, 1)
                dst.write_mask(cc_climate_mask)
            ones_resist_rast = sa.Con(
                sa.IsNull(cc_env.climate_rast),
                sa.Int(cc_env.climate_rast), 1)
            ones_resist_rast.save(cc_env.prj_resist_rast)


        # Create core raster
        cores_gpd = gpd.read_file(cc_env.core_fc)
        cores_it = ((feat[1].geometry, feat[1][cc_env.core_fld])
                   for feat in cores_gpd.iterrows())

        cores_im = features.rasterize(cores_it,
                                      out_shape=cc_climate.shape,
                                      transform=cc_climate.transform)

        cores_profile = cc_climate.meta.copy()
        cores_profile.update({
            'nodata': 0,
            'driver': 'HFA',
            'count': 1,
            'dtype': rasterio.uint8
        })

        with rasterio.open(cc_env.prj_core_rast, 'w',
                           **cores_profile) as dst:
            dst.write(cores_im, indexes=1)

        # Create boundary box
        ext_feat = box(610056.2425,
                       686239.224588,
                       842393.482691,
                       882993.283848)

        # Clip core feature class
        ext_gdf = gpd.GeoDataFrame(gpd.GeoSeries(ext_feat),
                                   columns=["geometry"],
                                   crs = cc_climate.crs)

        ext_gdf.to_file('box.shp')
        cores_clip = gpd.overlay(ext_gdf, cores_gpd, how="intersection")

        cores_clip.crs = cores_gpd.crs
        cores_clip.to_file(cc_env.prj_core_fc)

    except Exception:
        raise


def create_pair_tbl(climate_stats):
    """Create core pair table and limit to climate threshold """
    cpair_tbl = pair_cores("corepairs.dbf")
    if int(arcpy.GetCount_management(cpair_tbl).getOutput(0)) > 0:
        limit_cores(cpair_tbl, climate_stats)
    return cpair_tbl


def pair_cores(cpair_tbl):
    """Create table with all possible core to core combinations"""
    srows, outputrow, irows = None, None, None

    try:
        lm_util.gprint("\nCREATING CORE PAIRINGS TABLE")
        arcpy.CreateTable_management(cc_env.out_dir, cpair_tbl, "", "")
        arcpy.AddField_management(cpair_tbl, FR_COL, "Long", "", "",
                                  "", "", "NON_NULLABLE")
        arcpy.AddField_management(cpair_tbl, TO_COL, "Long", "", "",
                                  "", "", "NON_NULLABLE")
        arcpy.DeleteField_management(cpair_tbl, "Field1")

        srows = arcpy.SearchCursor(cc_env.prj_core_fc, "", "",
                                   cc_env.core_fld, cc_env.core_fld + " A")

        cores_list = [srow.getValue(cc_env.core_fld) for srow in srows]
        cores_product = list(itertools.combinations(cores_list, 2))

        lm_util.gprint("There are " + str(len(cores_list)) + " unique "
                       "cores and " + str(len(cores_product)) + " pairings")

        irows = arcpy.InsertCursor(cpair_tbl)
        for nrow in cores_product:
            outputrow = irows.newRow()
            outputrow.setValue(FR_COL, int(nrow[0]))
            outputrow.setValue(TO_COL, int(nrow[1]))
            irows.insertRow(outputrow)

        return cpair_tbl

    except Exception:
        raise
    finally:
        if srows:
            del srows
        if outputrow:
            del outputrow
        if irows:
            del irows


def limit_cores(pair_tbl, stats_tbl):
    """Limit core pairs based upon climate threshold"""
    pair_vw = "dist_tbvw"
    stats_vw = "stats_tbvw"
    core_id = cc_env.core_fld.upper()

    try:
        lm_util.gprint("\nLIMITING CORE PAIRS BASED UPON CLIMATE "
                       "THRESHOLD")

        arcpy.MakeTableView_management(pair_tbl, pair_vw)
        arcpy.MakeTableView_management(stats_tbl, stats_vw)

        # Add basic stats to distance table
        lm_util.gprint("Joining zonal statistics to pairings table")
        add_stats(stats_vw, core_id, "fr", pair_vw, TO_COL)
        add_stats(stats_vw, core_id, "to", pair_vw, FR_COL)

        # Calculate difference of 2 std
        lm_util.gprint("Calculating difference of 2 std")
        diffu_2std = "diffu_2std"
        arcpy.AddField_management(pair_vw, diffu_2std, "Float", "", "",
                                  "", "", "NULLABLE")
        arcpy.CalculateField_management(pair_vw, diffu_2std,
                                        "abs(!frumin2std! - !toumin2std!)",
                                        "PYTHON_9.3")

        # Filter distance table based on inputed threshold and delete rows
        lm_util.gprint("Filtering table based on threshold")
        diffu2std_fld = arcpy.AddFieldDelimiters(pair_vw, diffu_2std)
        expression = diffu2std_fld + " <= " + str(cc_env.climate_threshold)
        arcpy.SelectLayerByAttribute_management(pair_vw, "NEW_SELECTION",
                                                expression)
        rows_del = int(arcpy.GetCount_management(pair_vw).getOutput(0))
        if rows_del > 0:
            arcpy.DeleteRows_management(pair_vw)
        lm_util.gprint(str(rows_del) + " rows deleted")

    except Exception:
        raise
    finally:
        cc_util.delete_features([stats_vw, pair_vw])


def add_stats(stats_vw, core_id, fld_pre, table_vw, join_col):
    """Add zonal and calculated statistics to stick table"""
    tmp_mea = fld_pre + "_tmp_mea"
    tmp_std = fld_pre + "_tmp_std"
    umin2std = fld_pre + "umin2std"

    # Add fields to stick table - has to be done before join
    arcpy.AddField_management(table_vw, tmp_mea, "Float", "", "",
                              "", "", "NULLABLE")
    arcpy.AddField_management(table_vw, tmp_std, "Float", "", "",
                              "", "", "NULLABLE")
    arcpy.AddField_management(table_vw, umin2std, "Float", "", "",
                              "", "", "NULLABLE")

    # Join distance table to zonal stats table
    arcpy.AddIndex_management(table_vw, FR_COL, "fridx", "NON_UNIQUE",
                              "ASCENDING")
    arcpy.AddIndex_management(table_vw, TO_COL, "toidx", "NON_UNIQUE",
                              "ASCENDING")
    arcpy.AddIndex_management(stats_vw, core_id, "coreidx", "UNIQUE",
                              "ASCENDING")
    arcpy.AddJoin_management(table_vw, join_col, stats_vw, core_id)

    tbl_name = arcpy.Describe(table_vw).baseName
    stats_tbl_nm = arcpy.Describe(stats_vw).baseName

    # Insert values into fields
    mean_value = "!" + stats_tbl_nm + ".MEAN" + "!"
    std_value = "!" + stats_tbl_nm + ".STD" + "!"
    mea_fld = "!" + tbl_name + "." + tmp_mea + "!"
    std_fld = "!" + tbl_name + "." + tmp_std + "!"

    arcpy.CalculateField_management(table_vw, tmp_mea, mean_value,
                                    "PYTHON_9.3")
    arcpy.CalculateField_management(table_vw, tmp_std, std_value,
                                    "PYTHON_9.3")
    expression = mea_fld + " - " + std_fld + " - " + std_fld
    arcpy.CalculateField_management(table_vw, umin2std, expression,
                                    "PYTHON_9.3")

    # Remove join
    arcpy.RemoveJoin_management(table_vw, stats_tbl_nm)


def process_pairings(pairings):
    """Limit core pairings based on distance inputs and create linkage table

    Requires ArcInfo license.

    """
    lm_util.gprint("\nLIMITING CORE PAIRS BASED ON INPUTED DISTANCES AND "
                   "GENERATING LINK TABLE")
    # Simplify cores based on booolean in config
    if cc_env.simplify_cores:
        corefc = simplify_corefc()
    else:
        corefc = cc_env.prj_core_fc
    core_pairs, frm_cores = pairs_from_list(pairings)
    # Create link table
    core_list = create_lnk_tbl(corefc, core_pairs, frm_cores)
    return sorted(core_list)


def pairs_from_list(pairings):
    """Get list of core pairings and 'from cores'"""
    frm_cores = set()
    core_pairs = []
    srows = arcpy.SearchCursor(pairings, "", "", FR_COL + "; " + TO_COL)
    for srow in srows:
        from_core = srow.getValue(FR_COL)
        to_core = str(srow.getValue(TO_COL))
        frm_cores.add(from_core)
        core_pairs.append([str(from_core), to_core])
    frm_cores = [str(x) for x in frm_cores]
    return core_pairs, frm_cores


def create_lnk_tbl(corefc, core_pairs, frm_cores):
    """Create link table file and limit based on near table results"""
    fcore_vw = "fcore_vw"
    tcore_vw = "tcore_vw"
    jtocore_fn = cc_env.core_fld[:8] + "_1"  # dbf field length
    near_tbl = os.path.join(cc_env.out_dir, "neartbl.dbf")
    link_file = os.path.join(lm_env.DATAPASSDIR, "linkTable_s2.csv")

    link_tbl, srow, srows = None, None, None

    try:
        link_tbl = open(link_file, 'wb')
        writer = csv.writer(link_tbl, delimiter=',')
        headings = ["# link", "coreId1", "coreId2", "cluster1", "cluster2",
                    "linkType", "eucDist", "lcDist", "eucAdj", "cwdAdj"]
        writer.writerow(headings)

        core_list = set()
        no_cores = str(len(frm_cores))
        i = 1

        coreid_fld = arcpy.AddFieldDelimiters(corefc, cc_env.core_fld)

        for core_no, frm_core in enumerate(frm_cores):
            # From cores
            expression = coreid_fld + " = " + frm_core
            arcpy.MakeFeatureLayer_management(corefc, fcore_vw, expression)

            # To cores
            to_cores_lst = [x[1] for x in core_pairs if frm_core == x[0]]
            to_cores = ', '.join(to_cores_lst)
            expression = coreid_fld + " in (" + to_cores + ")"
            arcpy.MakeFeatureLayer_management(corefc, tcore_vw, expression)
            lm_util.gprint("Calculating Euclidean distance/s from Core " +
                           frm_core + " to " + str(len(to_cores_lst)) +
                           " other cores" + " (" + str(core_no + 1) + "/" +
                           no_cores + ")")

            # Generate near table for these core pairings
            arcpy.GenerateNearTable_analysis(
                fcore_vw, tcore_vw, near_tbl,
                cc_env.max_euc_dist, "NO_LOCATION", "NO_ANGLE", "ALL")

            # Join near table to core table
            arcpy.JoinField_management(near_tbl, "IN_FID", corefc,
                                       "FID", cc_env.core_fld)
            arcpy.JoinField_management(near_tbl, "NEAR_FID", corefc,
                                       "FID", cc_env.core_fld)

            # Limit pairings based on inputed Euclidean distances
            srow, srows = None, None
            euc_dist_fld = arcpy.AddFieldDelimiters(near_tbl, "NEAR_DIST")
            expression = (euc_dist_fld + " > " + str(cc_env.min_euc_dist))
            srows = arcpy.SearchCursor(near_tbl, expression, "",
                                       jtocore_fn + "; NEAR_DIST",
                                       jtocore_fn + " A; NEAR_DIST A")

            # Process near table and output into a link table
            srow = srows.next()
            if srow:
                core_list.add(int(frm_core))
                while srow:
                    to_coreid = srow.getValue(jtocore_fn)
                    dist_value = srow.getValue("NEAR_DIST")
                    writer.writerow([i, frm_core, to_coreid, -1, -1, 1,
                                     dist_value, -1, -1, -1])
                    core_list.add(to_coreid)
                    srow = srows.next()
                    i += 1

    except Exception:
        raise
    finally:
        cc_util.delete_features(
            [near_tbl, os.path.splitext(corefc)[0] + "_Pnt.shp"])
        if link_tbl:
            link_tbl.close()
        if srow:
            del srow
        if srows:
            del srows

    return core_list


def simplify_corefc():
    """Simplify core feature class"""
    lm_util.gprint("Simplifying polygons to speed up core pair "
                   "distance calculations")
    corefc = cc_env.core_simp
    climate_rast = arcpy.Raster(cc_env.prj_climate_rast)
    tolerance = climate_rast.meanCellHeight / 3
    arcpy.cartography.SimplifyPolygon(
        cc_env.prj_core_fc, corefc,
        "POINT_REMOVE", tolerance, "#", "NO_CHECK")
    return corefc


def delete_proj_files():
    """Delete project input files on ending of analysis

    Keep prj_resist_rast, prj_core_fc and out_dir for reruns.

    """
    cpath = os.getcwd()  # For files left behind by arcpy
    prj_files = (
        [cc_env.prj_climate_rast, cc_env.prj_core_rast, cc_env.tmp_dir,
         os.path.join(cpath, ".prj"), os.path.join(cpath, "info")])
    if cc_env.simplify_cores:
        prj_files.append(cc_env.core_simp)
    cc_util.delete_features(prj_files)


def print_runtime(stime):
    """Print process time when running from script"""
    etime = datetime.now()
    rtime = etime - stime
    hours, minutes = ((rtime.days * 24 + rtime.seconds // 3600),
                      (rtime.seconds // 60) % 60)
    #print "End time: %s" % etime.strftime(TFORMAT)
    #print "Elapsed time: %s hrs %s mins" % (hours, minutes)


if __name__ == "__main__":
    main()
