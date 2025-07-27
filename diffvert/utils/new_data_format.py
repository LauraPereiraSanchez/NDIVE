""" Module containing information on how data is formatting, and functions to deal with format.

As input to the jax model, the track data is completely flattened. This module allows
access to the data through more understandable enums.
"""
from enum import IntEnum
import jax.numpy as jnp

# NUM_JET_INPUT_PARAMETERS: int = 16
NUM_JET_INPUT_PARAMETERS: int = 18  # count used as inputs in flavor tagging -> this is JetData.TRACK_SIGNED_SIG_Z0 + JET_PT + JET_ETA, as defined in get_track_inputs()

## NEW FORMAT -> including tertiary vertex info
class JetData(IntEnum): 
    """ Store which indices of track inputs mean what. """
    TRACK_PT = 0
    TRACK_D0 = 1
    TRACK_Z0 = 2
    TRACK_PHI = 3
    TRACK_THETA = 4
    TRACK_RHO = 5
    TRACK_PT_FRACTION_LOG = 6  # log(track_pt / jet_pt)
    TRACK_DELTA_R = 7  # deltaR(track, jet)
    TRACK_PT_ERR = 8
    TRACK_D0_ERR = 9
    TRACK_Z0_ERR = 10
    TRACK_PHI_ERR = 11
    TRACK_THETA_ERR = 12
    TRACK_RHO_ERR = 13
    TRACK_SIGNED_SIG_D0 = 14  # signed d0 significance
    TRACK_SIGNED_SIG_Z0 = 15  # signed z0 significance
    # begin true production vertex info
    TRACK_PROD_VTX_X = 16
    TRACK_PROD_VTX_Y = 17
    TRACK_PROD_VTX_Z = 18
    # begin coordinates of true hadron decay for b,c jets ((0,0,0) otherwise) - secondary
    HADRON_X = 19
    HADRON_Y = 20
    HADRON_Z = 21
    # begin coordinates of true hadron decay for b,c jets ((0,0,0) otherwise) - tertiary
    HADRON_TV_X = 22
    HADRON_TV_Y = 23
    HADRON_TV_Z = 24
    # begin jet info
    N_TRACKS = 25
    # a counter of HF vertices
    N_HF_VERTICES = 26
    # a counter of b,c hadrons
    N_BHADRON = 27
    N_CHADRON = 28
    TRACK_VERTEX_INDEX = 29
    # begin jet-level variables (really should be renamed to only be JET_X not TRACK_JET_X)
    TRACK_JET_PHI = 30
    TRACK_JET_THETA = 31
    TRACK_JET_PT = 32
    TRACK_JET_ETA = 33
    TRACK_JET_FLAVOR = 34
    # 32-46 binary track-level variables for training the vertex pairs auxiliary task
    # 47-49 binary jet-level variables for training the jet flavor task
    # 50-53 binary track-level variables for training the track origin auxiliary task
    IS_BJET = 50
    IS_CJET = 51
    IS_LJET = 52
    TRACK_FROM_B = 53
    TRACK_FROM_C = 54
    TRACK_FROM_ORIGIN = 55
    TRACK_FROM_OTHER = 56
    

# NEW ONE - work in progress
class JetPrediction(IntEnum):
    """ Store which indices of JetPrediction mean what. """
    PROB_B = 0
    PROB_C = 1
    PROB_U = 2
    VERTEX1_X = 3
    VERTEX1_Y = 4
    VERTEX1_Z = 5
    VERTEX1_COV_XX = 6
    VERTEX1_COV_XY = 7
    VERTEX1_COV_XZ = 8
    VERTEX1_COV_YX = 9
    VERTEX1_COV_YY = 10
    VERTEX1_COV_YZ = 11
    VERTEX1_COV_ZX = 12
    VERTEX1_COV_ZY = 13
    VERTEX1_COV_ZZ = 14
    VERTEX1_CHISQ = 15
    VERTEX2_X = 16
    VERTEX2_Y = 17
    VERTEX2_Z = 18
    VERTEX2_COV_XX = 19
    VERTEX2_COV_XY = 20
    VERTEX2_COV_XZ = 21
    VERTEX2_COV_YX = 22
    VERTEX2_COV_YY = 23
    VERTEX2_COV_YZ = 24
    VERTEX2_COV_ZX = 25
    VERTEX2_COV_ZY = 26
    VERTEX2_COV_ZZ = 27
    VERTEX2_CHISQ = 28
    VERTEX3_X = 29
    VERTEX3_Y = 30
    VERTEX3_Z = 31
    VERTEX3_COV_XX = 32
    VERTEX3_COV_XY = 33
    VERTEX3_COV_XZ = 34
    VERTEX3_COV_YX = 35
    VERTEX3_COV_YY = 36
    VERTEX3_COV_YZ = 37
    VERTEX3_COV_ZX = 38
    VERTEX3_COV_ZY = 39
    VERTEX3_COV_ZZ = 40
    VERTEX3_CHISQ = 41
    VERTEX4_X = 42
    VERTEX4_Y = 43
    VERTEX4_Z = 44
    VERTEX4_COV_XX = 45
    VERTEX4_COV_XY = 46
    VERTEX4_COV_XZ = 47
    VERTEX4_COV_YX = 48
    VERTEX4_COV_YY = 49
    VERTEX4_COV_YZ = 50
    VERTEX4_COV_ZX = 51
    VERTEX4_COV_ZY = 52
    VERTEX4_COV_ZZ = 53
    VERTEX4_CHISQ = 54
    SLOT1_MASK = 55
    SLOT2_MASK = 56
    SLOT3_MASK = 57
    SLOT4_MASK = 58
    #TRACK_PRED_FROM_PV = 59
    #TRACK_PRED_FROM_SV = 60
    #TRACK_PRED_FROM_TV = 61
    #TRACK_PRED_FROM_OTHER = 62
    VERTEX_TRACK_STARTS = 59

def get_track_inputs(tracks):
    """ Return inpust from full track data.
    
    Args:
        tracks: 'num_jets' x 'max_num_tracks' x 'num_track_params' input tracks
    Returns:
        values used as inputs (specifically avoids truth-value data) of shape
            'num_jets' x 'max_num_tracks' x 'NUM_JET_INPUT_PARAMETERS'
    """
    # return tracks[:,:,0:JetData.TRACK_PROD_VTX_X]

    # add jet-level pt, eta to track-level params as done in gn1
    return jnp.concatenate(
        (
            tracks[:,:,0:JetData.TRACK_PROD_VTX_X], # all single-track, non-truth variables
            jnp.log(tracks[:,:,JetData.TRACK_JET_PT:JetData.TRACK_JET_PT+1]),
            tracks[:,:,JetData.TRACK_JET_ETA:JetData.TRACK_JET_ETA+1],
        ),
        axis=2,
    )


def create_tracks_mask(tracks, pad_for_ghost=False):
    """ Create a mask of size 'num_jets' x 'max_num_tracks' for which tracks are real.
    
    Args:
        tracks: 'num_jets' x 'max_num_tracks' x 'num_track_params' input tracks
        pad_for_ghost: whether or not to pad for later inclusion of ghost track
    Returns:
        boolean mask indicating if tracks are real or padding ('num_jets' x 'max_num_tracks')
    """
    num_jets, max_num_tracks, = tracks.shape[0:2]
    max_num_tracks += pad_for_ghost

    # each jet has track indices 0, 1, 2, ... max_num_tracks-1
    track_indices = jnp.tile(
        jnp.arange(0,max_num_tracks,dtype=jnp.int32),
        num_jets,
    ).reshape(num_jets, max_num_tracks)

    # 'num_jets' x 'max_num_tracks' array of real tracks in each jet (repeated)
    n_trks = jnp.repeat(
        tracks[:,0,JetData.N_TRACKS]+pad_for_ghost, max_num_tracks,
    ).reshape(num_jets, max_num_tracks)

    mask =  jnp.where(track_indices < n_trks, 1, 0)
    return mask


def create_track_pairs_mask(mask):
    """ Create a mask of size 'num_jets' x 'max_num_tracks' x 'max_num_tracks' for track pairs.

    A track pair is valid iff both tracks are valid.
    
    Args:
        mask: 'num_jets' x 'max_num_tracks' boolean mask array
    Returns:
        'num_jets' x 'max_num_tracks' x 'max_num_tracks' boolean track pair mask
    """
    num_jets, max_num_tracks = mask.shape

    mask_first_track = jnp.repeat(
        mask, max_num_tracks,
    ).reshape(num_jets, max_num_tracks, max_num_tracks)

    mask_second_track = jnp.repeat(
        mask.reshape(num_jets, 1, max_num_tracks), max_num_tracks, axis=1,
    ).reshape(num_jets, max_num_tracks, max_num_tracks)

    mask_track_pairs = jnp.logical_and(mask_first_track, mask_second_track)
    return mask_track_pairs


def ExtractTruth(batch_x, batch_y):
    
    jet_flav = batch_x[:, 0, JetData.IS_BJET:JetData.IS_LJET+1] # 0 bjet, 1 c-jet, 2 light jet
    SV_true = batch_x[:, 0, JetData.HADRON_X:JetData.HADRON_Z+1].reshape(-1, 3)
    TV_true = batch_x[:, 0, JetData.HADRON_TV_X:JetData.HADRON_TV_Z+1].reshape(-1, 3)

    return (jet_flav, SV_true, TV_true)

