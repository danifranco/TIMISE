import sys
import os
import numpy as np
import h5py
from tqdm import tqdm

####
# list of utility functions
# 0. I/O util
# 1. binary pred -> instance seg
# 2. instance seg + pred heatmap -> instance score
# 3. instance seg -> bbox
# 4. instance seg + gt seg + instance score -> sorted match result

def readh5(filename, datasetname=None):
    fid = h5py.File(filename,'r')

    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeh5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def readh5_handle(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '':
        if sys.version[0]=='3':
            vol = list(fid)[0]
        else: # python 2
            vol = fid.keys()[0]

    return fid[vol]


def getQueryCount(ui,uc,qid):
    # memory efficient
    ui_r = [ui[ui>0].min(),max(ui.max(),qid.max())]
    rl = np.zeros(1+int(ui_r[1]-ui_r[0]),uc.dtype)
    rl[ui[ui>0]-ui_r[0]] = uc[ui>0]

    cc = np.zeros(qid.shape,uc.dtype)
    gid = np.logical_and(qid>=ui_r[0], qid<=ui_r[1])
    cc[gid] = rl[qid[gid]-ui_r[0]]
    return cc

def unique_chunk(seg, slices, chunk_size = 50, do_count = True):
    # load unique segment ids and segment sizes (in voxels) chunk by chunk
    num_z = slices[1] - slices[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size

    uc_arr = None
    ui = []
    for cid in range(num_chunk):
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([(cid + 1) * chunk_size + slices[0], slices[1]])
        chunk = np.array(seg[cid * chunk_size + slices[0]: max_idx])

        if do_count:
            ui_c, uc_c = np.unique(chunk, return_counts = True)
            if uc_arr is None:
                uc_arr = np.zeros(ui_c.max()+1, int)
                uc_arr[ui_c] = uc_c
                uc_len = len(uc_arr)
            else:
                if uc_len <= ui_c.max():
                    # at least double the length
                    uc_arr = np.hstack([uc_arr, np.zeros(max(ui_c.max()-uc_len, uc_len) + 1, int)]) #max + 1 for edge case (uc_len = ui_c.max())
                    uc_len = len(uc_arr)
                uc_arr[ui_c] += uc_c
        else:
            ui = np.unique(np.hstack([ui, np.unique(chunk)]))

    if do_count:
        ui = np.where(uc_arr>0)[0]
        return ui, uc_arr[ui]
    else:
        return ui

def unique_chunks_bbox(seg1, seg2, seg2_val, bbox, chunk_size = 50, do_count = True):
    # load unique segment ids and segment sizes (in voxels) chunk by chunk
    num_z = bbox[1] - bbox[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size

    uc_arr = None
    ui = []
    seg2_count = 0
    for cid in range(num_chunk):
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([(cid + 1) * chunk_size + bbox[0], bbox[1]])
        chunk1 = np.array(seg1[cid * chunk_size + bbox[0]:max_idx, bbox[2]:bbox[3], bbox[4]:bbox[5]])
        chunk2 = (np.array(seg2[cid * chunk_size + bbox[0]:max_idx, bbox[2]:bbox[3], bbox[4]:bbox[5]]) == seg2_val)

        seg2_count += chunk2.sum()
        chunk = chunk1 * chunk2

        if do_count:
            ui_c, uc_c = np.unique(chunk, return_counts = True)
            if uc_arr is None:
                uc_arr = np.zeros(ui_c.max()+1, int)
                uc_arr[ui_c] = uc_c
                uc_len = len(uc_arr)
            else:
                if uc_len <= ui_c.max():
                    # at least double the length
                    uc_arr = np.hstack([uc_arr, np.zeros(max(ui_c.max()-uc_len, uc_len) + 1, int)]) #max + 1 for edge case (uc_len = ui_c.max())
                    uc_len = len(uc_arr)
                uc_arr[ui_c] += uc_c
        else:
            ui = np.unique(np.hstack([ui, np.unique(chunk)]))

    if do_count:
        ui = np.where(uc_arr>0)[0]
        return ui, uc_arr[ui], seg2_count
    else:
        return ui, seg2_count


# 3. instance seg -> bbox
def seg_bbox3d(seg, slices, uid=None, chunk_size=50, verbose=True):
    """returns bounding box of segments"""
    sz = seg.shape
    assert len(sz)==3
    uic = None
    if uid is None:
        uid, uic = unique_chunk(seg, slices, chunk_size)
        uic = uic[uid>0]
        uid = uid[uid>0]
    um = int(uid.max())
    out = np.zeros((1+um,7),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1], out[:,3], out[:,5] = sz[0], sz[1], sz[2]

    num_z = slices[1] - slices[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size
    for chunk_id in range(num_chunk):
        if verbose: print('\t\t chunk %d' % chunk_id)
        z0 = chunk_id * chunk_size + slices[0]
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([z0 + chunk_size, slices[1]])
        seg_c = np.array(seg[z0 : max_idx])
        # for each slice
        for zid in np.where((seg_c>0).sum(axis=1).sum(axis=1)>0)[0]:
            sid = np.unique(seg_c[zid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,1] = np.minimum(out[sid,1], z0 + zid)
            out[sid,2] = np.maximum(out[sid,2], z0 + zid)

        # for each row
        for rid in np.where((seg_c>0).sum(axis=0).sum(axis=1)>0)[0]:
            sid = np.unique(seg_c[:,rid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,3] = np.minimum(out[sid,3],rid)
            out[sid,4] = np.maximum(out[sid,4],rid)

        # for each col
        for cid in np.where((seg_c>0).sum(axis=0).sum(axis=0)>0)[0]:
            sid = np.unique(seg_c[:,:,cid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,5] = np.minimum(out[sid,5],cid)
            out[sid,6] = np.maximum(out[sid,6],cid)
    # max + 1
    out[:,2::2] += 1
    return out[uid]

def seg_iou3d(pred, gt, slices, th_group=None, areaRng=[0,1e10], todo_id=None, chunk_size=100, crumb_size = -1, pred_bbox=None, gt_bbox=None, aux_dir="aux", verbose=True):
    # returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
    # (pred,gt)
    # return: id_1,id_2,size_1,size_2,iou
    if pred_bbox is None:
        if not os.path.exists(os.path.join(aux_dir, "pred_id.npy")):
            pred_id, pred_sz = unique_chunk(pred, slices, chunk_size)
            # remove zero seg-id
            pred_sz = pred_sz[pred_id > 0]
            pred_id = pred_id[pred_id > 0]
            np.save(os.path.join(aux_dir, "pred_sz.npy"), pred_sz)
            np.save(os.path.join(aux_dir, "pred_id.npy"), pred_id)
        else:
            pred_sz = np.load(os.path.join(aux_dir, "pred_sz.npy"))
            pred_id = np.load(os.path.join(aux_dir, "pred_id.npy"))
    else:
        pred_id = pred_bbox[:, 0]
        pred_sz = None
        if pred_bbox.shape[1]==8:
            pred_sz = pred_bbox[:, -1]
        # okay not to have it as it can be computed later
    if todo_id.max() > pred_id.max():
        raise ValueError('The predict-score has bigger id (%d) than the prediction (%d)' % (todo_id.max(), pred_id.max()))

    predict_sz_rl = np.zeros(int(pred_id.max()) + 1,int)
    if pred_sz is not None:
        predict_sz_rl[pred_id] = pred_sz

    if gt_bbox is None or gt_bbox.shape[1]!=8:
        # input gt_bbox has to have volume, o/w recompute it
        # some gt can be huge and we need to do the chunk anyway
        if not os.path.exists(os.path.join(aux_dir, "gt_id.npy")):
            gt_id, gt_sz = unique_chunk(gt, slices, chunk_size)
            gt_sz = gt_sz[gt_id > 0]
            gt_id = gt_id[gt_id > 0]
            np.save(os.path.join(aux_dir, "gt_sz.npy"), gt_sz)
            np.save(os.path.join(aux_dir, "gt_id.npy"), gt_id)
        else:
            gt_sz = np.load(os.path.join(aux_dir, "gt_sz.npy"))
            gt_id = np.load(os.path.join(aux_dir, "gt_id.npy"))
    else:
        gt_id = gt_bbox[:, 0]
        gt_sz = gt_bbox[:, -1]
    rl_gt = None
    if crumb_size > -1 and gt_sz is not None:
        gt_id = gt_id[gt_sz >= crumb_size]
        gt_sz = gt_sz[gt_sz >= crumb_size]

    if todo_id is None:
        todo_id = pred_id
        todo_sz = pred_sz
    else:
        todo_sz = predict_sz_rl[todo_id]

    if pred_bbox is None:
        if not os.path.exists(os.path.join(aux_dir, "pred_bbox.npy")):
            if verbose: print('\t compute bounding boxes')
            pred_bbox = seg_bbox3d(pred, slices, uid = todo_id, chunk_size = chunk_size, verbose=verbose)[:,1:]
            np.save(os.path.join(aux_dir, "pred_bbox.npy"), pred_bbox)
        else:
            pred_bbox = np.load(os.path.join(aux_dir, "pred_bbox.npy"))

    if th_group is not None: # regular area range
        th_id = np.unique(th_group[:, 1])
        num_group = len(th_id)
    else: # threshold group
        num_group = areaRng.shape[0] - 1
        th_id = None
    # num_group+1: add all together
    result_p = np.zeros((len(todo_id), 2+3*(num_group+1)), float)
    result_p[:,0] = todo_id
    result_p[:,1] = todo_sz

    gt_matched_id = np.zeros(1+gt_id.max(), int)
    gt_matched_iou = np.zeros(1+gt_id.max(), float)

    if verbose: print('\t compute iou matching')
    
    for j,i in tqdm(enumerate(todo_id), total=len(todo_id)):
        # if not from_iou_matching:
        # Find intersection of pred and gt instance inside bbox, call intersection match_id
        bb = pred_bbox[j]
        # can be big memory
        #match_id, match_sz=np.unique(np.array(gt[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]])*(np.array(pred[bb[0]:bb[1],bb[2]:bb[3], bb[4]:bb[5]])==i),return_counts=True)
        match_id, match_sz, pred_sz_i  = unique_chunks_bbox(gt, pred, i, bb, chunk_size)
        
        # in case the pred_bbox doesn't have size computed
        predict_sz_rl[i] = pred_sz_i
        match_id_g = np.isin(match_id, gt_id)
        match_sz = match_sz[match_id_g] # get intersection counts
        match_id = match_id[match_id_g] # get intersection ids
                      
        if len(match_id)>0:
            # get count of all preds inside bbox (assume gt_id,match_id are of ascending order)
            gt_sz_match = getQueryCount(gt_id, gt_sz, match_id)
            ious = match_sz.astype(float)/(todo_sz[j] + gt_sz_match - match_sz) #all possible iou combinations of bbox ids are contained

            for r in range(num_group + 1): # fill up all, then s, m, l
                if th_id is None: # area-based grouping
                    gid = (gt_sz_match>areaRng[r,0])*(gt_sz_match<=areaRng[r,1])
                else: # precomputed grouping
                    if r == 0: # all groups
                        gid = gt_sz_match > 0
                    else:
                        gid = getQueryCount(th_group[:,0], th_group[:,1], match_id) == th_id[r-1]
                if sum(gid)>0:
                    idx_iou_max = np.argmax(ious*gid)
                    result_p[j,2+r*3:2+r*3+3] = [ match_id[idx_iou_max], gt_sz_match[idx_iou_max], ious[idx_iou_max] ]
            # update best pred match for each gt
            gt_todo = gt_matched_iou[match_id]<ious
            gt_matched_iou[match_id[gt_todo]] = ious[gt_todo]
            gt_matched_id[match_id[gt_todo]] = i

    # get the rest: false negative + dup
    fn_gid = gt_id[np.isin(gt_id, result_p[:,2], assume_unique=False, invert=True)]
    fn_gic = gt_sz[np.isin(gt_id, fn_gid)]
    fn_iou = gt_matched_iou[fn_gid]
    fn_pid = gt_matched_id[fn_gid]
    fn_pic = predict_sz_rl[fn_pid]

    # add back duplicate
    # instead of bookkeeping in the previous step, faster to redo them
    # fn_pic can be non-zero: exist a match, but gt is not the best match
    result_fn = np.vstack([fn_pid, fn_pic, fn_gid, fn_gic, fn_iou]).T

    return result_p, result_fn


def cable_length(vertices, edges, res = [1,1,1]):
    # make sure vertices and res have the same order of zyx
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    if len(edges) == 0:
        return 0
    v1 = vertices[edges[:,0]]
    v2 = vertices[edges[:,1]]

    delta = (v2 - v1) * res
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)

def volume_to_cable_length(vol,resolution=[30,8,8],dust_threshold=100):
    # resolution: zyx order 
    skels = kimimaro.skeletonize(vol, parallel=0, parallel_chunk_size=100, dust_threshold=dust_threshold)
    sids = skels.keys()
    result = np.zeros([len(sids),2])
    result[:,0] = sids
    for i,sid in enumerate(sids):
        ind_skel = skels[sid]
        vertices = ind_skel.vertices

        # Cable length
        l = cable_length(ind_skel.vertices, ind_skel.edges, res = resolution)
        result[i,1] = l
    return result
    
def seg_iou3d_sorted(pred, gt, score, slices, th_group=None, areaRng = [0,1e10], chunk_size = 250, crumb_size = -1, pred_bbox=None, gt_bbox=None, aux_dir="aux", verbose=True):
    # pred_bbox: precomputed if needed
    # pred_score: Nx2 [id, score]
    # 1. sort prediction by confidence score
    relabel = np.zeros(int(np.max(score[:,0])+1), float)
    relabel[score[:,0].astype(int)] = score[:,1]

    # 1. sort the prediction by confidence
    pred_id = np.unique(score[:,0])
    pred_id = pred_id[pred_id>0]
    pred_id_sorted = np.argsort(-relabel[pred_id])
    
    # Save prediction info
    os.makedirs(aux_dir, exist_ok=True)
    np.save(os.path.join(aux_dir, "pred_labels.npy"), pred_id[pred_id_sorted])
    
    result_p, result_fn = seg_iou3d(pred, gt, slices, th_group, areaRng, pred_id[pred_id_sorted], chunk_size, crumb_size, pred_bbox, gt_bbox, aux_dir, verbose=verbose)
    # format: pid,pc,p_score, gid,gc,iou
    pred_score_sorted = relabel[pred_id_sorted].reshape(-1,1)
    return result_p, result_fn, pred_score_sorted
