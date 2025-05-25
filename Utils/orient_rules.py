import numpy as np
from Utils.PAG_edge import PAGEdge

def updateList(path, set, old_list):  # arguments are all lists
    return old_list + [path + [s] for s in set]

def minDiscPath(pag, a, b, c):
    p = pag.shape[0]
    visited = set([a, b, c])

    indD = [i for i in np.where((pag[a, :] != PAGEdge.NONE) & (pag[:, a] == PAGEdge.ARROW))[0] if i not in visited]

    if indD:
        path_list = updateList([a], indD, [])
        while path_list:
            mpath = path_list.pop(0)
            d = mpath[-1]

            if pag[c][d] == PAGEdge.NONE and pag[d][c] == PAGEdge.NONE:
                return mpath[::-1] + [b, c]

            pred = mpath[-2]
            visited.add(d)

            if pag[d][c] == PAGEdge.ARROW and pag[c][d] == PAGEdge.TAIL and pag[pred][d] == PAGEdge.ARROW:
                indR = [i for i in np.where((pag[d, :] != PAGEdge.NONE) & (pag[:, d] == PAGEdge.ARROW))[0] if i not in visited]
                if indR:
                    path_list = updateList(mpath, indR, path_list)

    return []

def orient_rules(pag, sepset,find_edge,rules = np.full(10,True,dtype=bool), detail_out = False,debug=False):

    p = pag.shape[0]
    #  R1 - R4
    if sepset is None or find_edge is None:
        raise ValueError('sepset or find_edge is None')
    old_pag = np.zeros((p, p))
    while not np.array_equal(pag, old_pag):
        old_pag[:] = pag
        #--------------R1-------------------------------------------
        if rules[0]:
            #i *-> j
            inds = np.argwhere((pag == PAGEdge.ARROW) & (np.transpose(pag) != PAGEdge.NONE))
            for i, j in inds:
                Ks = np.where((pag[j, :] != PAGEdge.NONE) & (pag[:, j] == PAGEdge.CIRCLE) & (pag[i, :] == PAGEdge.NONE) & (pag[:, i] == PAGEdge.NONE))[0]
                if len(Ks) > 0:
                    for k in Ks:
                        if debug:
                            if not (pag[i, j] == PAGEdge.ARROW and pag[k, j] == PAGEdge.CIRCLE and pag[i, k] == PAGEdge.NONE and pag[k, i] == PAGEdge.NONE):
                                raise ValueError('find error in rule1')
                        if detail_out: print(f'find_edge[i, k]:{find_edge[i, k]}, pag[i, k]:{pag[i, k]}, pag[k, i]:{pag[k, i]}, sepset[i, k]:{sepset[i, k]}')
                        if find_edge[i, k] and pag[i, k] == PAGEdge.NONE and pag[k, i] == PAGEdge.NONE and (j in sepset[i, k]):
                            pag[j, k] = PAGEdge.ARROW
                            pag[k, j] = PAGEdge.TAIL
                            if detail_out:
                                print(f'Rule 1 \n Orient: {i} *-> {j} o-* {k} as: {j} -> {k}')


        # --------------R2-------------------------------------------
        if rules[1]:
            #i *-o k
            inds = np.argwhere((pag == PAGEdge.CIRCLE) & (np.transpose(pag) != PAGEdge.NONE))
            for i, k in inds:
                Js = np.where(((pag[i, :] == PAGEdge.ARROW) & (pag[:, i] == PAGEdge.TAIL) & (pag[:, k] == PAGEdge.ARROW) & (pag[k, :] != PAGEdge.NONE)) | ((pag[i, :] == PAGEdge.ARROW) & (pag[:, i] != PAGEdge.NONE) & (pag[:, k] == PAGEdge.ARROW) & (pag[k, :] == PAGEdge.TAIL)))[0]
                if Js.size > 0:
                    pag[i, k] = PAGEdge.ARROW
                    if debug:
                        for j in Js:
                            if not ((pag[i, j] == PAGEdge.ARROW and pag[j, i] == PAGEdge.TAIL and pag[j, k] == PAGEdge.ARROW) or (pag[j, k] == PAGEdge.ARROW and pag[k, j] == PAGEdge.TAIL and pag[i, j] == PAGEdge.ARROW)):
                                raise ValueError('find error in rule2')
                    if detail_out:
                        print(f'Rule 2  \n Orient: {i} -> {Js} *-> {k} or {i} *-> {Js} -> {k} with {i} *-o {k} as: {i} *-> {k} ')

        # --------------R3-------------------------------------------
        if rules[2]:
            #j o-* l
            inds = np.argwhere((pag != PAGEdge.NONE) & (np.transpose(pag) == PAGEdge.CIRCLE))
            for j, l in inds:
                indIK = np.where((pag[j, :] != PAGEdge.NONE) & (pag[:, j] == PAGEdge.ARROW) & (pag[:, l] == PAGEdge.CIRCLE) & (pag[l, :] != PAGEdge.NONE))[0]
                if indIK.size >= 2:
                    counter = 0
                    while ( (counter < len(indIK) - 1) and (pag[l, j] != PAGEdge.ARROW)):
                        ii = counter + 1
                        while ((ii < len(indIK)) and (pag[l, j] != PAGEdge.ARROW) ):
                            if (pag[indIK[counter], indIK[ii]] == PAGEdge.NONE) and (pag[indIK[ii], indIK[counter]] == PAGEdge.NONE) and find_edge[indIK[ii], indIK[counter]] and find_edge[indIK[counter], indIK[ii]] and (l in sepset[indIK[ii], indIK[counter]]) :
                                pag[l, j] = PAGEdge.ARROW
                                if detail_out:
                                    print(f'Orienting edge {l}*-o{j}  to {l}*->{j} with rule 3')
                            ii += 1
                        counter += 1
        # --------------R4-------------------------------------------
        if rules[3]:
            # j o-* k
            inds = np.argwhere((pag != PAGEdge.NONE) & (np.transpose(pag) == PAGEdge.CIRCLE))
            for j, k in inds:
                #find all i -> k and i <-* j
                indI = np.where((pag[j, :] == PAGEdge.ARROW) & (pag[:, j] != PAGEdge.NONE) & (pag[k, :] == PAGEdge.TAIL) & (pag[:, k] == PAGEdge.ARROW))[0]

                while(indI.size > 0 and pag[k, j] == PAGEdge.CIRCLE) :
                    i = indI[0]
                    indI = np.delete(indI, 0)
                    done = False
                    while done == False and pag[i][j] != PAGEdge.NONE and pag[i][k] != PAGEdge.NONE and pag[j][k] != PAGEdge.NONE:
                        
                        if debug: print(f'finding minDiscPath')
                        md_path = minDiscPath(pag, i, j, k)
                        if debug: print(f'md_path:{md_path}')
                        if len(md_path) == 0:
                            done = True
                        else:
                            # a path exists
                            if find_edge[md_path[0],md_path[-1]]:
                                if detail_out:print(f"ma_path:{md_path}, i:{i}, j:{j}, k:{k}")
                                if detail_out:print(f"sepset[{md_path[0]}][{md_path[-1]}] :{sepset[md_path[0]][md_path[-1]]}")
                                if (j in sepset[md_path[0]][md_path[-1]]) or (j in sepset[md_path[-1]][md_path[0]]):
                                    pag[j][k] = PAGEdge.ARROW
                                    pag[k][j] = PAGEdge.TAIL
                                    if detail_out:print(f'Orienting edge {k} *-o{j} to  {k}-->{j} with rule 4')
                                else:
                                    pag[i][j] = pag[j][k] = pag[k][j] = PAGEdge.ARROW
                                    if detail_out:print(f'Orienting edge {i}<->{j}<->{k} with rule 4')
                            done = True
    return pag