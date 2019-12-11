def resize_mask(mask,times):
    idx=np.nonzero(mask)
    #print(idx)
    idx2=[x/times for x in idx]
    idx=idx2
    mask2=np.zeros((int(mask.shape[0]/times),int(mask.shape[1]/times)))                                                                                            for i in range(len(idx[0])):
        row=int(idx[0][i])
                                                                                                                                                col=int(idx[1][i])                                                                                                                                                        mask2[row,col]=1
                                                                                                                                                                            return mask2

