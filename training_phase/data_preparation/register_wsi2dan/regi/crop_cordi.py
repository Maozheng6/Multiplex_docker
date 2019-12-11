wsi_size=[131092, 95179]
small_size=[955,673]
patches=[[553,42],[387,206],[303,254],[616,188]]
cord_on_wsi=[]
for x in patches:
    v1 = wsi_size[0]*x[0]/small_size[0]
    v2 = wsi_size[1]*x[1]/small_size[1]
    cord_on_wsi.append([int(v1),int(v2)])

print(cord_on_wsi)
