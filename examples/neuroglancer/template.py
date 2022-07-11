import neuroglancer
import numpy as np
import imageio
import h5py

ip = 'localhost' #or public IP of the machine for sharable display
port = 9999 #change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()

input_file=INPUT

res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales=SCALE)

print('load im and gt segmentation')
if input_file.endswith('.tif'):
    im = imageio.volread(input_file)
else:
    with h5py.File(input_file, 'r') as fl:
        im = np.array(fl['main'])
print(im.shape)

def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
    return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)

with viewer.txn() as s:
    s.layers.append(name='gt',layer=ngLayer(im,res,tt='segmentation'))

    s.layers["segmentation"].layer.segments = {7693}

    s.cross_section_background_color = "#ffffff" # white
    s.layout = "3d"
    s.showDefaultAnnotations = False # turn off yellow bounding box
    s.show_axis_lines=False # turn off axes

print("Open in you browser:")
print(viewer)