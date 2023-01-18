from numba import njit, prange
import numpy as np


@njit(parallel=True)
def back_project(volume, projection, recon_position, tilt_angle):

    dims = volume.shape
    dims_proj = projection.shape

    center_recon = ((dims[0] // 2 + 1) - recon_position[0],
                    (dims[1] // 2 + 1) - recon_position[1],
                    (dims[2] // 2 + 1) - recon_position[2])
    center_proj = ((dims_proj[0] // 2 + 1), (dims_proj[1] // 2 + 1))

    tilt_angle_radians = np.deg2rad(tilt_angle)
    tr11 = np.cos(tilt_angle_radians) * np.cos(0) * np.cos(0) - np.sin(0) * np.sin(0)
    tr21 = np.cos(tilt_angle_radians) * np.sin(0) * np.cos(0) + np.cos(0) * np.sin(0)
    tr31 = -np.sin(tilt_angle_radians) * np.cos(0)
    tr12 = -np.cos(tilt_angle_radians) * np.cos(0) * np.sin(0) - np.sin(0) * np.cos(0)
    tr22 = -np.cos(tilt_angle_radians) * np.sin(0) * np.sin(0) + np.cos(0) * np.cos(0)
    tr32 = np.sin(tilt_angle_radians) * np.sin(0)

    for i in prange(dims[0]):  # prange for parallel
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = i - center_recon[0] + 1, j - center_recon[1] + 1, k - center_recon[2] + 1

                ix = tr11 * x + tr21 * y + tr31 * z + center_proj[0]
                iy = tr12 * x + tr22 * y + tr32 * z + center_proj[1]

                px, py = int(ix), int(iy)
                xoff, yoff = ix - px, iy - py

                if 1 <= px < dims_proj[0] and 1 <= py < dims_proj[1]:
                    v1 = projection[px - 1, py - 1] + (projection[px, py - 1] - projection[px - 1, py - 1]) * xoff
                    v2 = projection[px - 1, py] + (projection[px, py] - projection[px - 1, py]) * xoff
                    v3 = v1 + (v2 - v1) * yoff

                    volume[i, j, k] += v3


@njit(parallel=True)
def fill_values_real_spline_parallel(src, dst, mtx):
    # dont need to pass the shape (dims_src, dims) numba works with numpy
    # 4d matrix with rotation and translation allowed (shear
    dims_src = src.shape
    dims = dst.shape

    for dx in prange(dims[0]):
        for dy in range(dims[1]):
            for dz in range(dims[2]):

                x = float(mtx[0, 0] * dx + mtx[0, 1] * dy + mtx[0, 2] * dz + mtx[0, 3])
                y = float(mtx[1, 0] * dx + mtx[1, 1] * dy + mtx[1, 2] * dz + mtx[1, 3])
                z = float(mtx[2, 0] * dx + mtx[2, 1] * dy + mtx[2, 2] * dz + mtx[2, 3])

                sizex, sizey = dims_src[:2]

                is3D = (len(dims_src) == 3 and dims_src[2] > 1)

                if is3D:
                    sizez = dims_src[2]

                # v will store the interpolated value for voxel dx,dy,dz
                v = None

                # is the current position in the data or outside. return default value if outside
                tooClose4Spline3D = (is3D and (x < 2. or y < 2. or z < 2. or x > (sizex - 3.) or y > (sizey - 3.) or
                                               z > (sizez - 3.)))
                tooClose4Spline2D = (not is3D and (x < 2. or y < 2. or x > (sizex - 3.) or y > (sizey - 3.)))

                # If too close to edge, spline will not work, thus resorts to cubic interpolation. If that does not work
                # 0 is returned.

                if tooClose4Spline2D or tooClose4Spline3D:
                    # is the current position in the data or outside. return default value if outside
                    a = (is3D and (x < 1. or y < 1. or z < 1. or x > (sizex-2.) or y > (sizey-2.) or z > (sizez-2.)))
                    b = (not is3D and (x < 1. or y < 1. or x > (sizex-2.) or y > (sizey-2.)))

                    if a or b:
                        v = 0.  # set to 0, but could also do: linearInterpolation(src, x, y, z)

                    # run cubic interpolation
                    if v is None:
                        floorx = int(x)  # integer position along x
                        floory = int(y)  # integer position along y
                        floorz = int(z)  # integer position along z
                        xoffseth = x - float(floorx)  # floating point offset along x
                        yoffseth = y - float(floory)  # floating point offset along y
                        zoffseth = z - float(floorz)  # floating point offset along z

                        if xoffseth < 0.0000001 and yoffseth < 0.00000001 and zoffseth < 0.0000001:
                            v = src[floorx, floory, floorz]

                        if v is None:
                            px_pl_1 = floorx + 1  # all voxels plus 1 from x -> p3,p6,p9
                            px_mi_1 = floorx - 1  # all voxels minus 1 from x -> p1,p4,p7

                            py_pl_1 = floory + 1  # all voxels plus 1 from y -> p1,p2,p3
                            py_mi_1 = floory - 1  # all voxels minus 1 from x -> p7,p8,p9

                            lowerLayerBound = -1
                            upperLayerBound = 1
                            layerOffset = 1

                            layerValues = [0., 0., 0.]

                            if not is3D:
                                lowerLayerBound = 0
                                upperLayerBound = 0
                                layerOffset = 0

                            # interpolation values for each layer (z)
                            for zIteration in range(lowerLayerBound, upperLayerBound +1):
                                # current position in memory plus current z layer offset in voxels (of type T)
                                # first will be negative (-1), second 0 (same layer), third is 1, next layer

                                # load the pixel values
                                v1 = src[floorx-1][floory-1][floorz+zIteration] #*(src-1-this->stridey + zIteration*this->stridez); //one line up in y direction, one position back in x
                                v2 = src[floorx  ][floory-1][floorz+zIteration] #*(src  -this->stridey + zIteration*this->stridez); //one line up in y direction
                                v3 = src[floorx+1][floory-1][floorz+zIteration] #*(src+1-this->stridey + zIteration*this->stridez); //one line up in y direction, one position forward in x
                                v4 = src[floorx-1][floory  ][floorz+zIteration] #*(src-1 + zIteration*this->stridez); //same line in y
                                v5 = src[floorx  ][floory  ][floorz+zIteration] #*(src + zIteration*this->stridez); //...
                                v6 = src[floorx+1][floory  ][floorz+zIteration] #*(src+1 + zIteration*this->stridez);
                                v7 = src[floorx-1][floory+1][floorz+zIteration] #*(src-1+this->stridey + zIteration*this->stridez);
                                v8 = src[floorx  ][floory+1][floorz+zIteration] #*(src  +this->stridey + zIteration*this->stridez);
                                v9 = src[floorx+1][floory+1][floorz+zIteration] #*(src+1+this->stridey + zIteration*this->stridez);
                                #print("Value %f %f %f %f %f %f %f %f %f \n".format(v1,v2,v3,v4,v5,v6,v7,v8,v9))

                                # interpolate first row 1 2 3
                                line1 = CUB_INT(x,v1,px_mi_1,floorx,px_pl_1) #; //px1,px2,px3
                                line2 = CUB_INT(x,v2,floorx,px_pl_1,px_mi_1) #; //px2,px3,px1
                                line3 = CUB_INT(x,v3,px_pl_1,px_mi_1,floorx) #; //px3,px1,px2
                                # line1 = v1*(x-floorx)/(px_mi_1-floorx)*(x-px_pl_1)/(px_mi_1-px_pl_1)
                                # line2 = v2*(x-px_pl_1)/(floorx-px_pl_1)*(x-px_mi_1)/(floorx-px_mi_1)
                                # line3 = v3*(x-px_mi_1)/(px_pl_1-px_mi_1)*(x-floorx)/(px_pl_1-floorx)
                                #//store values into v1
                                #//printf("Line 1 %f %f %f\n",line1,line2,line3);
                                v1 = line1 + line2 + line3

                                # same for the next rows
                                line1 = CUB_INT(x,v4,px_mi_1,floorx,px_pl_1)
                                line2 = CUB_INT(x,v5,floorx,px_pl_1,px_mi_1)
                                line3 = CUB_INT(x,v6,px_pl_1,px_mi_1,floorx)
                                # line1 = v4*(x-floorx)/(px_mi_1-floorx)*(x-px_pl_1)/(px_mi_1-px_pl_1)
                                # line2 = v5*(x-px_pl_1)/(floorx-px_pl_1)*(x-px_mi_1)/(floorx-px_mi_1)
                                # line3 = v6*(x-px_mi_1)/(px_pl_1-px_mi_1)*(x-floorx)/(px_pl_1-floorx)
                                #//printf("Line 2 %f %f %f\n",line1,line2,line3);
                                v2 = line1 + line2 + line3

                                line1 = CUB_INT(x,v7,px_mi_1,floorx,px_pl_1)
                                line2 = CUB_INT(x,v8,floorx,px_pl_1,px_mi_1)
                                line3 = CUB_INT(x,v9,px_pl_1,px_mi_1,floorx)
                                # line1 = v7*(x-floorx)/(px_mi_1-floorx)*(x-px_pl_1)/(px_mi_1-px_pl_1)
                                # line2 = v8*(x-px_pl_1)/(floorx-px_pl_1)*(x-px_mi_1)/(floorx-px_mi_1)
                                # line3 = v9*(x-px_mi_1)/(px_pl_1-px_mi_1)*(x-floorx)/(px_pl_1-floorx)
                                v3 = line1 + line2 + line3

                                # interpolate col 2 5 8 in y direction
                                line1 = CUB_INT(y,v1,py_mi_1,floory,py_pl_1)
                                line2 = CUB_INT(y,v2,floory,py_pl_1,py_mi_1)
                                line3 = CUB_INT(y,v3,py_pl_1,py_mi_1,floory)
                                # line1 = v1*(y-floory)/(py_mi_1-floory)*(y-py_pl_1)/(py_mi_1-py_pl_1)
                                # line2 = v2*(y-py_pl_1)/(floory-py_pl_1)*(y-py_mi_1)/(floory-py_mi_1)
                                # line3 = v3*(y-py_mi_1)/(py_pl_1-py_mi_1)*(y-floory)/(py_pl_1-floory)
                                #//printf("Row 1%f %f %f\n",line1,line2,line3);

                                layerValues[zIteration + layerOffset] = line1 + line2 + line3

                            #//printf("Layer Values %f %f %f \n",layerValues[0],layerValues[1],layerValues[2]);
                            #//printf("FloorZ %d %d %d \n",floorz-1,floorz,floorz +1);

                            if (is3D):
                                line1 = CUB_INT(z,layerValues[0],(floorz-1),floorz,(floorz+1))
                                line2 = CUB_INT(z,layerValues[1],floorz,(floorz+1),(floorz-1))
                                line3 = CUB_INT(z,layerValues[2],(floorz+1),(floorz-1),floorz)
                                # line1 = layerValues[0]*(z-floorz)/((floorz-1)-floorz)*(z-(floorz+1))/((floorz-1)-(floorz+1))
                                # line2 = layerValues[1]*(z-(floorz+1))/(floorz-(floorz+1))*(z-(floorz-1))/(floorz-(floorz-1))
                                # line3 = layerValues[2]*(z-(floorz-1))/((floorz+1)-(floorz-1))*(z-floorz)/((floorz+1)-floorz)

                            #//printf("Layer 1 %f %f %f\n",line1,line2,line3);

                            v = (line1 + line2 + line3)

                # If v is not set, go for spline interpolation!
                if v is None:
                    floorx = int(x)  # integer position along x
                    floory = int(y)  # integer position along y
                    floorz = int(z)  # integer position along z
                    xoffseth = x - float(floorx)  # floating point offset along x
                    yoffseth = y - float(floory)  # floating point offset along y
                    zoffseth = z - float(floorz)  # floating point offset along z

                    # in case the point is so close, we just copy the value
                    if xoffseth < 0.0000001 and yoffseth < 0.00000001 and zoffseth < 0.0000001:
                        v = src[floorx][floory][floorz]

                    if v is None:
                        f1 = -0.1556
                        f2 = 0.3111
                        f3 = -0.0889
                        f4 = 0.0444

                        lowerLayerBound = -1
                        upperLayerBound = 2
                        layerOffset = 1
                        layerValues = [0., 0., 0., 0.]

                        if (not is3D):
                            lowerLayerBound = 0
                            upperLayerBound = 0
                            layerOffset = 0

                        # interpolation values for each layer (z)
                        for zIteration in range(lowerLayerBound, upperLayerBound + 1):
                            # load the pixel values
                            # comments are from C++ implementation
                            v1 = src[floorx - 1][floory - 1][floorz + zIteration]  # one line up in y direction, one position back in x
                            v2 = src[floorx][floory - 1][floorz + zIteration]  # *(src  -this->stridey + zIteration*this->stridez); #one line up in y direction
                            v3 = src[floorx + 1][floory - 1][floorz + zIteration]  # *(src+1-this->stridey + zIteration*this->stridez); #one line up in y direction, one position forward in x
                            v4 = src[floorx + 2][floory - 1][floorz + zIteration]  # *(src+2-this->stridey + zIteration*this->stridez);

                            v5 = src[floorx - 1][floory][floorz + zIteration]  # *(src-1 + zIteration*this->stridez); //same line in y
                            v6 = src[floorx][floory][floorz + zIteration]  # *(src + zIteration*this->stridez); //...
                            v7 = src[floorx + 1][floory][floorz + zIteration]  # *(src+1 + zIteration*this->stridez);
                            v8 = src[floorx + 2][floory][floorz + zIteration]  # *(src+2 + zIteration*this->stridez);

                            v9 = src[floorx - 1][floory + 1][floorz + zIteration]  # *(src-1+this->stridey + zIteration*this->stridez);
                            v10 = src[floorx][floory + 1][floorz + zIteration]  # *(src  +this->stridey + zIteration*this->stridez);
                            v11 = src[floorx + 1][floory + 1][floorz + zIteration]  # *(src+1+this->stridey + zIteration*this->stridez);
                            v12 = src[floorx + 2][floory + 1][floorz + zIteration]  # *(src+2+this->stridey + zIteration*this->stridez);

                            v13 = src[floorx - 1][floory + 2][floorz + zIteration]  # *(src-1+2*this->stridey + zIteration*this->stridez);
                            v14 = src[floorx][floory + 2][floorz + zIteration]  # *(src  +2*this->stridey + zIteration*this->stridez);
                            v15 = src[floorx + 1][floory + 2][floorz + zIteration]  # *(src+1+2*this->stridey + zIteration*this->stridez);
                            v16 = src[floorx + 2][floory + 2][floorz + zIteration]  # *(src+2+2*this->stridey + zIteration*this->stridez);

                            # calculate spline value for line 1 2 3 4 above pixel of interest
                            D1 = CSPL_INT(f1, f2, f3, f4, v1, v2, v3, v4)
                            D2 = CSPL_INT(f4, f3, f2, f1, v1, v2, v3, v4)
                            line1 = CSPL_CALC(v2, v3, D1, D2, xoffseth)
                            # D1 = 3*(f1*(v2-v1)+f2*(v3-v1)+f3*(v4-v2)+f4*(v4-v3))
                            # D2 = 3*(f4*(v2-v1)+f3*(v3-v1)+f2*(v4-v2)+f1*(v4-v3))
                            # line1 = v2 + D1*xoffseth +(3*(v3-v2)-2*D1-D2)*xoffseth*xoffseth+(2*(v2-v3)+D1+D2)*(xoffseth**3)

                            # calculate spline value for line 5 6 7 8 above pixel of interest
                            D1 = CSPL_INT(f1, f2, f3, f4, v5, v6, v7, v8)
                            D2 = CSPL_INT(f4, f3, f2, f1, v5, v6, v7, v8)
                            line2 = CSPL_CALC(v6, v7, D1, D2, xoffseth)
                            # D1 = 3*(f1*(v6-v5)+f2*(v7-v5)+f3*(v8-v6)+f4*(v8-v7))
                            # D2 = 3*(f4*(v6-v5)+f3*(v7-v5)+f2*(v8-v6)+f1*(v8-v7))
                            # line2 = v6 + D1*xoffseth +(3*(v7-v6)-2*D1-D2)*xoffseth*xoffseth+(2*(v6-v7)+D1+D2)*(xoffseth**3)

                            # calculate spline value for line 9 10 11 12 above pixel of interest
                            D1 = CSPL_INT(f1, f2, f3, f4, v9, v10, v11, v12)
                            D2 = CSPL_INT(f4, f3, f2, f1, v9, v10, v11, v12)
                            line3 = CSPL_CALC(v10, v11, D1, D2, xoffseth)
                            # D1 = 3*(f1*(v10-v9)+f2*(v11-v9)+f3*(v12-v10)+f4*(v12-v11))
                            # D2 = 3*(f4*(v10-v9)+f3*(v11-v9)+f2*(v12-v10)+f1*(v12-v11))
                            # line3 = v10 + D1*xoffseth +(3*(v11-v10)-2*D1-D2)*xoffseth*xoffseth+(2*(v10-v11)+D1+D2)*(xoffseth**3)

                            # calculate spline value for line 13 14 15 16 above pixel of interest
                            D1 = CSPL_INT(f1, f2, f3, f4, v13, v14, v15, v16)
                            D2 = CSPL_INT(f4, f3, f2, f1, v13, v14, v15, v16)
                            line4 = CSPL_CALC(v14, v15, D1, D2, xoffseth)
                            # D1 = 3*(f1*(v14-v13)+f2*(v15-v13)+f3*(v16-v14)+f4*(v16-v15))
                            # D2 = 3*(f4*(v14-v13)+f3*(v15-v13)+f2*(v16-v14)+f1*(v16-v15))
                            # line4 = v14 + D1*xoffseth +(3*(v15-v14)-2*D1-D2)*xoffseth*xoffseth+(2*(v14-v15)+D1+D2)*(xoffseth**3)

                            # finally, calculate spline into y direction and save into value[z]
                            D1 = CSPL_INT(f1, f2, f3, f4, line1, line2, line3, line4)
                            D2 = CSPL_INT(f4, f3, f2, f1, line1, line2, line3, line4)
                            # D1 = 3*(f1*(line2-line1)+f2*(line3-line1)+f3*(line4-line2)+f4*(line4-line3))
                            # D2 = 3*(f4*(line2-line1)+f3*(line3-line1)+f2*(line4-line2)+f1*(line4-line3))
                            layerValues[zIteration + layerOffset] = CSPL_CALC(line2, line3, D1, D2, yoffseth)
                            # layerValues[zIteration + layerOffset] = \
                            #     line2 + D1*yoffseth +(3*(line3-line2)-2*D1-D2)*(yoffseth**2)+(2*(line2-line3)+D1+D2)*(yoffseth**3)

                        if is3D:
                            # calculate spline value for z direction
                            # D1 = 3*(f1*(layerValues[1]-layerValues[0])+f2*(layerValues[2]-layerValues[0]) +
                            #         f3*(layerValues[3]-layerValues[1])+f4*(layerValues[3]-layerValues[2]))
                            D1 = CSPL_INT(f1, f2, f3, f4, layerValues[0], layerValues[1], layerValues[2],
                                          layerValues[3])

                            # D2 = 3*(f4*(layerValues[1]-layerValues[0])+f3*(layerValues[2]-layerValues[0]) +
                            #         f2*(layerValues[3]-layerValues[1])+f1*(layerValues[3]-layerValues[2]))
                            D2 = CSPL_INT(f4, f3, f2, f1, layerValues[0], layerValues[1], layerValues[2], layerValues[3])

                            # D1 = layerValues[1] + \
                            #      D1*zoffseth + \
                            #      (3*(layerValues[2]-layerValues[1])-2*D1-D2)*(zoffseth**2) + \
                            #      (2*(layerValues[1]-layerValues[2])+D1+D2)*(zoffseth**3)
                            v = CSPL_CALC(layerValues[1], layerValues[2], D1, D2, zoffseth)

                        else:
                            v = layerValues[0]

                dst[dx, dy, dz] = v


@njit
def CUB_INT(x,y,xj,x2,x3):
    return y*(x-x2)/(xj-x2)*(x-x3)/(xj-x3)


@njit
def CSPL_INT(f1,f2,f3,f4,a,b,c,d):
    return 3*(f1*(b-a)+f2*(c-a)+f3*(d-b)+f4*(d-c))


@njit
def CSPL_CALC(c2,c3,D1,D2,off):
    return c2+D1*off+((3)*(c3-c2)-(2)*D1-D2)*off*off+((2)*(c2-c3)+D1+D2)*off*off*off