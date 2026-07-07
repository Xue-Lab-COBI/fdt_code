# Created by Renzhi He, COBI, UCDavis, 2023
#8/26/24 precomputed kernel

"""
EN: Core optics module for FDT (refractive-index tomography). It defines the physical
forward model used during reconstruction: the 3D phase object container (PhaseObject3D),
Fourier-grid / pupil / MTF generators, aperture-based operators (Aberration, Defocus),
and multi-slice beam-propagation scattering models (MultiTransmittance,
MultiPhaseContrast) that propagate a spherical illumination wave through the volume
slice by slice with angular-spectrum kernels. TomographySolver ties these pieces
together to predict camera-plane intensities from the trainable RI volume.

中文：FDT（折射率层析成像）的核心光学模块。它定义了重建过程中使用的物理正向模型：
三维相位物体容器（PhaseObject3D）、傅里叶网格/光瞳/MTF 生成函数、基于孔径的算子
（Aberration 像差、Defocus 离焦），以及多层束传播散射模型（MultiTransmittance、
MultiPhaseContrast）——后者利用角谱传播核将球面照明波逐层传播穿过体积。
TomographySolver 将这些组件组合起来，由可训练的折射率体积预测相机平面的强度图像。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

#import contexttimer

# EN: Global numeric precision switch: bit=32 selects float32/complex64, otherwise float64/complex128,
# for both NumPy and PyTorch, so the whole pipeline stays in a consistent dtype.
# 中文：全局数值精度开关：bit=32 时选用 float32/complex64，否则选用 float64/complex128，
# 同时作用于 NumPy 与 PyTorch，保证整个流程使用一致的数据类型。
bit = 32

np_float_datatype = np.float32 if bit == 32 else np.float64
np_complex_datatype = np.complex64 if bit == 32 else np.complex128

if bit == 32:
    torch_float_datatype = torch.float32
    torch_complex_datatype = torch.complex64
else:
    torch_float_datatype = torch.float64
    torch_complex_datatype = torch.complex128

# EN: Container for the 3D sample being reconstructed; stores the complex RI volume,
# voxel geometry, and conversions between RI / transmittance / phase-contrast / scattering potential.
# 中文：待重建三维样品的容器；保存复折射率体积、体素几何参数，并提供折射率 /
# 透过率 / 相位衬度 / 散射势之间的相互转换。
class PhaseObject3D:
    """
    Class created for 3D objects.
    Depending on the scattering model, one of the following quantities will be used:
    - Refractive index (RI)
    - Transmittance function (Trans)
    - PhaseContrast
    - Scattering potential (V)

    shape:              shape of object to be reconstructed in (x,y,z), tuple
    voxel_size:         size of each voxel in (x,y,z), tuple
    RI_obj:             refractive index of object(Optional)
    RI:                 background refractive index (Optional)
    slice_separation:   For multislice algorithms, how far apart are slices separated, array (Optional)
    """
    def __init__(self, shape, voxel_size, RI_obj = None, RI = 1.0, slice_separation = None,free_space=76,args=None):
        assert len(shape) == 3, "shape should be 3 dimensional!"
        self.shape           = shape
        self.RI_obj          = RI * torch.ones(shape, dtype = torch_complex_datatype) if RI_obj is None else RI_obj.to(dtype=torch_complex_datatype)
        self.RI              = RI
        self.pixel_size      = voxel_size[0]
        self.pixel_size_z    = voxel_size[2]
        self.free_space      = free_space
        # self.back_to_center_ratio=args.back_to_center_ratio
        self.args=args
        #for continuous slices
        self.slice_separation = self.pixel_size_z * torch.ones((RI_obj.shape[2]-1,), dtype = torch_float_datatype)
        a=1
    # EN: The four converters below map between equivalent object representations:
    # RI -> per-slice transmittance, RI -> phase contrast (RI - background), RI <-> scattering potential V.
    # 中文：下面四个转换函数在等价的物体表示之间映射：
    # 折射率 -> 各切片透过率、折射率 -> 相位衬度（RI 减背景）、折射率 <-> 散射势 V。
    def convertRItoTrans(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.trans_obj       = np.exp(1.0j*k0*(self.RI_obj - self.RI)*self.pixel_size_z)

    def convertRItoPhaseContrast(self):
        self.contrast_obj    = self.RI_obj - self.RI

    def convertRItoV(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.V_obj           = k0**2 * (self.RI**2 - self.RI_obj**2)

    def convertVtoRI(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        B                    = -1.0 * (self.RI**2 - self.V_obj.real/k0**2)
        C                    = -1.0 * (-1.0 * self.V_obj.imag/k0**2/2.0)**2
        RI_obj_real          = ((-1.0 * B + (B**2-4.0*C)**0.5)/2.0)**0.5
        RI_obj_imag          = -0.5 * self.V_obj.imag/k0**2/RI_obj_real
        self.RI_obj          = RI_obj_real + 1.0j * RI_obj_imag

class STEFunction(torch.autograd.Function):
    """
    EN: Straight-Through Estimator (STE): rounds in the forward pass but passes the
    gradient through unchanged in the backward pass. Used to snap illumination source
    positions onto the discrete pixel grid while keeping them differentiable for
    self-calibration.

    中文：直通估计器（STE）：前向传播时取整，反向传播时梯度原样通过。用于把照明
    光源位置对齐到离散像素网格，同时保持可微，从而支持光源位置的自校准。
    """
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        xlin       - 1D Fourier grid

    """
    xlin = (torch.arange(size) - size//2) * dx
    if flag_shift:
        xlin = torch.roll(xlin, -1 * size//2)
    return xlin.to(torch_complex_datatype)
def genPupil(shape, pixel_size, NA, wavelength):
    """
    EN: Generate a binary circular pupil in the Fourier domain with cutoff NA/wavelength,
    additionally masked by the maximum representable spatial frequency of the grid.

    中文：在傅里叶域生成二值圆形光瞳，截止频率为 NA/波长，并用网格可表示的最大
    空间频率作额外掩膜。
    """
    assert len(shape) == 2, "pupil should be two dimensional!"
    # print("shape to gen pupil:", shape)
    fxlin = genGrid(shape[1], 1 / pixel_size / shape[1], flag_shift=True)
    # print("fxlin:", fxlin)
    fylin = genGrid(shape[0], 1 / pixel_size / shape[0], flag_shift=True)
    # print("fxlin shape:", fxlin.shape, ", fylin shape:", fylin.shape)
    fxlin = fxlin.unsqueeze(0).repeat(shape[0], 1)
    fylin = fylin.unsqueeze(1).repeat(1, shape[1])

    pupil_radius = NA / wavelength
    # print("fxlin shape:", fxlin.shape, ", fylin shape:", fylin.shape, ", pupil radius:", pupil_radius)
    pupil = (abs(fxlin ** 2 + fylin ** 2) <= abs(pupil_radius ** 2)).to(torch_complex_datatype)
    pupil_mask = (abs(fxlin ** 2 + fylin.real ** 2) <= torch.max(abs(fxlin)) ** 2).to(torch_complex_datatype)
    pupil = pupil * pupil_mask
    return pupil
def genMTF(shape, pixel_size, NA, wavelength, pupil_sigma, pupil_power):
    """
    EN: Generate a smooth Butterworth-style low-pass transfer function (MTF) in the
    Fourier domain, controlled by cutoff pupil_sigma and roll-off order pupil_power.

    中文：在傅里叶域生成平滑的巴特沃斯型低通传递函数（MTF），截止频率由
    pupil_sigma 控制，滚降陡峭度由 pupil_power 控制。
    """
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin = genGrid(shape[1], 1 / pixel_size / shape[1], flag_shift=True)
    #     print(fxlin)
    fylin = genGrid(shape[0], 1 / pixel_size / shape[0], flag_shift=True)
    fxlin = fxlin.unsqueeze(0).repeat(shape[0], 1)
    fylin = fylin.unsqueeze(1).repeat(1, shape[1])

    #     pupil_radius = NA/wavelength
    #     pupil        = (fxlin**2 + fylin**2 <= pupil_radius**2).to(torch_complex_datatype)
    #     pupil_mask   = (fxlin**2 + fylin**2 <= torch.max(fxlin)**2).to(torch_complex_datatype)
    q2 = fxlin ** 2 + fylin ** 2
    pupil_mask = (1 / (1 + (q2 / (pupil_sigma ** 2)) ** pupil_power)).to(torch_complex_datatype)  # .numpy()
    #     pupil        = pupil * pupil_mask
    return pupil_mask
class Aperture:
    """
    Class for optical aperture (general)
    """
    def __init__(self, shape, pixel_size, na, pad = True, pad_size = None, **kwargs):
        """
        shape:          shape of object (y,x,z)
        pixel_size:     pixel size of the system
        na:             NA of the system
        pad:            boolean variable to pad the reconstruction
        pad_size:       if pad is true, default pad_size is shape//2. Takes a tuple, pad size in dimensions (y, x)
        """
        self.shape          = shape
        self.pixel_size     = pixel_size
        self.na             = na
        self.pad            = pad
        # EN: When padding is enabled, precompute crop slices that recover the original
        # (unpadded) field of view from the padded reconstruction; default pad is shape//4 per side.
        # 中文：启用填充时，预先计算裁剪切片，以便从填充后的重建结果中截取原始（未填充）
        # 视场；默认每侧填充 shape//4。
        if self.pad:
            self.pad_size       = pad_size
            if self.pad_size == None:
                self.pad_size   = (self.shape[0]//4, self.shape[1]//4)
            self.row_crop   = slice(self.pad_size[0], self.shape[0] - self.pad_size[0])
            self.col_crop   = slice(self.pad_size[1], self.shape[1] - self.pad_size[1])
        else:
            self.row_crop   = slice(0, self.shape[0])
            self.col_crop   = slice(0, self.shape[1])

    def forward(self):
        pass

    def adjoint(self):
        pass

class Aberration(Aperture):
    """
    Aberration class used for pupil recovery
    """
    def __init__(self, shape, pixel_size, wavelength, na, pad = True, flag_update = False, pupil_step_size = 1.0, update_method = "gradient", **kwargs):
        """
        Initialization of the class

        wavelength:         wavelength of light
        flag_update:        boolean variable to update pupil
        pupil_step_size:    if update the pupil, what is a step size for gradient method
        update_method:      can be "gradient" or "GaussNewton"
        """
        super().__init__(shape, pixel_size, na, pad, **kwargs)
        self.pupil            = genPupil(self.shape, self.pixel_size, self.na, wavelength)
        #print("pupil:",self.pupil)
        self.mtf              = genMTF([1200, 1200], self.pixel_size, self.na, wavelength, pupil_sigma=0.3, pupil_power = 4)
        self.wavelength       = wavelength
        self.pupil_support    = self.pupil.clone()
        self.pupil_step_size  = pupil_step_size
        self.flag_update      = flag_update
        self.update_method    = update_method

    def forward(self, field):
        """Apply pupil"""
        # EN: Filter the field through the pupil in the Fourier domain (FFT -> multiply
        # by pupil -> IFFT), then crop back to the unpadded region.
        # 中文：在傅里叶域用光瞳对场进行滤波（FFT -> 乘以光瞳 -> 逆 FFT），
        # 然后裁剪回未填充的区域。
        self.field_f          = torch.fft.fft2(field)
        if self.update_method == "GaussNewton":
            #not GaussNewton
            self.approx_hessian[:, :] += self.field_f*torch.conj(self.field_f)
        field_pupil           = torch.fft.ifft2(self.pupil * self.field_f)[self.row_crop, self.col_crop]
        # plt.figure()
        # plt.imshow(torch.abs(self.pupil))
        # plt.figure()
        # plt.imshow(torch.abs(self.field_f))
        # plt.figure()
        # plt.imshow(torch.angle(self.field_f))
        # plt.pause(0.05)
        return field_pupil

    # EN: Adjoint of the pupil operator (conjugate pupil in Fourier space); optionally
    # accumulates a gradient to update/recover the pupil after a full set of measurements.
    # 中文：光瞳算子的伴随算子（在傅里叶域乘以光瞳的共轭）；可选地累积梯度，
    # 在一整组测量结束后更新/恢复光瞳。
    def adjoint(self, field):
        """Adjoint operator for pupil (and estimate pupil if selected)"""
        field_f               = torch.zeros((self.shape[0], self.shape[1]), dtype=torch_complex_datatype)
        field_f[self.row_crop, self.col_crop] = field
        torch.fft.fft2(field_f, out=field_f)
        field_pupil_adj       = torch.fft.ifft2(torch.conj(self.pupil) * field_f)
        #Pupil recovery
        if self.flag_update:
            #not this case
            self.pupil_gradient[:, :] += torch.conj(self.field_f) * field_f
            self.measure_count        += 1
            if self.measure_count == self.measurement_num:
                self._update()
                self.measure_count     = 0
        return field_pupil_adj

    def _update(self):
        """function to recover pupil"""
        if self.update_method == "gradient":
            self.pupil[:, :] -= self.pupil_step_size * self.pupil_gradient * self.pupil_support
        elif self.update_method == "GaussNewton":
            self.pupil[:, :] -= self.pupil_step_size * 0.25 / (self.approx_hessian + 1e-8) * self.pupil_gradient * self.pupil_support
            self.approx_hessian[:, :] = 0.0
        else:
            print("there is no update_method \"%s\"!" %(self.update_method))
            raise
        self.pupil_gradient[:, :] = 0.0

class Defocus(Aperture):
    """Defocus subclass for tomography"""
    def __init__(self, shape, pixel_size, wavelength, na, RI_measure = 1.0, pad = True, **kwargs):
        """
        Initialization of the class

        RI_measure: refractive index on the detection side (example: oil immersion objectives)
        """
        super().__init__(shape, pixel_size, na, pad, **kwargs)

        fxlin                   = genGrid(self.shape[1], 1.0/self.pixel_size/self.shape[1], flag_shift = True)
        fylin                   = genGrid(self.shape[0], 1.0/self.pixel_size/self.shape[0], flag_shift = True)
        fxlin                   = fxlin.unsqueeze(0).repeat(self.shape[0], 1)
        fylin                   = fylin.unsqueeze(1).repeat(1, self.shape[1])
        self.pupil              = genPupil(self.shape, self.pixel_size, self.na, wavelength)
        # EN: Precompute the angular-spectrum phase factor i*2*pi*fz (with fz = sqrt((n/lambda)^2 - fx^2 - fy^2)),
        # band-limited by the pupil and a frequency stop; exp(phase*z) later propagates the field by distance z.
        # 中文：预先计算角谱相位因子 i*2*pi*fz（其中 fz = sqrt((n/λ)^2 - fx^2 - fy^2)），
        # 并用光瞳和频率截止掩膜进行带限；之后 exp(phase*z) 即可把场传播距离 z。
        self.pupilstop          = (abs(fxlin**2 + fylin**2) <= torch.max(abs(fxlin))**2).to(torch_complex_datatype)
        self.prop_kernel_phase  = 1.0j*2.0*np.pi*self.pupil*self.pupilstop*((RI_measure/wavelength)**2 - fxlin*torch.conj(fxlin) - fylin*torch.conj(fylin)+1e-8)**0.5
        # self
#         plt.figure()
#         plt.imshow(np.imag(self.prop_kernel_phase))
#         plt.pause(0.05)


    def forward(self, field, propagation_distances):
        """defocus with angular spectrum"""
        # EN: Apply the pupil in Fourier space, then defocus the field to each requested plane by
        # multiplying with the propagation kernel; only a single defocus distance is supported here.
        # 中文：先在傅里叶域施加光瞳，再通过乘以传播核将场离焦到每个指定平面；
        # 此处仅支持单一离焦距离。
        # print(field.shape,self.pupil.shape)
        # print("field1:",field)
        field_defocus           = self.pupil * torch.fft.fft2(field)
        field_defocus1 = field_defocus
        # print(field_defocus.shape)
        # print(len(propagation_distances))
        # print("field_defocus1:",field_defocus)

        field_defocus           = field_defocus.unsqueeze(2).repeat(1, 1, len(propagation_distances))

        for z_idx, propagation_distance in enumerate(propagation_distances):
            # print("propagation_distance:", propagation_distances[z_idx])
            propagation_kernel          = torch.exp(self.prop_kernel_phase*propagation_distances[z_idx])
            # print("propagation_kernel:", propagation_kernel)
            # print(propagation_kernel.shape, field_defocus.shape)
            # if(z_idx==0):
            #     print("before:",field_defocus[:, :, z_idx])
            field_defocus[:, :, z_idx] *= propagation_kernel
            # if(z_idx==0):
            #     print("after:",field_defocus[:, :, z_idx])
        # print("field_defocus2:",field_defocus[:,:,0])

        assert(len(propagation_distances)==1)
        field_defocus=torch.fft.ifft2(field_defocus[:,:,0])
        # print("field2:",field_defocus)
        return field_defocus[self.row_crop, self.col_crop]

    # EN: Adjoint of the defocus operator: back-propagate residuals from each defocus
    # plane with the conjugate kernel and sum, then apply the pupil.
    # 中文：离焦算子的伴随算子：用共轭传播核把各离焦平面的残差反向传播并求和，
    # 再施加光瞳。
    def adjoint(self, residual, propagation_distances):
        """adjoint operator for defocus with angular spectrum"""
        field_focus             = torch.zeros((self.shape[0], self.shape[1]), dtype = torch_complex_datatype)
        field_pad               = torch.zeros((self.shape[0], self.shape[1]), dtype = torch_complex_datatype)
        for z_idx, propagation_distance in enumerate(propagation_distances):
            #z_idx=0, propagation_distane=0
            field_pad[self.row_crop, self.col_crop] = residual[:, :, z_idx]
            propagation_kernel_conj  = torch.exp(-1.0*self.prop_kernel_phase*propagation_distances[z_idx])
            field_focus[:, :]       += propagation_kernel_conj * torch.fft.fft2(field_pad)
        field_focus            *= self.pupil
        torch.fft.ifft2(field_focus, out=field_focus)
        return field_focus

# EN: Base class shared by all scattering models: stores volume geometry, builds real/frequency
# grids, generates plane-wave or spherical illumination, and handles z-binning of the object.
# 中文：所有散射模型共享的基类：保存体积几何信息、构建实空间/频率网格、生成平面波
# 或球面波照明，并负责物体在 z 方向上的分箱（binning）。
class ScatteringModels:
    """
    Core of the scattering models
    """
    def __init__(self, phase_obj_3d, wavelength, slice_binning_factor = 1, pad = True, **kwargs):
        """
        Initialization of the class

        phase_obj_3d:           object of the class PhaseObject3D
        wavelength:             wavelength of the light
        slice_binning_factor:   the object is compress in z-direction by this factor
        pad:                    boolean variable to pad the reconstruction.
                                If true, reconstruction size is padded
                                If false, reconstruction size is the same as measurement size
        """
        self.pad                  = pad
        self.slice_binning_factor = slice_binning_factor
        self._shape_full          = phase_obj_3d.shape
        self.shape                = phase_obj_3d.shape[0:2] + (int(np.ceil(phase_obj_3d.shape[2]/self.slice_binning_factor)),)
        self.RI                   = phase_obj_3d.RI
        self.wavelength           = wavelength
        self.pixel_size           = phase_obj_3d.pixel_size
        self.pixel_size_z         = phase_obj_3d.pixel_size_z * self.slice_binning_factor
        self.back_scatter         = False
        #Broadcasts b to a, size(a) > size(b)
        self.assign_broadcast     = lambda a, b : a - a + b
        #0130katz
        self.free_space=phase_obj_3d.free_space
        # self.back_to_center_ratio=phase_obj_3d.back_to_center_ratio
        self.args=phase_obj_3d.args

    def _genRealGrid(self, fftshift = False):
        xlin                = genGrid(self.shape[1], self.pixel_size, flag_shift = fftshift)
        ylin                = genGrid(self.shape[0], self.pixel_size, flag_shift = fftshift)
        xlin                = xlin.unsqueeze(0).repeat(self.shape[0], 1)
        ylin                = ylin.unsqueeze(1).repeat(1, self.shape[1])
        return xlin, ylin

    def _genFrequencyGrid(self):
        fxlin               = genGrid(self.shape[1], 1.0/self.pixel_size/self.shape[1], flag_shift = True)
        fylin               = genGrid(self.shape[0], 1.0/self.pixel_size/self.shape[0], flag_shift = True)
        fxlin               = fxlin.unsqueeze(0).repeat(self.shape[0], 1)
        fylin               = fylin.unsqueeze(1).repeat(1, self.shape[1])
        return fxlin, fylin

    # EN: Generate a tilted plane-wave illumination for the given spatial-frequency
    # coordinates (snapped onto the grid) plus its axial frequency fz.
    # 中文：根据给定的空间频率坐标（先对齐到网格）生成倾斜平面波照明，并返回
    # 其轴向频率 fz。
    def _genIllumination(self, fx_illu, fy_illu):
        fx_illu, fy_illu    = self._setIlluminationOnGrid(fx_illu, fy_illu)
        xlin, ylin          = self._genRealGrid()
        fz_illu             = ((self.RI/self.wavelength)**2 - fx_illu**2 - fy_illu**2)**0.5
        illumination_xy     = torch.exp(1.0j*2.0*np.pi*(fx_illu*xlin + fy_illu*ylin))
        return illumination_xy, fx_illu, fy_illu, fz_illu

    # EN: Generate a diverging spherical-wave illumination centered at a lateral source
    # position (fx_illu, fy_illu are interpreted as source shifts, not angles); the epsilon
    # avoids a NaN/singularity when the source lands exactly on a grid point.
    # 中文：生成以横向光源位置为中心的发散球面波照明（fx_illu、fy_illu 表示光源
    # 偏移量而非角度）；epsilon 用于避免光源恰好落在网格点上时的 NaN/奇异性。
    def _genSphericalIllumination(self, fx_illu, fy_illu):
        fx_illu, fy_illu    = self._setIlluminationOnGrid(fx_illu, fy_illu)
        xlin, ylin          = self._genRealGrid()
        fz_illu             = self.RI/self.wavelength
        #instead of angle, spherical wave should input shift location
        #remove nan value, only appear when the focus in the center
        Amplitude_Source    = 10

        epsilon = 1e-5
        distance_squared = (((xlin - fx_illu * self.shape[0]) * torch.conj(xlin - fx_illu * self.shape[0])) + \
                            ((ylin - fy_illu * self.shape[1]) * torch.conj(ylin - fy_illu * self.shape[1])))
        distance = torch.sqrt(distance_squared + epsilon)
        illumination_xy = Amplitude_Source*torch.exp(1.0j * 2.0 * np.pi / self.wavelength * distance) / distance
        # illumination_xy     = torch.exp(1.0j*2.0*np.pi/self.wavelength*\
        #                              (((xlin-fx_illu*self.sbinghape[0])*torch.conj(xlin-fx_illu*self.shape[0]))+\
        #                                ((ylin-fy_illu*self.shape[1])*torch.conj(ylin-fy_illu*self.shape[1]))+epsilon)**0.5)/\
        #                               (((xlin-fx_illu*self.shape[0])*torch.conj(xlin-fx_illu*self.shape[0]))+\
        #                                ((ylin-fy_illu*self.shape[1])*torch.conj(ylin-fy_illu*self.shape[1]))+epsilon)**0.5
        indCenterCol        = illumination_xy.shape[0]//2+fx_illu*self.shape[0]/self.pixel_size
        indCenterRow        = illumination_xy.shape[1]//2+fy_illu*self.shape[1]/self.pixel_size
        #illumination_xy[int(indCenterRow),int(indCenterCol)] #= 0.00000001
        #illumination_xy[int(indCenterRow),int(indCenterCol)] += 10 + 0.0j
        #illumination_xy[int(indCenterRow),int(indCenterCol)] += 0.000001 + 0.0j
        #illumination_xy=illumination_xy/1000

        return illumination_xy, fx_illu, fy_illu, fz_illu

    # EN: Snap the (trainable) illumination coordinates onto the discrete pixel grid using the
    # straight-through estimator, so source self-calibration remains differentiable.
    # 中文：使用直通估计器（STE）将（可训练的）照明坐标对齐到离散像素网格，
    # 使光源自校准过程保持可微。
    def _setIlluminationOnGrid(self, fx_illu, fy_illu):
        #dfx                 = 1.0/self.pixel_size/self.shape[1]
        #dfy                 = 1.0/self.pixel_size/self.shape[0]
        #fx_illu_on_grid     = np.round(fx_illu/dfx)*dfx
        #fy_illu_on_grid     = np.round(fy_illu/dfy)*dfy
        ## torch.round 1122
        # fx_illu_on_grid = torch.round(fx_illu*self.shape[1]/self.pixel_size)*self.pixel_size/self.shape[1] #modified by Yi
        # fy_illu_on_grid = torch.round(fy_illu*self.shape[0]/self.pixel_size)*self.pixel_size/self.shape[0]
        fx_illu_on_grid = STEFunction.apply(fx_illu * self.shape[1] / self.pixel_size) * self.pixel_size / self.shape[1]
        fy_illu_on_grid = STEFunction.apply(fy_illu * self.shape[0] / self.pixel_size) * self.pixel_size / self.shape[0]
        # fx_illu_on_grid = STEFunction.apply(fx_illu_on_grid)
        # fy_illu_on_grid = STEFunction.apply(fy_illu_on_grid)
        return fx_illu_on_grid, fy_illu_on_grid


    # EN: Compress (forward) or expand (adjoint) the object along z by slice_binning_factor;
    # with factor 1 (the common case) the object is returned untouched.
    # 中文：按 slice_binning_factor 在 z 方向压缩（正向）或展开（伴随）物体；
    # 当因子为 1（常见情况）时直接原样返回。
    def _binObject(self, obj, adjoint = False):
        """
        function to bin the object by factor of slice_binning_factor
        """
        if self.slice_binning_factor == 1:
            #execute this one, in both forward and reconstruction
            return obj
        if adjoint:
            obj_out = torch.zeros((self._shape_full[0], self._shape_full[1], self._shape_full[2]), dtype = torch_complex_datatype)
            for idx in range((self.shape[2]-1)*self.slice_binning_factor, -1, -self.slice_binning_factor):
                idx_slice = slice(idx, np.min([obj_out.shape[2],idx+self.slice_binning_factor]))
                obj_out[:, :, idx_slice] = torch.broadcast_to(obj[:, :, idx // self.slice_binning_factor], obj_out[:, :, idx_slice].shape)
        else:
            obj_out = torch.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype = torch_complex_datatype)
            for idx in range(0, obj.shape[2], self.slice_binning_factor):
                idx_slice = slice(idx, np.min([obj.shape[2], idx+self.slice_binning_factor]))
                obj_out[:,:,idx//self.slice_binning_factor] = torch.sum(obj[:,:,idx_slice], dim=2)
        return obj_out

    def forward(self, x_obj, fx_illu, fy_illu):
        pass

    def adjoint(self, residual, cache):
        pass
class MultiTransmittance(ScatteringModels):
    """
    MultiTransmittance scattering model. This class also serves as a parent class for all multi-slice scattering methods
    """

    def __init__(self, phase_obj_3d, wavelength, **kwargs):
        # EN: Aggregate slice separations after binning, precompute the band-limited angular-spectrum
        # propagation kernel for one slice spacing (self.kernel), and decide where the focal plane sits
        # (volume center for uniform spacing).
        # 中文：按分箱因子合并切片间距，预先计算单个切片间距对应的带限角谱传播核
        # （self.kernel），并确定焦平面位置（切片均匀分布时位于体积中心）。
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        self.slice_separation = [sum(phase_obj_3d.slice_separation[x:x + self.slice_binning_factor]) \
                                 for x in range(0, len(phase_obj_3d.slice_separation), self.slice_binning_factor)]
        sample_thickness = torch.sum(torch.tensor(self.slice_separation))
        total_prop_distance = torch.sum(torch.tensor(self.slice_separation[0:self.shape[2] - 1]))
        # self.distance_end_to_center = sample_thickness * self.back_to_center_ratio
        self.distance_end_to_center = 400
        # self.distance_end_to_center = total_prop_distance - sample_thickness * self.back_to_center_ratio
        self.distance_end_to_begin = total_prop_distance
        fxlin, fylin = self._genFrequencyGrid()
        # fzlin = ((self.RI / self.wavelength) ** 2 - fxlin * torch.conj(fxlin) - fylin * torch.conj(fylin)) ** 0.5
        # 1/25/24 Renzhi
        temp = (self.RI / self.wavelength) ** 2 - fxlin * torch.conj(fxlin) - fylin * torch.conj(fylin)
        temp_clamped = temp+1e-5  # make sure it is above 0
        fzlin = torch.sqrt(temp_clamped)

        self.pupilstop = (abs(fxlin ** 2 + fylin ** 2) <= torch.max(abs(fxlin)) ** 2).to(torch_complex_datatype)
        self.prop_kernel_phase = 1.0j * 2.0 * np.pi * self.pupilstop * fzlin
        self.kernel = torch.exp(self.prop_kernel_phase * self.slice_separation[0])
        if torch.abs(torch.mean(
                torch.stack(self.slice_separation)) - self.pixel_size_z) < 1e-6 or self.slice_binning_factor > 1:
            self.focus_at_begin = False
            self.focus_at_center = True
            self.initial_z_position = torch.tensor(-1 * (self.shape[2] // 2) * self.pixel_size_z)
        else:
            self.focus_at_begin = False
            self.focus_at_center = False
            self.initial_z_position = torch.tensor(0.0)

    # EN: Multi-slice forward model: generate spherical illumination, then alternately multiply
    # by each slice's transmittance and propagate to the next slice; finally back-propagate to
    # the focal plane. Conjugate per-layer fields are cached for the adjoint pass.
    # 中文：多层正向模型：先生成球面波照明，然后交替执行“乘以该层透过率 -> 角谱传播
    # 到下一层”；最后反向传播到焦平面。每层场的共轭被缓存以供伴随运算使用。
    def forward(self, trans_obj, fx_illu, fy_illu):
        if self.slice_binning_factor > 1:
            print("Slicing is not implemented for MultiTransmittance algorithm!")
            raise

        # compute illumination
        # field, fx_illu, fy_illu, fz_illu   = self._genIllumination(fx_illu, fy_illu)
        field, fx_illu, fy_illu, fz_illu = self._genSphericalIllumination(fx_illu, fy_illu)
        field[:, :] *= np.exp(1.0j * 2.0 * np.pi * fz_illu * self.initial_z_position)

        # multi-slice transmittance forward propagation
        field_layer_conj = torch.zeros((trans_obj.shape[0], trans_obj.shape[1], trans_obj.shape[2]),
                                       dtype=torch_complex_datatype)
        if (type(trans_obj).__module__ == np.__name__):
            trans_obj_torch = torch.from_numpy(trans_obj)
            flag_gpu_inout = False
        else:
            trans_obj_torch = trans_obj
            flag_gpu_inout = True

        for layer in range(self.shape[2]):
            field_layer_conj[:, :, layer] = torch.conj(field)
            field[:, :] *= trans_obj_torch[:, :, layer]
            if layer < self.shape[2] - 1:
                field = self._propagationInplace(field, self.slice_separation[layer])

        # store intermediate variables for adjoint operation
        cache = (trans_obj_torch, field_layer_conj, flag_gpu_inout)

        if self.focus_at_begin:
            # only for USAF target
            field = self._propagationInplace(field, self.distance_end_to_begin, adjoint=True)
        if self.focus_at_center:
            # propagate to volume center
            field = self._propagationInplace(field, self.distance_end_to_center, adjoint=True)
        return {'forward_scattered_field': field, 'cache': cache}

    # EN: Adjoint (error back-propagation) of the multi-slice model: walk slices in reverse,
    # forming the per-slice gradient from the residual and the cached conjugate fields.
    # 中文：多层模型的伴随运算（误差反向传播）：逆序遍历各切片，由残差与缓存的
    # 共轭场逐层构造梯度。
    def adjoint(self, residual, cache):
        trans_obj_torch, field_layer_conj_or_grad, flag_gpu_inout = cache

        # back-propagte to volume center
        field_bp = residual

        if self.focus_at_begin:
            # only for USAF target
            field = self._propagationInplace(field, self.distance_end_to_begin)
        if self.focus_at_center:
            # propagate to the center
            field_bp = self._propagationInplace(field_bp, self.distance_end_to_center)

        # multi-slice transmittance backward
        for layer in range(self.shape[2] - 1, -1, -1):
            field_layer_conj_or_grad[:, :, layer] = field_bp * field_layer_conj_or_grad[:, :, layer]
            if layer > 0:
                field_bp[:, :] *= torch.conj(trans_obj_torch[:, :, layer])
                field_bp = self._propagationInplace(field_bp, self.slice_separation[layer - 1], adjoint=True)
        if flag_gpu_inout:
            return {'gradient': field_layer_conj_or_grad}
        else:
            return {'gradient': np.array(field_layer_conj_or_grad)}

    def _propagationInplace(self, field, propagation_distance, adjoint=False, in_real=True,precomputed_kernel=None):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field
        propagation_distance:   distance to propagate the wave
        adjoint:                boolean variable to perform adjoint operation (i.e. opposite direction)
        """
        if in_real:
            field = torch.fft.fft2(field)

        # EN: Use the precomputed kernel when supplied (8/26/24 optimization for uniform slice
        # spacing); otherwise build exp(+/- prop_kernel_phase * distance) on the fly, conjugated
        # for the adjoint direction.
        # 中文：若提供了预计算的传播核则直接使用（8/26/24 针对均匀切片间距的优化）；
        # 否则即时构造 exp(±prop_kernel_phase * 距离)，伴随方向时取共轭。
        # if adjoint:
        #     field[:, :] *= torch.conj(torch.exp(self.prop_kernel_phase * propagation_distance))
        # else:
        #     field[:, :] *= torch.exp(self.prop_kernel_phase * propagation_distance)
        if precomputed_kernel is None:
            if adjoint:
                kernel = torch.conj(torch.exp(self.prop_kernel_phase * propagation_distance))
            else:
                kernel = torch.exp(self.prop_kernel_phase * propagation_distance)
        else:
            kernel = precomputed_kernel
        field *= kernel
        if in_real:
            field = torch.fft.ifft2(field)
        return field

class MultiPhaseContrast(MultiTransmittance):
    """ MultiPhaseContrast, solves directly for the phase contrast {i.e. Transmittance = exp(sigma * PhaseContrast)} """

    def __init__(self, phase_obj_3d, wavelength, sigma=1, **kwargs):
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        self.sigma = sigma

    # EN: Variant of forward() that keeps the field at every slice, returning the full 3D field
    # stack instead of only the exit wave (useful for visualization/debugging).
    # 中文：forward() 的变体：保留每一层的场，返回完整的三维场堆栈而非仅出射波
    # （便于可视化/调试）。
    def forward3D(self, contrast_obj, fx_illu, fy_illu, fz_illu_layer):
        field, fx_illu, fy_illu, fz_illu = self._genSphericalIllumination(fx_illu, fy_illu)
        field3D = torch.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype=torch_complex_datatype)
        field3D[:, :, 0] = field * np.exp(1.0j * 2.0 * np.pi * fz_illu * self.initial_z_position)
        field_layer_conj = torch.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype=torch_complex_datatype)
        if (type(contrast_obj).__module__ == np.__name__):
            phasecontrast_obj_torch = torch.from_numpy(contrast_obj)
            flag_gpu_inout = False
        else:
            phasecontrast_obj_torch = contrast_obj
            flag_gpu_inout = True
        # Binning
        obj_torch = self._binObject(
            phasecontrast_obj_torch)  # binning in z. if slice_binning_factor=1, obj_af=phasecontrast_obj_af=contrast_obj

        # Potentials to Transmittance
        obj_torch = torch.exp(1.0j * self.sigma * obj_torch)

        # for layer in range(self.shape[2]):
        for layer in range(int(fz_illu_layer), self.shape[2]):
            field_layer_conj[:, :, layer] = torch.conj(field3D[:, :, layer])
            field3D[:, :, layer] *= obj_torch[:, :, layer]  # field interacts with the phantom.
            kernel = torch.exp(self.prop_kernel_phase * self.slice_separation[layer])
            if layer < self.shape[2] - 1:
                field3D[:, :, layer + 1] = self._propagationInplace(field3D[:, :, layer], self.slice_separation[layer])

        cache = (obj_torch, field_layer_conj, flag_gpu_inout)

        return {'forward_scattered_field': field3D, 'cache': cache}

    # EN: Main forward model used in the paper: spherical illumination first travels a free-space
    # distance (measured experimentally) before entering the volume, the phase-contrast object is
    # converted to transmittance exp(i*sigma*contrast), then propagated slice by slice with the
    # precomputed kernel; the exit wave is finally refocused using the calibrated b2b/b2c distances.
    # 中文：论文中使用的主正向模型：球面波照明先经过一段自由空间距离（由实验测得）再进入
    # 体积；相位衬度物体被转换为透过率 exp(i*sigma*contrast)，随后使用预计算的传播核逐层
    # 传播；最后用标定得到的 b2b/b2c 距离将出射波重新聚焦。
    def forward(self, contrast_obj, fx_illu, fy_illu, fz_illu_layer):
        #0130katz
        free_layers=self.free_space
        # compute illumination
        # field, fx_illu, fy_illu, fz_illu   = self._genIllumination(fx_illu, fy_illu)
        field, fx_illu, fy_illu, fz_illu = self._genSphericalIllumination(fx_illu, fy_illu)
        # propagate without interaction to object, the value is measured from experimental data
        field = self._propagationInplace(field, free_layers)
        # field[:, :]                       *= np.exp(1.0j * 2.0 * np.pi * fz_illu)
        field[:, :] *= torch.exp(1.0j * 2.0 * np.pi * fz_illu * self.initial_z_position)#1/10/24

        # multi-slice transmittance forward
        field_layer_conj = torch.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype=torch_complex_datatype)
        if (type(contrast_obj).__module__ == np.__name__):
            phasecontrast_obj_torch = torch.from_numpy(contrast_obj)
            flag_gpu_inout = False
        else:
            phasecontrast_obj_torch = contrast_obj
            flag_gpu_inout = True

        # Binning
        obj_torch = self._binObject(
            phasecontrast_obj_torch)  # binning in z. if slice_binning_factor=1, obj_af=phasecontrast_obj_af=contrast_obj

        # Potentials to Transmittance
        obj_torch = torch.exp(1.0j * self.sigma * obj_torch)
        # EN: Multi-slice loop starting at the source's z-layer (fz_illu_layer): multiply by the
        # slice transmittance, then propagate one uniform slice spacing with the cached kernel.
        # The commented block below shows an optional gradient-checkpointing version to save memory.
        # 中文：多层循环，从光源所在的 z 层（fz_illu_layer）开始：先乘以该层透过率，
        # 再用缓存的传播核传播一个均匀切片间距。下方注释掉的代码是可选的梯度检查点
        # （checkpoint）版本，用于节省显存。
        # for layer in range(self.shape[2]):
        for layer in range(int(fz_illu_layer), self.shape[2]):
            field_layer_conj[:, :, layer] = torch.conj(field)
            field[:, :] *= obj_torch[:, :, layer]  # field interacts with the phantom.
            if layer < self.shape[2] - 1:

                # kernel = torch.exp(self.prop_kernel_phase * self.slice_separation[layer])
                # self.slice_separation = self.pixel_size_z * np.ones((shape[2]-1,), dtype = np_float_datatype) #even speration
                field = self._propagationInplace(field, self.slice_separation[layer],precomputed_kernel=self.kernel)
            # def layer_forward(field, obj_slice):
            #     field = field * torch.exp(1.0j * self.sigma * obj_slice)
            #     if layer < self.shape[2] - 1:
            #         field = self._propagationInplace(field, self.slice_separation[layer], precomputed_kernel=self.kernel)
            #     return field
            # field = checkpoint(layer_forward, field, obj_torch[:, :, layer])
                
        cache = (obj_torch, field_layer_conj, flag_gpu_inout)

        if self.focus_at_center:
            # propagate to volume center
            field = self._propagationInplace(field, self.args.b2b, adjoint=False)
            field = self._propagationInplace(field,  self.args.b2c, adjoint=True)
        # print("field here:", field)

        return {'forward_scattered_field': field, 'cache': cache}

    # EN: Adjoint for the phase-contrast model: back-propagate the residual through the slices in
    # reverse, applying the chain-rule factor (-i*sigma*conj(trans)) per layer, then unbin in z.
    # 中文：相位衬度模型的伴随运算：将残差逆序反向传播穿过各切片，逐层乘以链式法则
    # 因子（-i*sigma*conj(透过率)），最后在 z 方向做反分箱。
    def adjoint(self, residual, cache):
        phasecontrast_obj_torch, field_layer_conj_or_grad, flag_gpu_inout = cache
        trans_obj_torch_conj = torch.conj(phasecontrast_obj_torch)
        # back-propagte to volume center
        field_bp = residual
        # propagate to the last layer
        if self.focus_at_begin:
            field = self._propagationInplace(field, self.distance_end_to_begin)
        if self.focus_at_center:
            field_bp = self._propagationInplace(field_bp, self.distance_end_to_center)
        # multi-slice transmittance backward
        for layer in range(self.shape[2] - 1, -1, -1):
            #             plt.figure()
            #             plt.imshow(np.real(field_bp))
            #             plt.colorbar()
            field_layer_conj_or_grad[:, :, layer] = field_bp * field_layer_conj_or_grad[:, :, layer] * (
                -1.0j) * self.sigma * trans_obj_torch_conj[:, :, layer]
            if layer > 0:
                field_bp[:, :] *= trans_obj_torch_conj[:, :, layer]
                field_bp = self._propagationInplace(field_bp, self.slice_separation[layer - 1], adjoint=True)

        # Unbinning
        grad = self._binObject(field_layer_conj_or_grad, adjoint=True)

        if flag_gpu_inout:
            return {'gradient': grad}
        else:
            return {'gradient': np.array(grad)}


class TomographySolver(nn.Module):
    """
    Highest level solver object for tomography problem

    phase_obj_3d:               phase_obj_3d object defined from class PhaseObject3D
    fx_illu_list:               illumination coordinate in x, default = [0] (on axis)
    fy_illu_list:               illumination coordinate in y
    fz_illu_list:               illumination coordinate in z
    rotation_angle_list:        angles of rotation in tomogrpahy
    propagation_distance_list:  defocus distances for each illumination
    """

    def __init__(self, phase_obj_3d, fx_illu_list=[0], fy_illu_list=[0], fz_illu_list=[0], rotation_angle_list=[0],
                 propagation_distance_list=[0], **kwargs):
        # EN: Assemble the full imaging pipeline: illumination source lists (x/y/z), an
        # Aberration (pupil) operator, a Defocus operator, and the chosen scattering model.
        # 中文：组装完整成像流水线：照明光源列表（x/y/z 坐标）、像差（光瞳）算子、
        # 离焦算子，以及所选的散射模型。
        self.phase_obj_3d = phase_obj_3d
        self.wavelength = kwargs["wavelength"]
        # Rotation angels and objects
        self.rot_angles = rotation_angle_list
        self.number_rot = len(self.rot_angles)
        self.rotation_pad = kwargs.get("rotation_pad", True)

        # Illumination source coordinates
        assert len(fx_illu_list) == len(fy_illu_list)
        self.fx_illu_list = fx_illu_list
        self.fy_illu_list = fy_illu_list
        self.fz_illu_list = fz_illu_list
        self.number_illum = len(self.fx_illu_list)
        # Aberation object
        self._aberration_obj = Aberration(phase_obj_3d.shape[:2], phase_obj_3d.pixel_size, self.wavelength,
                                          kwargs["na"], pad=False)
        # print("_aberration_obj:", self._aberration_obj)

        # Defocus distances and object
        self.prop_distances = propagation_distance_list
        self._defocus_obj = Defocus(phase_obj_3d.shape[:2], phase_obj_3d.pixel_size, **kwargs)
        self.number_defocus = len(self.prop_distances)

        # Scattering models and algorithms
        self._opticsmodel = {"MultiTrans": MultiTransmittance,
                             "MultiPhaseContrast": MultiPhaseContrast,
                             }
        # self._algorithms     = {"GradientDescent":    self._solveFirstOrderGradient,
        #                         "FISTA":              self._solveFirstOrderGradient
        #                        }
        self.scat_model_args = kwargs

    def setScatteringMethod(self, model="MultiTrans"):
        """
        Define scattering method for tomography

        model: scattering models, it can be one of the followings:
               "MultiTrans", "MultiPhaseContrast"(Used in the paper)
        """
        # EN: Convert the stored RI volume into the representation required by the chosen model
        # (transmittance, phase contrast, or scattering potential) and instantiate that model.
        # 中文：将存储的折射率体积转换为所选模型需要的表示（透过率、相位衬度或散射势），
        # 并实例化对应的散射模型。
        self.scat_model = model
        if hasattr(self, '_scattering_obj'):
            del self._scattering_obj

        if model == "MultiTrans":
            self.phase_obj_3d.convertRItoTrans(self.wavelength)
            self.phase_obj_3d.convertRItoV(self.wavelength)
            self._x = self.phase_obj_3d.trans_obj
            # if np.any(self.rot_angles != [0]):
            #     self._rot_obj = ImageRotation(self.phase_obj_3d.shape, axis=0, pad=self.rotation_pad, pad_value=1, \
            #                                   flag_gpu_inout=True, flag_inplace=True)
        elif model == "MultiPhaseContrast":
            if not hasattr(self.phase_obj_3d, 'contrast_obj'):
                self.phase_obj_3d.convertRItoPhaseContrast()
            self._x = self.phase_obj_3d.contrast_obj  # this line is the only line excuted in this moment.
            # if np.any(self.rot_angles != [0]):
            #     self._rot_obj = ImageRotation(self.phase_obj_3d.shape, axis=0, pad=self.rotation_pad, pad_value=0, \
            #                                   flag_gpu_inout=True, flag_inplace=True)
        else:
            if not hasattr(self.phase_obj_3d, 'V_obj'):
                self.phase_obj_3d.convertRItoV(self.wavelength)
            self._x = self.phase_obj_3d.V_obj

        self._scattering_obj = self._opticsmodel[model](self.phase_obj_3d, **self.scat_model_args)

    def forwardPredict(self, field=False):
        """
        Uses current object in the phase_obj_3d to predict the amplitude of the exit wave
        Before calling, make sure correct object is contained
        """
        # EN: Loop over all illumination sources, run the forward model for each, and stack the
        # predicted exit-wave fields into a tensor shaped for the loss computation.
        # 中文：遍历所有照明光源，对每个光源运行正向模型，并将预测的出射波场堆叠成
        # 适合损失计算的张量形状。
        obj_gpu = self._x
        #with contexttimer.Timer() as timer:
        forward_scattered_predict = []
        # forward_scattered_predict.append([])
        for illu_idx in range(self.number_illum):
            fx_illu = self.fx_illu_list[illu_idx]  # every illumination source
            fy_illu = self.fy_illu_list[illu_idx]
            fz_illu_layer = self.fz_illu_list[illu_idx]
            fields = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj=obj_gpu)
            # print(fx_illu,fy_illu,fz_illu_layer)
            # print("fields:",fields["forward_scattered_field"])
            if field:
                forward_scattered_predict.append(fields["forward_scattered_field"])
        if len(forward_scattered_predict[0][0].shape) == 2:
            forward_scattered_predict = forward_scattered_predict[0][0].unsqueeze(0).unsqueeze(0)
            forward_scattered_predict = forward_scattered_predict.permute(2, 3, 1, 0)
        elif len(forward_scattered_predict[0][0].shape) == 3:
            forward_scattered_predict = forward_scattered_predict[0][0].unsqueeze(0).unsqueeze(0)
            forward_scattered_predict = forward_scattered_predict.permute(2, 3, 4, 1, 0)
        return forward_scattered_predict, fields

    def _forwardMeasure(self, fx_illu, fy_illu, fz_illu_layer, obj=None):

        """
        From an illumination position, this function computes the exit wave.
        fx_illu, fy_illu, fz_illu:       illumination coordinates in x, y and z (scalars)
        obj:                    object to be passed through (Optional, default pick from phase_obj_3d)
        """
        # if obj is None:
        #     #not this case
        #     fields = self._scattering_obj.forward(self._x, fx_illu, fy_illu, fz_illu_layer) #self._x = self.phase_obj_3d.contrast_obj=self.RI_obj - self.RI
        # else:
        # EN: Full measurement chain: scattering model (multi-slice propagation) -> pupil/aberration
        # filtering -> defocus to the camera plane.
        # 中文：完整测量链路：散射模型（多层传播）-> 光瞳/像差滤波 -> 离焦到相机平面。
        # execute this case, fields is a dictionary contains a complex 2D array "field" and a tuple "cache"
        fields = self._scattering_obj.forward(obj, fx_illu, fy_illu, fz_illu_layer)
        field_scattered = self._aberration_obj.forward(fields["forward_scattered_field"])
        field_scattered = self._defocus_obj.forward(field_scattered, self.prop_distances)
        fields["forward_scattered_field"] = field_scattered

        # if self._scattering_obj.back_scatter:
        #     field_scattered                   = self._aberration_obj.forward(fields["back_scattered_field"])
        #     field_scattered                   = self._defocus_obj.forward(field_scattered, self.prop_distances)
        #     fields["back_scattered_field"]    = field_scattered
        return fields


