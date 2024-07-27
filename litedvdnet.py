import torch
from torch import nn
from enum import Enum

class InferenceMode(Enum):
    Basic = 0
    Cached = 1

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class LiteCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU)'''
    def __init__(self, in_ch, out_ch):
        super(LiteCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, interm_ch, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = interm_ch
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch, simplifed_cv):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            LiteCvBlock(out_ch, out_ch) if simplifed_cv else CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)

class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU) + Upscale'''
    def __init__(self, in_ch, out_ch, simplifed_cv):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            LiteCvBlock(in_ch, in_ch) if simplifed_cv else CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, channels, interm_ch, simp_cv, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = channels[0]
        self.chs_lyr1 = channels[1]
        self.chs_lyr2 = channels[2]

        self.inc = InputCvBlock(num_in_frames=num_input_frames, interm_ch=interm_ch, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, simplifed_cv=simp_cv)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, simplifed_cv=simp_cv)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, simplifed_cv=simp_cv)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, simplifed_cv=simp_cv)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input_tensors: list, noise_map):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        input_tensor = None
        for inX in input_tensors:
            if input_tensor is None:
                input_tensor = torch.cat((inX, noise_map), dim=1)
            else:
                input_tensor = torch.cat((input_tensor, inX, noise_map), dim=1)

        denoisedFrame = input_tensors[len(input_tensors) // 2]

        x0 = self.inc(input_tensor)
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)
        # Residual
        x = denoisedFrame - x

        return x

class LiteDVDNet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=5, inference_mode='Basic', interm_ch=30, simple_cv=False,
                 channels=[32, 64, 128], pretrain_ckpt=None):
        super(LiteDVDNet, self).__init__()
        self.num_input_frames = num_input_frames

        self.inference_mode = InferenceMode[inference_mode]
        self.simple_cv = simple_cv
        self.interm_ch = interm_ch

        self.prev_den_frame = None
        self.current_den_frame = None
        self.future_den_frame = None

        self.channels = channels

        # Define models of each denoising stage
        self.temp1 = DenBlock(channels, interm_ch, simple_cv, num_input_frames=3)
        self.temp2 = DenBlock(channels, interm_ch, simple_cv, num_input_frames=3)

        # Init weights
        self.reset_params()

        if pretrain_ckpt is not None:
            self.load(pretrain_ckpt)

    def get_desciption(self):
        chs_lyr0 = self.channels[0]
        chs_lyr1 = self.channels[1]
        chs_lyr2 = self.channels[2]
        simple_cv = "_s" if self.simple_cv else ""
        interm_ch = self.interm_ch
        return f'{__class__.__name__.lower()}_{chs_lyr0}_{chs_lyr1}_{chs_lyr2}_ich{interm_ch}{simple_cv}'

    def are_buffers_empty(self) -> bool:
        return self.prev_den_frame is None and self.current_den_frame is None and self.future_den_frame is None

    def load(self, pretrain_ckpt):
        state_temp_dict = torch.load(pretrain_ckpt)
        state_dict = self.extract_dict(state_temp_dict, string_name="module.")
        self.load_state_dict(state_dict)

    def extract_dict(self, ckpt_state, string_name, replace_name=''):
        m_dict = {}
        for k, v in ckpt_state.items():
            if string_name in k:
                m_dict[k.replace(string_name, replace_name)] = v
        return m_dict

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        match(self.inference_mode):
            case InferenceMode.Basic:
                return self.forward_basic(x, noise_map)
            case InferenceMode.Cached:
                return self.forward_cached(x, noise_map)

    def forward_basic(self, x, noise_map):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1([x0, x1, x2], noise_map)
        x21 = self.temp1([x1, x2, x3], noise_map)
        x22 = self.temp1([x2, x3, x4], noise_map)

        # Second stage
        x = self.temp2([x20, x21, x22], noise_map)

        return x


    def forward_cached(self, x, noise_map):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames))

        # Denoise prev frame and buffer it, or get from buffer
        if self.prev_den_frame is None:
            x20 = self.temp1([x0, x1, x2], noise_map)
            self.prev_den_frame = x20
        else:
            x20 = self.prev_den_frame

        # Denoise current frame and buffer it, or get from buffer
        if self.current_den_frame is None:
            x21 = self.temp1([x1, x2, x3], noise_map)
            self.current_den_frame = x21
        else:
            x21 = self.current_den_frame

        # Denoise future frame and buffer it, or get from buffer
        if self.future_den_frame is None:
            x22 = self.temp1([x2, x3, x4], noise_map)
            self.future_den_frame = x22
        else:
            x22 = self.future_den_frame

        # Second stage
        x = self.temp2([x20, x21, x22], noise_map)

        # Shift all buffers
        self.prev_den_frame = self.current_den_frame
        self.current_den_frame = self.future_den_frame
        self.future_den_frame = None


        return x