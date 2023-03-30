from .base_options import BaseOptions


class Data_Options(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='60', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # Pu Li, 03-14-2019
        # pos_model_file = '/home/lipu/Code/GAN/wgan-gp/models/whistle_pos2/WGAN_iter50000'
        # neg_model_file = '/home/lipu/Code/GAN/wgan-gp/models/whistle_neg/WGAN_iter50000'

        # pos_model_file = '/home/lipu/Code/GAN/wgan-gp/models/whistle_pos_tanh/WGAN_iter90000'
        # neg_model_file = '/home/lipu/Code/GAN/wgan-gp/models/whistle_neg_tanh/WGAN_iter140000'
        # gan_type = 'wgan2'


        self.parser.add_argument('--pos_G', type=str, default=gan_type,
                                 help='choice of negative patch G net: wgan | dcgan')
        self.parser.add_argument('--pos_Gnet_model', type=str, default=pos_model_file,
                                 help='path to negative patch G net model')
        self.parser.add_argument('--pos_n_latent', type=int, default='128',
                                 help='number of latent variable for positive G net')
        self.parser.add_argument('--neg_G', type=str, default=gan_type,
                                 help='choice of negative patch G net: wgan | dcgan')
        self.parser.add_argument('--neg_Gnet_model', type=str, default=neg_model_file,
                                 help='path to negative patch G net model')
        self.parser.add_argument('--neg_n_latent', type=int, default='128',
                                 help='number of latent variable for negative G net')

        netG_B_model_file = '/home/lipu/Documents/whale_recognition/experiments/DCL/Manual_data/' + \
                            'ResSeg_BN_logmespec_lineGT_common_bottlenose_lineGT_mix_1-1/models/DSRCNN-iter_1000000'
        self.parser.add_argument('--netG_B_model', type=str, default=netG_B_model_file,
                                 help='path to net G B model file')

        output_dir = './generated_data'
        self.parser.add_argument('--output_dir', type=str, default=output_dir,
                                 help='path for data output')

        netG_B_model_file = '/home/lipu/Documents/whale_recognition/experiments/DCL/Manual_data/' + \
                            'ResSeg_BN_logmespec_lineGT_common_bottlenose_lineGT_mix_1-1/models/DSRCNN-iter_1000000'
        self.parser.add_argument('--contour_model', type=str, default=netG_B_model_file,
                                 help='path to contour net model file')

        # num_patch = 4864
        num_patch = 73600
        # # num_patch = 73600
        # # num_patch = 15872
        self.parser.add_argument('--num_pos_data', type=int, default=num_patch,
                                 help='number of positive data')
        self.parser.add_argument('--num_neg_data', type=int, default=num_patch,
                                 help='number of negative data')
        self.isTrain = False
