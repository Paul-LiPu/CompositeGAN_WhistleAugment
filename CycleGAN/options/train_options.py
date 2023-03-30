from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Newly added arguments
        self.parser.add_argument('--pos_G', type=str, default='wgan2',
                                 help='choice of negative patch G net: wgan | dcgan')
        self.parser.add_argument('--pos_Gnet_model', type=str, required=True,
                                 help='path to negative patch G net model')
        self.parser.add_argument('--pos_n_latent', type=int, default='128',
                                 help='number of latent variable for positive G net')

        self.parser.add_argument('--neg_G', type=str, default='wgan2',
                                 help='choice of negative patch G net: wgan | dcgan')
        self.parser.add_argument('--neg_Gnet_model', type=str, required=True,
                                 help='path to negative patch G net model')
        self.parser.add_argument('--neg_n_latent', type=int, default='128',
                                 help='number of latent variable for negative G net')

        self.parser.add_argument('--contour_model', type=str, required=True,
                                 help='path to contour net model file')
        self.parser.add_argument('--LAMBDA_CONFIDENCE', type=float, default=0.5,
                                 help='weight for confidence_weight loss ')
        self.parser.add_argument('--LAMBDA_A_neg', type=float, default=0.5, help='weight for confidence_weight loss ')
        self.parser.add_argument('--LAMBDA_B_neg', type=float, default=0.5, help='weight for confidence_weight loss ')


        # Arguments from CycleGAN
        self.isTrain = True
        self.parser.add_argument('--niter', type=int, required=True, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, required=True,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay_iters', type=int, required=True,
                                 help='multiply by a gamma every lr_decay_iters iterations, applied when use step lr_policy')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=10, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=100, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5,
                                 help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                                      'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')


