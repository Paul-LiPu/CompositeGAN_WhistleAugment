def create_model(opt):
    model = None
    if opt.model == 'whistle_cycle_deepContour_negID':
        from .whistle_cycleGAN_deepcontour_negId_model import WhistleCycleGANDeepContourNegIDModel
        model = WhistleCycleGANDeepContourNegIDModel()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual1':
        from .whistle_cycleGAN_deepcontour_negId_model_residual1 import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual0':
        from .whistle_cycleGAN_deepcontour_negId_model_residual0 import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual0_woMag':
        from .whistle_cycleGAN_deepcontour_negId_model_residual0_woMag import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual0_e2e':
        from .whistle_cycleGAN_deepcontour_negId_model_residual0_e2e import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual2':
        from .whistle_cycleGAN_deepcontour_negId_model_residual2 import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    elif opt.model == 'whistle_cycle_deepContour_negID_residual3':
        from .whistle_cycleGAN_deepcontour_negId_model_residual3 import WhistleCycleGANDeepContourNegIDModel_Res
        model = WhistleCycleGANDeepContourNegIDModel_Res()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
