class ScalesInjector():
    def __init__(self, args, vae, scale_schedule, tgt_h, tgt_w):
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.inject_scales, self.inject_scales_bits = self._get_inject_scales(args.inject_scales, args.inject_scales_path, vae, scale_schedule, tgt_h, tgt_w)
        self.teacher_forcing = args.inject_scales_teacher_forcing
        
    
    def _get_inject_scales(self, inject_scales, img_path, vae_wrapper, scale_schedule, tgt_h, tgt_w):
        match inject_scales:
            case 6: inject_scales = range(2, 13)
            case 5: inject_scales = range(0,13)
            case 4: inject_scales = range(0,12)
            case 3: inject_scales = range(0,9)
            case 2: inject_scales = range(0,6)
            case 1: inject_scales = [0,1,2,3]
            case 0: inject_scales = []
            case _: raise(NotImplementedError) 
        encoding_bit_indices = None
        if inject_scales:
             _, _, encoding_bit_indices, *_ = vae_wrapper.encode(img_path, False)

        return inject_scales, encoding_bit_indices