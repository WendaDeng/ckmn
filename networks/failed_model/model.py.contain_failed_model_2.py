from networks import model_3detectors
from networks import model_3detectors_nonlocal
from networks import model_3detectors_dilated
from networks import model_3detectors_dilated_nonlocal
from networks import model_3detectors_dilatednonlocal_coatt_serial_concat
from networks import model_3detectors_dilatednonlocal_coatt_parallel_concat
from networks import model_3detectors_coatt
from networks import model_3detectors_nonlocal_coatt
from networks import model_3detectors_coatt_selfatt
from networks import model_3detectors_coatt_trilinear
from networks import model_3detectors_coatt_bilinear
from networks import model_3detectors_coatt_bitrilinear
from networks import model_3detectors_nonlocal_coatt_bitrilinear
from networks import model_3detectors_transformer


def generate_model(opt):

    if opt.model_name == 'FtDetectorFc-3detectors':
        model = model_3detectors.Event_Model(opt)
      
        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('c', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Coatt':
        model = model_3detectors_coatt.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_att = []
 
        #bn_weight =['scene_coattention_bn1', 'object_coattention_bn1', 'action_coattention_bn1', 'concat_bn1', 'final_bn1']
        #temp_bn = []
     
        #scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim','action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)

            #elif k[:-5] in bn_weight or k[:-7] in bn_weight:
            #    print('e', k)
            #    temp_bn.append(v)

            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch# + temp_bn
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Nonlocal':
        model = model_3detectors_nonlocal.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        attention_weight = 'nonlocal'
        temp_att = []
      
        #scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim','action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        #scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif attention_weight in k:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Dilated':
        model = model_3detectors_dilated.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        dilated_weight = 'dilated'
        temp_att = []
      
        #scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim','action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
        #scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif dilated_weight in k:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Dilated-Nonlocal':
        model = model_3detectors_dilated_nonlocal.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        dilated_weight = 'dilated'
        attention_weight = 'nonlocal'
        temp_att = []
      
        #scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim','action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
        #scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif dilated_weight in k:
                print('c', k)
                temp_att.append(v)
            elif attention_weight in k:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Serial-Concat':
        model = model_3detectors_dilatednonlocal_coatt_serial_concat.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        coatt_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_coatt = []
 
        dilated_weight = 'dilated'
        nonlocal_weight = 'nonlocal'
        temp_dnl = []
        
        bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'concat_bn1', 'final_bn1']
        temp_bn = []
     
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif k in coatt_weight:
                print('c', k)
                temp_coatt.append(v)

            elif dilated_weight in k:
                print('d', k)
                temp_dnl.append(v)
            elif nonlocal_weight in k:
                print('d', k)
                temp_dnl.append(v)

            elif k[:-5] in bn_weight or k[:-7] in bn_weight:
                print('e', k)
                temp_bn.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('f', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False

        temp = temp_fc + temp_coatt + temp_dnl + temp_bn + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Parallel-Concat':
        model = model_3detectors_dilatednonlocal_coatt_parallel_concat.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        coatt_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_coatt = []
 
        dilated_weight = 'dilated'
        nonlocal_weight = 'nonlocal'
        temp_dnl = []
        
        bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'dilatednonlocal_redudim_bn1', 'coattention_redudim_bn1', 'concat_bn1', 'final_bn1']
        temp_bn = []
     
        scratch_train_module_names = ['dilatednonlocal_reduce_dim', 'coattention_reduce_dim', 'concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif k in coatt_weight:
                print('c', k)
                temp_coatt.append(v)

            elif dilated_weight in k:
                print('d', k)
                temp_dnl.append(v)
            elif nonlocal_weight in k:
                print('d', k)
                temp_dnl.append(v)

            elif k[:-5] in bn_weight or k[:-7] in bn_weight:
                print('e', k)
                temp_bn.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('f', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False

        temp = temp_fc + temp_coatt + temp_dnl + temp_bn + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Nonlocalcoatt':
        model = model_3detectors_nonlocal_coatt.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        nonlocal_weight = 'nonlocal'
        temp_att = []
      
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)

            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)
            elif nonlocal_weight in k:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Coatt-Selfatt':
        model = model_3detectors_coatt_selfatt.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        self_attention = ['scene_attention', 'object_attention', 'action_attention']
        temp_att = []
        
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)
            
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)
            elif k in self_attention:
                print('c', k)
                temp_att.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Coatt-Trilinear':
        model = model_3detectors_coatt_trilinear.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_att = []

        bitri_weight = ['scene_trilinear', 'object_trilinear', 'action_trilinear']
        temp_bitri = []
 
        #bn_weight =['scob_trilinear_bn1', 'scac_trilinear_bn1', 'obac_trilinear_bn1', 'final_bn1']
        bn_weight =['final_bn1']
        temp_bn = []
       
        scratch_train_module_names = ['final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)
            
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in bitri_weight or k[:-7] in bitri_weight:
                print('d', k)
                temp_bitri.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('e', k)
                temp_scratch.append(v)

            elif k[:-5] in bn_weight or k[:-7] in bn_weight:
                print('d', k)
                temp_bitri.append(v)

            else:
                v.requires_grad = False
        
        temp = temp_fc + temp_bitri + temp_att + temp_scratch + temp_bn
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Coatt-Bilinear':
        model = model_3detectors_coatt_bilinear.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_att = []

        bitri_weight = ['scene_bilinear_scob', 'object_bilinear_scob', 'scene_bilinear_scac', 'action_bilinear_scac', 'object_bilinear_obac', 'action_bilinear_obac']
        temp_bitri = []
        
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)
            
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in bitri_weight or k[:-7] in bitri_weight:
                print('d', k)
                temp_bitri.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('e', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        
        temp = temp_fc + temp_bitri + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Coatt-Bitrilinear':
        model = model_3detectors_coatt_bitrilinear.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_att = []

        bitri_weight = ['scene_bilinear_scob', 'object_bilinear_scob', 'scene_bilinear_scac', 'action_bilinear_scac', 'object_bilinear_obac', 'action_bilinear_obac', 'scob_trilinear', 'scac_trilinear', 'obac_trilinear']
        temp_bitri = []
        
        bn_weight =['scene_coattention_bn1', 'object_coattention_bn1', 'action_coattention_bn1', 'scene_bilinear_scob_bn1', 'object_bilinear_scob_bn1', 'scene_bilinear_scac_bn1', 'action_bilinear_scac_bn1', 'object_bilinear_obac_bn1', 'action_bilinear_obac_bn1', 'scob_trilinear_bn1', 'scac_trilinear_bn1', 'obac_trilinear_bn1', 'final_bn1']
        temp_bn = []

        scratch_train_module_names = ['final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)
            
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in bitri_weight or k[:-7] in bitri_weight:
                print('d', k)
                temp_bitri.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('e', k)
                temp_scratch.append(v)

            elif k[:-5] in bn_weight or k[:-7] in bn_weight:
                print('f', k)
                temp_bn.append(v)

            else:
                v.requires_grad = False
        
        temp = temp_fc + temp_bitri + temp_att + temp_scratch + temp_bn
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Nonlocal-Coatt-Bitrilinear':
        model = model_3detectors_nonlocal_coatt_bitrilinear.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []
        
        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
       
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        nonlocal_weight = 'nonlocal'
        temp_att = []

        bitri_weight = ['scene_bilinear_scob', 'object_bilinear_scob', 'scene_bilinear_scac', 'action_bilinear_scac', 'object_bilinear_obac', 'action_bilinear_obac', 'scob_trilinear', 'scac_trilinear', 'obac_trilinear']
        temp_bitri = []
        
        scratch_train_module_names = ['final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)
            
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif k in attention_weight:
                print('c', k)
                temp_att.append(v)
            elif nonlocal_weight in k:
                print('c', k)
                temp_att.append(v)

            elif k[:-5] in bitri_weight or k[:-7] in bitri_weight:
                print('d', k)
                temp_bitri.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('e', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        
        temp = temp_fc + temp_bitri + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-Transformer':
        model = model_3detectors_transformer.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        transformer_ft_module_names = 'encoding'
        temp_trans = []
        
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('b', k)
                temp_fc.append(v)
            
            elif transformer_ft_module_names in k:
                print('c', k)
                temp_trans.append(v)
            
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_trans + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})
    
    return model, parameters
