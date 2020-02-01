from networks import model_3detectors
from networks import model_3detectors_1dconv
from networks import model_3detectors_coatt
from networks import model_3detectors_coatt_selfatt
from networks import model_3detectors_coatt_bitrilinear
from networks import model_3detectors_transformer
from networks import model_3detectors_vlad
from networks import model_3detectors_transformervlad
from networks import model_3detectors_transformervlad_coatt


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
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-1dconv':
        model = model_3detectors_1dconv.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        dconv_weight = ['scene_conv', 'object_conv', 'action_conv']
        temp_dconv = []
      
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

            elif k[:-5] in dconv_weight or k[:-7] in dconv_weight:
                print('c', k)
                temp_dconv.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_dconv + temp_scratch
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

    elif opt.model_name == 'FtDetectorFc-3detectors-Vlad':
        model = model_3detectors_vlad.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        vlad_center_ft_module_names = 'centroids'
        vlad_weight_ft_module_names = 'soft_weight'
        temp_vlad = []
        
        #scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim', 'action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
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
            
            elif vlad_center_ft_module_names in k:
                print('c', k)
                temp_vlad.append(v)
            elif vlad_weight_ft_module_names in k:
                print('c', k)
                temp_vlad.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_vlad + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-TransformerVlad':
        model = model_3detectors_transformervlad.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        vlad_center_ft_module_names = 'centroids'
        vlad_weight_ft_module_names = 'soft_weight'
        temp_vlad = []
        
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
            
            elif vlad_center_ft_module_names in k:
                print('c', k)
                temp_vlad.append(v)
            elif vlad_weight_ft_module_names in k:
                print('c', k)
                temp_vlad.append(v)
           
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_vlad + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-3detectors-TransformerVlad-Coatt':
        model = model_3detectors_transformervlad_coatt.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        attention_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac', 
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac', 
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_att = []

        vlad_center_ft_module_names = 'centroids'
        vlad_weight_ft_module_names = 'soft_weight'
        temp_vlad = []
     
        scratch_train_module_names = ['scene_reduce_dim', 'object_reduce_dim','action_reduce_dim', 'concat_reduce_dim', 'final_classifier']
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

            elif vlad_center_ft_module_names in k:
                print('d', k)
                temp_vlad.append(v)
            elif vlad_weight_ft_module_names in k:
                print('d', k)
                temp_vlad.append(v)

            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('e', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_att + temp_vlad + temp_scratch
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    return model, parameters
