from networks import model_sc
from networks import model_ob
from networks import model_ac
from networks import model_scob
from networks import model_3detectors
from networks import model_2detectors_graph
from networks import model_3detectors_nonlocal
from networks import model_3detectors_dilated
from networks import model_3detectors_dilated_nonlocal
from networks import model_3detectors_nonlocal_dilated
from networks import model_3detectors_coatt
from networks import model_3detectors_dilatednonlocal_coatt_serial_concat
from networks import model_3detectors_dilatednonlocal_coatt_parallel_concat
from networks import model_3detectors_dilatednonlocal_coatt_parallel_bilinear


def generate_model(opt):

    if opt.model_name == 'FtDetectorFc-3detectors':
        model = model_3detectors.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []

        # scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier', 'fc_verb', 'fc_noun']
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

    elif opt.model_name == 'FtDetectorFc-2detectors-Graph':
        model = model_2detectors_graph.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = 'object_detector.fc'
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []

        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier', 'fc_verb', 'fc_noun']
        temp_scratch = []

        graph_weight = 'gcn'
        temp_graph = []

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
        temp = temp_fc + temp_scratch + temp_graph
        parameters.append({'params': temp_conv})
        parameters.append({'params': temp})

    elif opt.model_name == 'FtDetectorFc-Sc':
        model = model_sc.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc']
        temp_fc = []

        scratch_train_module_names = ['final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

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

    elif opt.model_name == 'FtDetectorFc-Ob':
        model = model_ob.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['object_detector.fc']
        temp_fc = []

        scratch_train_module_names = ['final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

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

    elif opt.model_name == 'FtDetectorFc-Ac':
        model = model_ac.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []

        scratch_train_module_names = ['final_classifier', 'fc_verb', 'fc_noun']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

            elif action_detectors_ft_module_names in k:
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

    elif opt.model_name == 'FtDetectorFc-Scob':
        model = model_scob.Event_Model(opt)

        conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        temp_fc = []

        scratch_train_module_names = ['concat_reduce_dim', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if conv_ft_module_names in k:
                print('a', k)
                temp_conv.append(v)

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
        #scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        scratch_train_module_names = ['final_bn1', 'final_classifier']
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

    elif opt.model_name == 'FtDetectorFc-3detectors-Nonlocal-Dilated':
        model = model_3detectors_nonlocal_dilated.Event_Model(opt)

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

        bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'dilatednonlocal_redudim_bn1', 'coattention_redudim_bn1', 'final_bn1']
        # bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'concat_bn1', 'final_bn1']
        temp_bn = []

        scratch_train_module_names = ['dilatednonlocal_reduce_dim', 'coattention_reduce_dim', 'final_classifier']
        # scratch_train_module_names = ['dilatednonlocal_reduce_dim', 'coattention_reduce_dim', 'concat_reduce_dim', 'final_classifier']
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

    elif opt.model_name == 'FtDetectorFc-3detectors-DilatedNonlocal-Coatt-Parallel-Bilinear':
        model = model_3detectors_dilatednonlocal_coatt_parallel_bilinear.Event_Model(opt)

        #conv_ft_module_names = 'layer4.2.conv3'
        temp_conv = []

        #detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        #action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []

        coatt_weight = ['Wb_sc_ob', 'Wv_sc_ob', 'Ws_sc_ob', 'Whv_sc_ob', 'Whs_sc_ob', 'Wb_sc_ac',
                           'Wv_sc_ac', 'Ws_sc_ac', 'Whv_sc_ac', 'Whs_sc_ac', 'Wb_ob_ac', 'Wv_ob_ac',
                           'Ws_ob_ac', 'Whv_ob_ac', 'Whs_ob_ac']
        temp_coatt = []

        dilated_weight = 'dilated'
        nonlocal_weight = 'nonlocal'
        temp_dnl = []

        bn_weight =['scene_dilated_bn1', 'object_dilated_bn1', 'action_dilated_bn1', 'scene_nonlocal_bn1', 'object_nonlocal_bn1', 'action_nonlocal_bn1', 'final_bn1']
        temp_bn = []

        scratch_train_module_names = ['dilatednonlocal_bilinear', 'coattention_bilinear', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            #if conv_ft_module_names in k:
            #    print('a', k)
            #    temp_conv.append(v)

            #elif action_detectors_ft_module_names in k:
            #    print('b', k)
            #    temp_fc.append(v)
            #elif k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
            #    print('b', k)
            #    temp_fc.append(v)

            if k in coatt_weight:
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


    return model, parameters
