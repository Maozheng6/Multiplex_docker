import cntk
from cntk.layers import *
from cntk.initializer import *
from cntk.ops import *
import cntk.train.distributed as distributed


import numpy as np

def ddist(prediction, c_interval_center, c_interval_radius):
    ''' Distance of the predictions from the edges of the intervals '''
    #return cntk.relu(cntk.abs(prediction - c_interval_center) - c_interval_radius)
    return cntk.abs(prediction - c_interval_center)


def make_trainer(epoch_size, mb_size_in_samples, output, high_res_loss, loss, max_epochs,
    my_rank, number_of_workers, lr_adjustment_factor, log_dir, initial_lr):
    ''' Define the learning rate schedule, trainer, and evaluator '''
    lr_per_mb = [initial_lr] * 200 + [initial_lr*0.1] * 200 + [initial_lr*0.01] * 300 + [initial_lr*0.001] * 1000
    lr_per_mb = [lr * lr_adjustment_factor for lr in lr_per_mb]

    lr_schedule = cntk.learning_parameter_schedule(lr_per_mb, epoch_size=epoch_size*mb_size_in_samples)

    learner = cntk.rmsprop(
        parameters=output.parameters,
        lr=lr_schedule,
        gamma=0.95,
        inc=1.1,
        dec=0.9,
        max=1.1,
        min=0.9
    )

    '''
    learner = cntk.learners.adam(
        parameters=output.parameters,
        lr=lr_schedule
    )
    '''

    progress_printer = cntk.logging.ProgressPrinter(
        tag='Training',
        num_epochs=max_epochs,
        freq=epoch_size,
        rank=my_rank
    )

    tensorboard = cntk.logging.TensorBoardProgressWriter(freq=1, log_dir=log_dir, rank=None, model=output)
    #trainer = cntk.Trainer(output, (loss, high_res_loss), learner, [progress_printer, tensorboard])
    trainer = cntk.Trainer(output, (loss,loss), learner, [progress_printer, tensorboard])

    #evaluator = cntk.Evaluator(loss)

    return (trainer, tensorboard)


def get_model_muti_stain_idenpendent(f_dim, c_dim, l_dim, m_dim, num_stack_layers,
        super_res_class_weight, super_res_loss_weight, high_res_loss_weight,unet_level,stain_num=1,start_stain=0):
    # Define the variables into which the minibatch data will be loaded.
    num_nlcd_classes, num_landcover_classes = c_dim
    _, block_size, _ = f_dim
    _,block_size_m,_=l_dim
    input_im = cntk.input_variable(f_dim, np.float32)
    lc = cntk.input_variable(l_dim, np.float32)
    lc_weight_map = cntk.input_variable((1, l_dim[1], l_dim[2]), np.float32)
    interval_center = cntk.input_variable(c_dim, np.float32)
    interval_radius = cntk.input_variable(c_dim, np.float32)
    mask = cntk.input_variable(m_dim, np.float32)


    # Create the model definition. c_map defines the number of filters trained
    # at layers of different depths in the model. num_stack_layers defines the
    # number of (modified) residual units per layer.
    #c_map=[64, 32, 32, 32, 32],
    model = cnn_model(
        input_tensor=input_im,
        num_stack_layers=num_stack_layers,
        c_map=[16, 32, 32, 32, 32],
        num_classes=num_landcover_classes*stain_num,
        bs=block_size,
        level=unet_level
    )

    # At this stage the model produces output for the whole region in the input
    # image, but we will focus only on the center of that region during
    # training. Here we drop the predictions at the edges.
    output = cntk.reshape(model, (num_landcover_classes*stain_num, int(block_size_m), int(block_size_m)))
    probs = output#cntk.reshape(cntk.softmax(output, axis=0),
                         #(num_landcover_classes*5, int(block_size_m), int(block_size_m)))

    # Now we calculate the supre-res loss. Note that this loss function has the
    # potential to become negative since the variance is fractional.
    # Additionally, we need to make sure that when the nlcd mask[0, ...]
    # is always 1, which means that there's no nlcd label everywhere,
    # the supre_res_loss comes out as a constant.
    super_res_crit = 0
    high_res_crit=0
    each_stain_out_dim=num_landcover_classes
    # Not considering nlcd class 0
    for stain_color in range(start_stain,start_stain+stain_num):
        mask_size = cntk.reshape(
                    cntk.reduce_sum(cntk.slice(mask, 0, stain_color*2, stain_color*2+2)), (1,)
                ) + 1.0
        probs_sliced=cntk.slice(probs, 0, (stain_color-start_stain)*2, (stain_color-start_stain)*2+2)
        probs_sliced=cntk.reshape(cntk.softmax(probs_sliced, axis=0),(num_landcover_classes, int(block_size_m), int(block_size_m)))

        for nlcd_id in range(2):#range(num_of_stain*num_nlcd_classes,(num_of_stain+1)*num_nlcd_classes)
            # for no uppooling /16
            # c_mask: the low res label mask layer, if it has this label, this layer would be all 1, otherwise 0.
            c_mask = cntk.reshape(cntk.slice(mask, 0, 2*stain_color+nlcd_id, 2*stain_color+nlcd_id+1),(1, int(block_size_m), int(block_size_m)))
            #c_mask_size: total pixels of this class, if this layer is not the low res label, it would be 0.
            c_mask_size = cntk.reshape(cntk.reduce_sum(c_mask), (1,)) + 0.000001
            c_interval_center = cntk.reshape(cntk.slice(interval_center, 0, 2*stain_color+nlcd_id, 2*stain_color+nlcd_id+1),
                                             (each_stain_out_dim, ))
            c_interval_radius = cntk.reshape(cntk.slice(interval_radius, 0, 2*stain_color+nlcd_id, 2*stain_color+nlcd_id+1),
                                             (each_stain_out_dim, ))

            # For each nlcd class, we have a landcover distribution:
            #probs=probs[num_of_stain:num_of_stain+1]
            masked_probs = probs_sliced * c_mask
            # Mean mean of predicted distribution
            mean = cntk.reshape(cntk.reduce_sum(masked_probs, axis=(1, 2)),
                                (each_stain_out_dim, )) / c_mask_size
            # Mean var of predicted distribution
            var  = cntk.reshape(cntk.reduce_sum(masked_probs * (1.-masked_probs), axis=(1, 2)),
                                (each_stain_out_dim, )) / c_mask_size
            c_super_res_crit = cntk.square(ddist(mean, c_interval_center, c_interval_radius)) / (
                                var / c_mask_size + c_interval_radius * c_interval_radius + 0.000001) \
                            + cntk.log(var + 0.03)

            super_res_crit += c_super_res_crit * c_mask_size / mask_size * super_res_class_weight[nlcd_id]
            #high res loss
        log_probs = cntk.log(probs_sliced)
        lc_slice=cntk.slice(lc, 0, (stain_color-start_stain)*2, (stain_color-start_stain)*2+2)
        high_res_crit += cntk.times([1.0,]*2,
                cntk.element_times(-cntk.element_times(log_probs, lc_slice), lc_weight_map),
                                    output_rank=2)
            # Average across spatial dimensions
            # Sum over all landcover classes, only one of the landcover classes is non-zero

    # Weight super_res loss according to the ratio of unlabeled LC pixels
    #super_res_loss = cntk.reduce_sum(super_res_crit) * cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
    super_res_loss = cntk.reduce_sum(super_res_crit) #* cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
    high_res_loss = cntk.reduce_mean(high_res_crit)
    '''
    log_probs = cntk.log(probs)
    high_res_crit = cntk.times([0.0,]+[1.0,]*2,
            cntk.element_times(-cntk.element_times(log_probs, lc), lc_weight_map),
                                output_rank=2)
    # Average across spatial dimensions
    # Sum over all landcover classes, only one of the landcover classes is non-zero
    high_res_loss = cntk.reduce_mean(high_res_crit)
    '''
    #high_res_loss=0
    loss = super_res_loss_weight * super_res_loss + high_res_loss_weight * high_res_loss

    return input_im, lc, lc_weight_map, mask, interval_center, interval_radius, \
           output, high_res_loss_weight * high_res_loss, loss


def get_model(f_dim, c_dim, l_dim, m_dim, num_stack_layers,
        super_res_class_weight, super_res_loss_weight, high_res_loss_weight,unet_level,stain_num=1,start_stain=0):
    # Define the variables into which the minibatch data will be loaded.
    num_nlcd_classes, num_landcover_classes = c_dim
    _, block_size, _ = f_dim
    _,block_size_m,_=l_dim
    input_im = cntk.input_variable(f_dim, np.float32)
    lc = cntk.input_variable((6,block_size_m, block_size_m), np.float32)
    lc_weight_map = cntk.input_variable((1, l_dim[1], l_dim[2]), np.float32)
    interval_center = cntk.input_variable((5,), np.float32)
    interval_radius = cntk.input_variable((5,), np.float32)
    mask = cntk.input_variable((5,block_size_m, block_size_m), np.float32)

    # Create the model definition. c_map defines the number of filters trained
    # at layers of different depths in the model. num_stack_layers defines the
    # number of (modified) residual units per layer.
    model = cnn_model(
        input_tensor=input_im,
        num_stack_layers=num_stack_layers,
        c_map=[64, 32, 32, 32, 32],
        num_classes=6,
        bs=block_size,
        level=unet_level
    )

    # At this stage the model produces output for the whole region in the input
    # image, but we will focus only on the center of that region during
    # training. Here we drop the predictions at the edges.
    output = cntk.reshape(model, (6, block_size_m, block_size_m))
    probs = cntk.reshape(cntk.softmax(output, axis=0),
                         (6, block_size_m, block_size_m))

    # Now we calculate the supre-res loss. Note that this loss function has the
    # potential to become negative since the variance is fractional.
    # Additionally, we need to make sure that when the nlcd mask[0, ...]
    # is always 1, which means that there's no nlcd label everywhere,
    # the supre_res_loss comes out as a constant.
    super_res_loss=0
    if super_res_loss_weight!=0:
        super_res_crit = 0
        mask_size = cntk.reshape(
                        cntk.reduce_sum(mask), (1,)
                    ) + 10.0
        # Not considering nlcd class 0
        nlcd_id=1
        # for no uppooling /16
        c_mask = cntk.reshape(cntk.slice(mask, 0, nlcd_id, nlcd_id+1),(1, block_size_m, block_size_m))

        c_mask_size = cntk.reshape(cntk.reduce_sum(c_mask), (1,)) + 0.000001
        c_interval_center = cntk.reshape(interval_center,
                                         (5, ))
        c_interval_radius = cntk.reshape(interval_radius,
                                         (5, ))

        # For each nlcd class, we have a landcover distribution:
        masked_probs = cntk.slice(probs, 0, 0, 5)
        # Mean mean of predicted distribution
        mean = cntk.reshape(cntk.reduce_sum(masked_probs, axis=(1, 2)),
                            (5, )) / c_mask_size
        # Mean var of predicted distribution
        var  = cntk.reshape(cntk.reduce_sum(masked_probs * (1.-masked_probs), axis=(1, 2)),
                            (5, )) / c_mask_size
        c_super_res_crit = cntk.square(ddist(mean, c_interval_center, c_interval_radius)) / (
                            var / c_mask_size + c_interval_radius * c_interval_radius + 0.000001) \
                        + cntk.log(var + 0.03)
        super_res_crit += c_super_res_crit * super_res_class_weight[nlcd_id]

        # Weight super_res loss according to the ratio of unlabeled LC pixels
        #super_res_loss = cntk.reduce_sum(super_res_crit) * cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
        super_res_loss = cntk.reduce_sum(super_res_crit) #* cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
    high_res_loss=0
    if high_res_loss_weight!=0:
        log_probs = cntk.log(probs)
        high_res_crit = cntk.times([0.99,0.99,0.99,0.99,0.99,0.01],
                cntk.element_times(-cntk.element_times(log_probs, lc), lc_weight_map),
                                    output_rank=2)
        # Average across spatial dimensions
        # Sum over all landcover classes, only one of the landcover classes is non-zero
        high_res_loss = cntk.reduce_mean(high_res_crit)


    loss = super_res_loss_weight * super_res_loss + high_res_loss_weight * high_res_loss

    return input_im, lc, lc_weight_map, mask, interval_center, interval_radius, \
           output, high_res_loss_weight * high_res_loss, loss
def get_wrapped_model(f_dim, c_dim, l_dim, m_dim, num_stack_layers,
        super_res_class_weight, super_res_loss_weight, high_res_loss_weight,unet_level,stain_num=1,start_stain=0,channel_times=6):

        num_color_channels, block_size, _=f_dim
        lc_6_layer, block_size_m, _=l_dim
        num_nlcd_classes, num_landcover_classes=c_dim

        input_im_stack = cntk.input_variable((num_color_channels*channel_times, block_size, block_size), np.float32)
        lc_stack = cntk.input_variable((lc_6_layer*channel_times,block_size_m, block_size_m), np.float32)
        interval_center_stack = cntk.input_variable((stain_num*channel_times,), np.float32)
        interval_radius_stack = cntk.input_variable((stain_num*channel_times,), np.float32)
        ##########################
        lc_weight_map = cntk.input_variable((1, l_dim[1], l_dim[2]), np.float32)
        mask = cntk.input_variable((stain_num,block_size_m, block_size_m), np.float32)
        ################
        input_im_slice=[]
        lc_slice=[]
        interval_center_slice=[]
        interval_radius_slice=[]
        for iter in range(channel_times):
            #######slice input
            input_im_slice.append(cntk.slice(input_im_stack,0,iter*num_color_channels,(iter+1)*num_color_channels))
            #lc_slice.append(cntk.slice(lc_stack,0,iter*lc_6_layer,(iter+1)*lc_6_layer))
            interval_center_slice.append(cntk.slice(interval_center_stack,0,iter*stain_num,(iter+1)*stain_num))
            interval_radius_slice.append(cntk.slice(interval_radius_stack,0,iter*stain_num,(iter+1)*stain_num))
            ##sub_stream

            ########feed to model
        for iter in range(channel_times):
            ########feed to model
            if iter==0:
                probs=get_sub_model(input_im_slice[iter],  l_dim,  num_stack_layers, unet_level,block_size,block_size_m,stain_num)
                test_output=probs
                probs_stack=cntk.slice(probs, 0, 0, stain_num)
                probs_stack1=cntk.slice(probs, 0, 0, stain_num+1)
                lc_stack_dim1=cntk.slice(lc_stack,0,iter*lc_6_layer,(iter+1)*lc_6_layer)
                mu_probs = cntk.reshape(cntk.slice(cntk.reduce_mean(probs, axis=(1,2)),0,0,stain_num), (stain_num,1 ))
                var_probs = cntk.reshape(cntk.slice(cntk.element_divide(cntk.reduce_sum(cntk.element_times(probs,1.0-probs), axis=(1, 2)),block_size_m**4),0,0,stain_num),(stain_num,1 ))

            else:
                probs=probs.clone(method  = 'share',substitutions={input_im_slice[iter-1].output:input_im_slice[iter].output})
                probs_stack=cntk.splice(probs_stack,cntk.slice(probs, 0, 0, stain_num),axis=1)
                probs_stack1=cntk.splice(probs_stack1,cntk.slice(probs, 0, 0, stain_num+1),axis=1)
                lc_stack_dim1=cntk.splice(lc_stack_dim1,cntk.slice(lc_stack,0,iter*lc_6_layer,(iter+1)*lc_6_layer),axis=1)
                mu_probs = cntk.splice(mu_probs,cntk.reshape(cntk.slice(cntk.reduce_mean(probs, axis=(1,2)),0,0,stain_num), (stain_num,  1 )),axis=1)
                var_probs = cntk.splice(var_probs,cntk.reshape(cntk.slice(cntk.element_divide(cntk.reduce_sum(cntk.element_times(probs,1.0-probs), axis=(1, 2)),block_size_m**4),0,0,stain_num),(stain_num,1 )),axis=1)

        mean_mu=cntk.reshape(cntk.reduce_mean(mu_probs, axis=(1,)),(stain_num,1 ))
        var_mean=cntk.reduce_mean(var_probs,axis=(1,))
        mean2_mean=cntk.reduce_mean(cntk.element_times(mu_probs,mu_probs),axis=(1,))
        VAR=var_mean+mean2_mean-cntk.element_times(mean_mu,mean_mu)
        var_mu=VAR
        mean=mean_mu
        var=var_mu
        super_res_loss=0
        if super_res_loss_weight!=0:
            super_res_crit = 0
            for nlcd_id in range(stain_num):
                mask_size = cntk.reshape(
                                cntk.reduce_sum(mask), (1,)
                            ) + 10.0
                # Not considering nlcd class 0

                # for no uppooling /16
                c_mask = cntk.reshape(cntk.slice(mask, 0, nlcd_id, nlcd_id+1),(1, block_size_m, block_size_m))

                c_mask_size = cntk.reshape(cntk.reduce_sum(c_mask), (1,)) + 0.000001
                c_interval_center = cntk.reshape(cntk.slice(interval_center_slice[0],0,nlcd_id, nlcd_id+1),
                                                 (1, ))
                c_interval_radius = cntk.reshape(cntk.slice(interval_radius_slice[0],0,nlcd_id, nlcd_id+1),
                                                 (1, ))

                # For each nlcd class, we have a landcover distribution:
                #masked_probs = cntk.reshape(cntk.slice(probs_stack,0,nlcd_id, nlcd_id+1),(block_size_m*channel_times, block_size_m))*c_mask
                # Mean mean of predicted distribution
                #mean = cntk.reshape(cntk.reduce_sum(masked_probs, axis=(1, 2)),
                #                    (1, )) / c_mask_size
                # Mean var of predicted distribution
                #var  = cntk.reshape(cntk.reduce_sum(masked_probs * (1.-masked_probs), axis=(1, 2)),
                #                    (1, )) / c_mask_size
                '''
                c_super_res_crit = cntk.square(ddist(mean, c_interval_center, c_interval_radius)) / (var + c_interval_radius * c_interval_radius + 0.000001) \
                                + cntk.log(var*c_mask_size + 0.03)
                '''
                #actual
                c_super_res_crit = cntk.square(ddist(mean, c_interval_center, c_interval_radius))*(var ) / cntk.square(
                                   var + c_interval_radius * c_interval_radius + 0.000001) \
                                   + cntk.log(var + 0.03)

                '''
                #KL
                c_super_res_crit = (cntk.square(ddist(mean, c_interval_center, c_interval_radius))+(var )) / (
                                 c_interval_radius * c_interval_radius + 0.000001) \
                                 - cntk.log(var + 0.03)
                '''
                super_res_crit += c_super_res_crit *c_mask_size/mask_size#* super_res_class_weight[nlcd_id]

                # Weight super_res loss according to the ratio of unlabeled LC pixels
                #super_res_loss = cntk.reduce_sum(super_res_crit) * cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
            super_res_loss = cntk.reduce_sum(super_res_crit) #* cntk.reduce_mean(cntk.slice(lc, 0, 0, 1))
        high_res_loss=0

        if high_res_loss_weight!=0:
            log_probs = cntk.log(probs_stack1)
            high_res_crit = cntk.times([1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1],
                    -cntk.element_times(log_probs, lc_stack_dim1),
                                        output_rank=2)
            # Average across spatial dimensions
            # Sum over all landcover classes, only one of the landcover classes is non-zero
            high_res_loss = cntk.reduce_mean(high_res_crit)


        loss = super_res_loss_weight * super_res_loss + high_res_loss_weight * high_res_loss


        return input_im_stack,lc_stack,lc_weight_map,mask,interval_center_stack,interval_radius_stack,probs,probs_stack1,loss,loss

            ########collect output


        ################loss computing


def get_sub_model(input_im,  l_dim,  num_stack_layers, unet_level,block_size,block_size_m,stain_num):
    # Define the variables into which the minibatch data will be loaded.
    '''
    num_nlcd_classes, num_landcover_classes = c_dim
    _, block_size, _ = f_dim
    _,block_size_m,_=l_dim
    input_im = cntk.input_variable(f_dim, np.float32)
    lc = cntk.input_variable((6,block_size_m, block_size_m), np.float32)
    lc_weight_map = cntk.input_variable((1, l_dim[1], l_dim[2]), np.float32)
    interval_center = cntk.input_variable((5,), np.float32)
    interval_radius = cntk.input_variable((5,), np.float32)
    mask = cntk.input_variable((5,block_size_m, block_size_m), np.float32)
    '''
    # Create the model definition. c_map defines the number of filters trained
    # at layers of different depths in the model. num_stack_layers defines the
    # number of (modified) residual units per layer.
    model = cnn_model(
        input_tensor=input_im,
        num_stack_layers=num_stack_layers,
        c_map=[32, 64, 128, 256, 512],
        num_classes=stain_num+1,
        bs=block_size,
        level=unet_level
    )

    # At this stage the model produces output for the whole region in the input
    # image, but we will focus only on the center of that region during
    # training. Here we drop the predictions at the edges.
    output = cntk.reshape(model, (stain_num+1, block_size_m, block_size_m))
    probs = cntk.reshape(cntk.softmax(output, axis=0),
                         (stain_num+1, block_size_m, block_size_m))



    return probs
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Model structure code
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=True)(input)
    b = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True)(c)
    r = relu(b)
    return r

def pool_block(input, fsize=(2,2), strides=(2,2)):
    p = MaxPooling(fsize, strides=strides, pad=False)(input)
    return p

def conv_block(input, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for i in range(num_stack_layers):
        l = conv_bn_relu(l, (3,3), num_filters)
    return l

def unpl_block(input, output_shape, num_filters, strides, pname):
    #x_dup0 = cntk.reshape(input, (input.shape[0], input.shape[1], 1, input.shape[2], 1))
    #x_dup1 = cntk.splice(x_dup0, x_dup0, axis=-1)
    #x_dup2 = cntk.splice(x_dup1, x_dup1, axis=-3)
    #upsampled = cntk.reshape(x_dup2, (input.shape[0], input.shape[1]*2, input.shape[2]*2))
    #return upsampled
    c = Convolution(
            (1,1), num_filters, activation=cntk.relu,
            init=he_normal(), strides=(1,1), name=pname)(input)
    ct = ConvolutionTranspose(
            (3,3), num_filters, strides=strides,
            output_shape=output_shape, pad=True,
            bias=True, init=bilinear(3,3))(c)
    ctf = cntk.ops.combine([ct]).clone(
            cntk.ops.functions.CloneMethod.freeze,
            {c: cntk.ops.placeholder(name=pname)})(c)
    l = conv_bn_relu(ctf, (3,3), num_filters)
    return l

def merg_block(in1, in2, num_filters):
    #c = Convolution((1, 1), num_filters, activation=None, bias=True)(in1) + \
    #    Convolution((1, 1), num_filters, activation=None, bias=True)(in2)
    #b = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True)(c)
    #r = relu(b)
    #return r
    s = splice(in1, in2, axis=0)
    return s

def shallow_cnn_model(input_tensor, num_stack_layers, c_map, num_classes, bs):
    r1_1 = input_tensor

    r1_2 = conv_block(r1_1, num_stack_layers=8, num_filters=64)
    o1_1 = Convolution((3, 3), num_classes, activation=None, pad=True, bias=True)(r1_2)
    return o1_1

def cnn_model(input_tensor, num_stack_layers, c_map, num_classes, bs, level):
    r1_1 = input_tensor

    r1_2 = conv_block(r1_1, num_stack_layers, c_map[0])

    r2_1 = pool_block(r1_2)
    r2_2 = conv_block(r2_1, num_stack_layers, c_map[1])

    r3_1 = pool_block(r2_2)
    r3_2 = conv_block(r3_1, num_stack_layers, c_map[2])

    r4_1 = pool_block(r3_2)
    r4_2 = conv_block(r4_1, num_stack_layers, c_map[3])

    r5_1 = pool_block(r4_2)
    r5o5 = conv_block(r5_1, num_stack_layers, c_map[4])
    last_conv=r5o5
    if level<16:
        o5_1 = unpl_block(r5o5, (bs/8, bs/8),     c_map[4], 2, 'o5_1')

        o4_3 = merg_block(o5_1, r4_2,             c_map[3])
        o4_2 = conv_block(o4_3, num_stack_layers, c_map[3])
        last_conv=o4_2
        if level<8:
            o4_1 = unpl_block(o4_2, (bs/4, bs/4),     c_map[3], 2, 'o4_1')

            o3_3 = merg_block(o4_1, r3_2,             c_map[2])
            o3_2 = conv_block(o3_3, num_stack_layers, c_map[2])
            last_conv=o3_2
            if level<4:
                o3_1 = unpl_block(o3_2, (bs/2, bs/2),     c_map[2], 2, 'o3_1')

                o2_3 = merg_block(o3_1, r2_2,             c_map[1])
                o2_2 = conv_block(o2_3, num_stack_layers, c_map[1])
                last_conv=o2_2
                if level<2:
                    o2_1 = unpl_block(o2_2, (bs/1, bs/1),     c_map[1], 2, 'o2_1')

                    o1_3 = merg_block(o2_1, r1_2,             c_map[0])
                    o1_2 = conv_block(o1_3, num_stack_layers, c_map[0])
                    last_conv=o1_2
                    #o1_1 = Convolution((3, 3), num_classes, activation=None, pad=True, bias=True)(o1_2)

    o1_1 = Convolution((3, 3), num_classes, activation=None, pad=True, bias=True)(last_conv)
    #return o1_1
    return o1_1

