import logging
import math
import torch
from models.modules.inv_arch import TIRN_2,TIRN_4,TSAIN_2,TSAIN_4
from models.modules.subnet_constructor import subnet
logger = logging.getLogger('base')


####################
# define network
####################

def define(opt):
    opt_net = opt['network']
    subnet_type = opt_net['subnet']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'
    down_num = int(math.log(opt_net['scale'], 2))
    net_0 = TIRN_2(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], down_num)
    net_1 = TIRN_4(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], opt_net['v_blocks'], down_num)
    net_2 = TSAIN_2(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], opt_net['v_blocks'], down_num)
    net_3 = TSAIN_4(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], opt_net['v_blocks'], down_num,opt_net['f_blocks'])
    return net_0,net_1,net_2,net_3


