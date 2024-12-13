import logging

logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'TSAIN':
        from .model import TSAIN as M
    elif model == 'TIRN':
        from .model import TIRN as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))

    return m
