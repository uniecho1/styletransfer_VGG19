import torchvision
import utils
from torch import nn


def extract_features(net, X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(net, image, image_shape, content_layers, style_layers, device):
    content_X = utils.preprocess(image, image_shape).to(device)
    contents_Y, _ = extract_features(net,
                                     content_X, content_layers, style_layers)
    return contents_Y


def get_styles(net, image, image_shape, content_layers, style_layers, device):
    style_X = utils.preprocess(image, image_shape).to(device)
    _, styles_Y = extract_features(net,
                                   style_X, content_layers, style_layers)
    return styles_Y
