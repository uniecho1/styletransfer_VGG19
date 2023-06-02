import model
import utils
import extract
import train
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn

path_to_content_img = "content_img.jpg"
path_to_style_img = "style_img.jpg"

content_image = Image.open(path_to_content_img)
style_image = Image.open(path_to_style_img)

# print(type(content_image))

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

device = utils.try_gpu()
image_shape = (content_image.size[1]//2, content_image.size[0]//2)
net = model.getnet(max(style_layers + content_layers)+1, device)

X = utils.preprocess(content_image, image_shape).to(device)
contents_Y = extract.get_contents(
    net, content_image, image_shape, content_layers, style_layers, device)
styles_Y = extract.get_styles(
    net, style_image, image_shape, content_layers, style_layers, device)


output = train.train(net, X, content_layers, style_layers, contents_Y,
                     styles_Y, device, 0.3, 500, 50)
plt.axis("off")

plt.imshow(utils.postprocess(output))
plt.savefig("outputImage.png", bbox_inches='tight', pad_inches=0)
plt.show()
