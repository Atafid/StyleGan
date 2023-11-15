import torch
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from super_image import ImageLoader, EdsrModel
from labml import experiment, lab

from src.configs import Configs


def train_new_model():
    experiment.create(name="waifu-gan2")

    configs = Configs()

    experiment.configs(configs, {
        'device.cuda_device': 0,
        'img_size': 64,
        'log_generated_interval': 400
    })

    configs.init(str(lab.get_data_path() / 'stylegan'))

    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with (experiment.start()):
        configs.train()


def train_existing_model():
    experiment.create(name="waifu-gan")

    configs = Configs()

    experiment.configs(configs, {
        'device.cuda_device': 0,
        'img_size': 64,
        'log_generated_interval': 200
    })

    configs.init(str(lab.get_data_path() / 'stylegan'))

    configs.discriminator.load_state_dict(
        torch.load("best_model/discriminator.pth"))
    configs.mapping_network.load_state_dict(
        torch.load("best_model/mapping_network.pth"))
    configs.generator.load_state_dict(torch.load("best_model/generator.pth"))

    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with (experiment.start()):
        configs.train()


def convert_generated_images(img):
    img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    img = np.clip(img, 0, 1)

    pil_image = Image.fromarray((img*255).astype("uint8"))

    return (pil_image)


def upscale_image(img, scale):
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    inputs = ImageLoader.load_image(img)
    scaled_img = model(inputs)

    pil_scaled_img = convert_generated_images(scaled_img[0])

    return (pil_scaled_img)


def generate_new_images(configs):

    img, w = configs.generate_images(configs.batch_size)

    pil_images = list(map(convert_generated_images, img))
    # pil_scaled_images = list(
    #    map(lambda img: upscale_image(img, 1), pil_images))

    # for i in range(len(pil_scaled_images)):
    # pil_scaled_images[i].save("generated/"+str(i)+".jpg")
    for i in range(len(pil_images)):
        pil_images[i].save("generated/"+str(i)+".jpg")


def load_images(canvas, batch_size):
    global tk_images

    tk_images = []

    for i in range(batch_size):
        tk_image = ImageTk.PhotoImage(file="generated/"+str(i)+".jpg")
        tk_images.append(tk_image)

        row = i//4
        col = i % 4

        canvas.create_image(col*64+32, row*64+32, image=tk_images[i])


def update_images(configs, canvas, nb_img_display):
    generate_new_images(configs)
    load_images(canvas, 2**(int(nb_img_display)-1))


def change_nb_images(text, canvas, v):
    nb_img = 2**(int(v)-1)

    text["text"] = str(nb_img)
    if (nb_img == 16):
        text["text"] = "12"

    load_images(canvas, nb_img)


def init_generation():

    configs = Configs()
    configs.init(str(lab.get_data_path() / 'stylegan'))
    configs.mapping_network.load_state_dict(
        torch.load("best_model/mapping_network.pth"))
    configs.generator.load_state_dict(torch.load("best_model/generator.pth"))

    root = tk.Tk()
    root.geometry("512x350")

    canvas = tk.Canvas(root, width=256, height=256, bg="white")
    canvas.pack()

    nb_img_display = tk.DoubleVar(value=1)

    text = tk.Label(root, text=str(int(2**(nb_img_display.get()-1))))
    scale = tk.Scale(root, variable=nb_img_display, from_=1, to=5,
                     orient=tk.HORIZONTAL, showvalue=0, command=lambda nb_img_display: change_nb_images(text, canvas, nb_img_display))
    scale.pack()
    text.pack()

    boutton = tk.Button(root, text="generate", command=lambda: update_images(
        configs, canvas, nb_img_display.get()))
    boutton.pack()

    update_images(configs, canvas, nb_img_display.get())

    def on_closing():
        root.quit()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    return (root, canvas)
