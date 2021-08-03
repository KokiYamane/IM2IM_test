import numpy as np
import torch
import socket
import select
import sys
sys.path.append('.')
sys.path.append('..')
from model.IM2IM import IM2IM
from model.SPAN import SPAN
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                        (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                        (img_height + crop_height) // 2))


def load_image(image_path):
    if not os.path.exists(image_path):
        return
    image = Image.open(image_path)
    image = crop_center(image, min(image.size), min(image.size))
    image = image.resize((64, 64))
    return np.array(image)


def imageNormalizaion(image):
    channel, width, height = image.shape
    image = image.reshape(channel, -1)
    channel_mean = np.mean(image, axis=1, keepdims=True)
    channel_std = np.std(image, axis=1, keepdims=True)
    image = (image - channel_mean) / channel_std
    image = image * 0.25 + 0.5
    image = image.clip(0, 1)
    image = image.reshape(channel, width, height)
    return image


def main():
    # pytorch device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # folder = 'model/20210723_214631'
    folder = 'model_param/20210803_203023'
    ModelDirName = '{}/model_param_best.pt'.format(folder)
    # ModelDirName = '{}/model_param/model_param_003000.pt'.format(folder)
    # ModelDirName = 'model/model_param/model_param_000600.pt'
    # model = IM2IM(state_dim=18, image_feature_dim=15)
    model = SPAN(state_dim=18, image_feature_dim=15)
    state_dict = torch.load(ModelDirName, map_location=torch.device(device))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    h = torch.zeros((model.LSTM_layer_num, 1, model.LSTM_dim)).to(device)
    c = torch.zeros((model.LSTM_layer_num, 1, model.LSTM_dim)).to(device)

    mean = np.loadtxt('{}/norm_mean.csv'.format(folder), delimiter=',', dtype=np.float32)
    std = np.loadtxt('{}/norm_std.csv'.format(folder), delimiter=',', dtype=np.float32)

    print('load {}'.format(ModelDirName))
    print('/////////////////////////////////////////////////')
    print('/               please run repro                /')
    print('/////////////////////////////////////////////////')

    # socket setting
    host = '127.0.0.1'
    port = 10051
    backlog = 10
    bufsize = 4096
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    readfds = set([server_sock])

    try:
        server_sock.bind((host, port))
        server_sock.listen(backlog)

        while True:
            rready, wready, xready = select.select(readfds, [], [])

            for sock in rready:
                if sock is server_sock:
                    conn, address = server_sock.accept()
                    readfds.add(conn)
                    continue

                # receive data
                msg = sock.recv(bufsize)

                if len(msg) == 0:
                    sock.close()
                    readfds.remove(sock)
                    continue

                if msg == '**':
                    print('finish socket')
                    break

                print(msg)
                data = np.fromstring(msg, dtype=np.float32 , sep=' ')
                state_dim = 9
                state = np.tile(data[:state_dim], 2)
                print('state shape:', state.shape)
                # image = None
                # while image == None:
                image = load_image('../repro/video_rgb0/rgb{:.3f}.jpg'.format(data[state_dim]))
                image = image.transpose(2, 0, 1) / 256
                print('image shape:', image.shape)

                # prediction
                state = (state - mean) / std
                state = state[np.newaxis, np.newaxis, :]

                image = imageNormalizaion(image)
                image = image[np.newaxis, np.newaxis, :, :, :]

                state = torch.from_numpy(state.astype(np.float32)).to(device)
                image = torch.from_numpy(image.astype(np.float32)).to(device)

                state_hat, image_hat, _, _, (h, c) = model(state, image, (h, c))

                print(h, c)

                state_hat = state_hat.cpu().detach().numpy().flatten()
                state_hat = state_hat * std + mean
                print('state_hat:', state_hat)

                image_hat = image_hat.cpu().detach().numpy()[0, 0]
                image_hat = image_hat.transpose(1, 2, 0)
                image_hat_pil = Image.fromarray((225 * image_hat).astype(np.uint8), mode=None)

                os.makedirs('../repro/video_pred0/', exist_ok=True)
                image_hat_pil.save('../repro/video_pred0/pred{:.3f}.jpg'.format(data[state_dim]))

                # submit data
                msg = ''
                for y_element in state_hat[:9]:
                    msg += str(y_element.item()) + ','
                msg = bytes(msg, encoding='utf8')
                sock.send(msg)
                print('msg:', msg)
        
    except:
        for sock in readfds:
            sock.close()

    finally:
        for sock in readfds:
            sock.close()


if __name__ == '__main__':
      main()
