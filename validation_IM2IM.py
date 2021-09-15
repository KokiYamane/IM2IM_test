import numpy as np
import torch
from torchvision import transforms
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


def load_image(image_path, image_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])

    if not os.path.exists(image_path):
        return
    image = Image.open(image_path)
    image = transform(image)
    return image


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


def load_model_param(filepath, device='cpu'):
    state_dict = torch.load(filepath, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def main():
    # pytorch device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # folder = 'model_param/20210808_194702'
    folder = 'model_param/20210808_235751'
    # ModelDirName = '{}/model_param_best.pt'.format(folder)
    ModelDirName = '{}/model_param/model_param_{:06}.pt'.format(folder, 1500)
    # ModelDirName = 'model/model_param/model_param_000600.pt'
    model = IM2IM(
        state_dim=9,
        image_feature_dim=15,
        LSTM_dim=200,
        LSTM_layer_num=5,
    )
    state_dict = load_model_param('./model_param/IM2IM_param.pt')
    # model = SPAN(
    #     state_dim=9,
    #     image_feature_dim=15,
    #     LSTM_dim=200,
    #     LSTM_layer_num=5,
    # )
    # state_dict = load_model_param('./model_param/SPAN_param.pt')
    model.load_state_dict(state_dict, device)
    model.eval()
    model.to(device)

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
                state = data[:state_dim]
                # state = np.tile(data[:state_dim], 2)
                print('state shape:', state.shape)
                # image = None
                # while image == None:
                image = load_image('../repro/video_rgb0/rgb{:.3f}.jpg'.format(data[state_dim]))
                print('image shape:', image.shape)

                # prediction
                state = (state - mean) / std

                state = torch.from_numpy(state.astype(np.float32)).to(device)
                image = image.to(device)

                state = state.unsqueeze(0).unsqueeze(0)
                image = image.unsqueeze(0).unsqueeze(0)

                print('state shape:', state.shape)
                print('image shape:', image.shape)

                state_hat, image_hat, (h, c) = model(state, image, (h, c))
                # state_hat, image_hat, _, _, (h, c) = model(state, image, (h, c))

                print(h, c)

                state_hat = state_hat.cpu().detach().numpy().flatten()
                state_hat = state_hat * std + mean
                print('state_hat:', state_hat)

                image = image.squeeze().cpu().detach().numpy()
                image = image.transpose(1, 2, 0)
                image_hat = image_hat.squeeze().cpu().detach().numpy()
                image_hat = image_hat.transpose(1, 2, 0)
                image_hat = np.concatenate([image, image_hat], axis=1)
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
        
    # except:
    #     print('error')
    #     for sock in readfds:
    #         sock.close()

    finally:
        for sock in readfds:
            sock.close()


if __name__ == '__main__':
      main()
