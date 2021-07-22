import numpy as np
import torch
import socket
import select
from IM2IM import IM2IM
from PIL import Image
import os


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
    image = image.resize((128, 128))
    return np.array(image)


def main():
    # pytorch device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    ModelDirName = 'model/20210722_172206/model_param_best.pt'
    # ModelDirName = 'model/model_param/model_param_000600.pt'
    model = IM2IM(state_dim=18, image_feature_dim=15)
    model.load_state_dict(torch.load(ModelDirName, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    mean = np.loadtxt('norm_mean.csv', delimiter=',', dtype=np.float32)
    std = np.loadtxt('norm_std.csv', delimiter=',', dtype=np.float32)

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

                # dd = []
                # smsg = msg.split()
                # for m in smsg:
                #     dd.append(float(m))
                # InputData = torch.from_numpy(data)
                # state = InputData[:len(mean)]
                # image = InputData[len(mean):].reshape(3, 128, 128)
                print(msg)
                data = np.fromstring(msg, dtype=np.float32 , sep=' ')
                state_dim = 9
                state = np.tile(data[:state_dim], 2)
                print('state shape:', state.shape)
                image = load_image('../repro/video_rgb0/rgb{:.3f}.jpg'.format(data[state_dim]))
                image = image.transpose(2, 0, 1) / 256
                print('image shape:', image.shape)

                # prediction
                state = (state - mean) / std
                state = state[np.newaxis, np.newaxis, :]
                image = image[np.newaxis, np.newaxis, :, :, :]

                state = torch.from_numpy(state.astype(np.float32)).to(device)
                image = torch.from_numpy(image.astype(np.float32)).to(device)

                state_hat, image_hat = model(state, image)
                
                state_hat = state_hat.cpu().detach().numpy().flatten()
                state_hat = state_hat * std + mean
                print('state_hat:', state_hat)

                # submit data
                msg = ''
                for y_element in state_hat[:9]:
                    msg += str(y_element.item()) + ','
                msg = bytes(msg, encoding='utf8')
                sock.send(msg)
                print('msg:', msg)

    finally:
        for sock in readfds:
            sock.close()


if __name__ == '__main__':
      main()
