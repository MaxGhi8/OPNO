"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer
from utilities import *
import h5py
import fourierpack as sp
import functools
from functools import partial as PARTIAL
from chebypack import CGL_points
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
import matplotlib

device = torch.device("cuda:0")
data_name = '2d-CahnHilliard'

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from utilities import get_args
import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='/', type=str, help='dataset folder')
    parser.add_argument('--epochs', default=500, type=int, help='training iterations')
    parser.add_argument('--sub', default=4, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=3, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=100, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=16, type=int, help='Fourier-like modes')
    parser.add_argument('--width', default=24, type=int, help='channel width')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--scdl', default='plat', type=str, help='')
    parser.add_argument('--sub-t', default=1, type=int, help='')
    parser.add_argument('--init-t', default=1, type=int, help='')
    return parser.parse_args()

#### parameters settings
args = get_args()

epochs =  args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 50
batch_size = args.batch_size  # default 20
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
triL = args.triL
scdl = args.scdl
sub_t = args.sub_t
suffix = args.suffix
initial_step = args.init_t
width = args.width

gamma = 0.5  # for StepLR
weight_decay = 1e-4
num_workers = 0

end_time = 1 + 5
t_train = (end_time - 1) // sub_t + 1
train_size, test_size = 5900, 100
ntrain, ntest = train_size, test_size

if sys.argv[0][:5] == '/home':
    print('------PYCHARM test--------')
    epochs = 0
    sub = 4

data_PATH = args.data_dict + data_name + '.mat'

file_name = 'opno-' + data_name + '-al-' + str(sub) + '-modes' + str(modes) + '-width' + str(width) + \
    '-end' + str(end_time) + '-ntrain' + str(ntrain)
result_PATH = '/home/father/OPNO/model/new/' + file_name + '.pkl'

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)

import os

if os.path.exists(result_PATH):
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)

## main

## model

raw_data = h5py.File(data_PATH, 'r')

x_data, y_data = np.array(raw_data['u_cgl'], dtype=np.float32), np.array(raw_data['u_cgl'], dtype=np.float32)
x_data = x_data[..., 0:1]
x_data, y_data = torch.from_numpy(x_data),  torch.from_numpy(y_data)
y_data = y_data[..., :end_time]

x_train, x_test = x_data[:ntrain,::sub,::sub, ...], x_data[-ntest:,::sub,::sub, ...]
y_train, y_test = y_data[:ntrain,::sub,::sub, ...], y_data[-ntest:,::sub,::sub, ...]

_, Nx, Ny, _ = x_train.shape

grid_x = CGL_points(Nx).view(1, Nx, 1, 1)
grid_y = CGL_points(Nx).view(1, 1, Ny, 1)
grid = torch.cat([grid_x.repeat(batch_size, 1, Ny, 1), grid_y.repeat(batch_size, Nx, 1, 1)], dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False)

from model import OPNO_Neumann2d
model = OPNO_Neumann2d(initial_step+2, modes, width, bandwidth, out_channels=1, triL=triL).to(device)

if epochs == 0:  # load model
    print('model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    print('test_l2:', loader['loss_list'][-1])
print('model parameters number =', count_params(model))

training_type = 'autoregressive'

from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scdl == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=50, verbose=True)

train_list, loss_list = [], []
t1 = default_timer()

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0

        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        # Initialize the prediction tensor
        pred = yy[..., 0:1]
        # Extract shape of the input tensor for reshaping (i.e. stacking the
        # time and channels dimension together)

        if training_type in ['autoregressive']:
            # Autoregressive loop
            for t in range(1, end_time):
                # Extract target at current time step
                y = yy[..., t:t + 1]

                # Model run
                im = model(torch.cat([xx, grid], dim=-1))

                # Loss calculation
                _batch = im.size(0)
                loss += myloss(im.reshape(_batch, -1), y.reshape(_batch, -1))

                # Concatenate the prediction at current time step into the
                # prediction tensor
                pred = torch.cat((pred, im), -1)

                # Concatenate the prediction at the current time step to be used
                # as input for the next time step
                xx = im

            train_l2_step += loss.item()
            _batch = yy.size(0)
            _yy = yy[..., initial_step:t_train]  # if t_train is not -1
            _pred = pred[..., initial_step:t_train]
            l2_full = myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    train_l2 = train_l2_full / ntrain
    if scdl == 'step':
        scheduler.step()
    else:
        scheduler.step(train_l2)

    if True:
        val_l2_step = 0
        val_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                if training_type in ['autoregressive']:
                    pred = yy[..., 0:1]

                    for t in range(1, end_time):
                        y = yy[..., t:t + 1]
                        # im = model(inp, grid)
                        im = model(torch.cat([xx, grid], dim=-1))
                        _batch = im.size(0)
                        loss += myloss(im.reshape(_batch, -1), y.reshape(_batch, -1))

                        pred = torch.cat((pred, im), -1)

                        xx = im

                    val_l2_step += loss.item()
                    _batch = yy.size(0)
                    _pred = pred[..., initial_step:t_train]
                    _yy = yy[..., initial_step:t_train]
                    val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

    test_l2 = val_l2_full / ntest
    train_list.append(train_l2)
    loss_list.append(test_l2)

    t2 = default_timer()
    if (ep + 1) % 10 == 0 or ep < 30:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_l2, test_l2)

if epochs >= 200:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)

j = -1

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([])
model.eval()
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          num_workers=num_workers,
                                          shuffle=False)
grid = grid[0:1, ...].to(device)
with torch.no_grad():

    for xx, yy in test_loader:
        val_l2_step = 0
        val_l2_full = 0
        loss = 0
        xx, yy = xx.to(device), yy.to(device)

        if training_type in ['autoregressive']:
            pred = yy[..., 0:1]

            for t in range(1, end_time):
                y = yy[..., t:t + 1]
                im = model(torch.cat([xx, grid], dim=-1))
                _batch = im.size(0)
                loss += myloss(im.reshape(_batch, -1), y.reshape(_batch, -1))

                pred = torch.cat((pred, im), -1)

                xx = im

            val_l2_step += loss.item()
            _batch = yy.size(0)
            _pred = pred[..., initial_step:t_train]
            _yy = yy[..., initial_step:t_train]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
            # print(val_l2_full)
            test_err = torch.cat([test_err,
                              torch.tensor([val_l2_full])],
                             dim=0)

print('test_l2', test_err.sum().item() / test_size)
print('test_l2 min-max:', test_err.min().item(), test_err.max().item())


PRE_PLT

sub = 1
end_time = 1 + 10

x_data, y_data = np.array(raw_data['u_cgl'], dtype=np.float32), np.array(raw_data['u_cgl'], dtype=np.float32)
x_data = x_data[..., 0:1]
y_data = y_data[..., :end_time]
x_data, y_data = torch.from_numpy(x_data),  torch.from_numpy(y_data)
t_train = (end_time - 1) // sub_t + 1

x_test = x_data[-ntest:,::sub,::sub, ...]
y_test = y_data[-ntest:,::sub,::sub, ...]

_, Nx, Ny, _ = x_test.shape

grid_x = CGL_points(Nx).view(1, Nx, 1, 1)
grid_y = CGL_points(Nx).view(1, 1, Ny, 1)
grid = torch.cat([grid_x.repeat(1, 1, Ny, 1), grid_y.repeat(1, Nx, 1, 1)], dim=-1)

X, Y = grid[0, ..., 0].cpu(), grid[0, ..., 1].cpu()

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([])
model.eval()
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          num_workers=num_workers,
                                          shuffle=False)
grid = grid[0:1, ...].to(device)
with torch.no_grad():

    for xx, yy in test_loader:
        val_l2_step = 0
        val_l2_full = 0
        loss = 0
        xx, yy = xx.to(device), yy.to(device)

        if training_type in ['autoregressive']:
            pred = yy[..., 0:1]

            for t in range(1, end_time):
                y = yy[..., t:t + 1]
                im = model(torch.cat([xx, grid], dim=-1))
                _batch = im.size(0)
                loss += myloss(im.reshape(_batch, -1), y.reshape(_batch, -1))

                pred = torch.cat((pred, im), -1)

                xx = im

            val_l2_step += loss.item()
            _batch = yy.size(0)
            _pred = pred[..., initial_step:t_train]
            _yy = yy[..., initial_step:t_train]
            val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
            # print(val_l2_full)
            test_err = torch.cat([test_err,
                              torch.tensor([val_l2_full])],
                             dim=0)

print('test_l2', test_err.sum().item() / test_size)
print('test_l2 min-max:', test_err.min().item(), test_err.max().item())
j = -1

PLT

with torch.no_grad():
    j += 1
    xx, yy = x_test[j:j+1, ...], y_test[j:j+1, ...]
    x = xx.cpu().clone()
    val_l2_step = 0
    val_l2_full = 0
    loss = 0
    xx, yy = xx.to(device), yy.to(device)

    if training_type in ['autoregressive']:
        pred = xx[..., 0:1]

        for t in range(1, end_time):
            y = yy[..., t:t + 1]
            im = model(torch.cat([xx, grid], dim=-1))
            _batch = im.size(0)
            loss += myloss(im.reshape(_batch, -1), y.reshape(_batch, -1))

            pred = torch.cat((pred, im), -1)

            xx = im

        val_l2_step += loss.item()
        _batch = yy.size(0)
        _pred = pred[..., initial_step:t_train]
        _yy = yy[..., initial_step:t_train]
        val_l2_full += myloss(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

yy, pred = yy.cpu(), pred.cpu()
plt.cla()
fig, axs = plt.subplots(4, 6, num=0, clear=True, figsize=(11, 7))

for t in range(1, 6):
    im = axs[0, t].pcolor(X, Y, pred[0, ..., t], cmap='RdBu')
    fig.colorbar(im, ax=axs[1, t], shrink=0.5)
    axs[0, t].set_title("$t=" + str(t)+"$", fontsize=12, loc='center')

    im = axs[1, t].pcolor(X, Y, yy[0, ..., t], cmap='RdBu')
    fig.colorbar(im, ax=axs[0, t], shrink=0.5)

    im = axs[2, t].pcolor(X, Y, pred[0, ..., 5+t], cmap='RdBu')
    fig.colorbar(im, ax=axs[3, t], shrink=0.5)
    axs[2, t].set_title("$t=" + str(5+t)+"$", fontsize=12, loc='center')

    im = axs[3, t].pcolor(X, Y, yy[0, ..., 5+t], cmap='RdBu')
    fig.colorbar(im, ax=axs[2, t], shrink=0.5)

im = axs[0, 0].pcolor(X, Y, x[0, ..., 0].cpu(), cmap='RdBu')
fig.colorbar(im, ax=axs[0, 0], shrink=0.5)
axs[0, 0].set_title("input $t=0$", fontsize=12, loc='center')

for ax in axs.flatten():
    ax.set_axis_off()
    ax.set_aspect('equal')
plt.subplots_adjust(wspace=0.6, hspace=-0.3)
# plt.tight_layout()
axs[0, 1].text(-1.3, 0, 'Prediction', rotation='vertical', va='center', ha='center', fontsize=14)
axs[1, 1].text(-1.3, 0, 'Ground Truth', rotation='vertical', va='center', ha='center', fontsize=14)
axs[2, 1].text(-1.3, 0, 'Prediction', rotation='vertical', va='center', ha='center', fontsize=14)
axs[3, 1].text(-1.3, 0, 'Ground Truth', rotation='vertical', va='center', ha='center', fontsize=14)

savefig(dpi=400)
plt.show()
