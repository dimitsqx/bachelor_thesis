import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA = {
    'name': 'EMNIST CNN B=10 E=10',
    'one': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_100_50_one',
             'server': 100,
             'client': 50},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_100_30_one',
             'server': 100,
             'client': 30}
            ],
    'stream': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_CNN_100_50_stream',
                'server': 100,
                'client': 50},
               {'path': 'model/quic/data/good/10_10_0.01_CNN_100_30_stream',
                'server': 100,
                'client': 30}
               ]}


def plot(data):
    linestyles = ['-', '--', ':', '-.']
    accuracy = {'one': [], 'stream': []}
    loss = {'one': [], 'stream': []}
    for element in data.get('one'):
        with open('{}/history.pickle'.format(element.get('path')), 'rb') as fp:
            dt = pickle.load(fp)
            acc = []
            los = []
            for point in dt:
                inf = point.get('accuracy')
                acc.append(inf[1])
                los.append(inf[0])
            accuracy['one'].append(acc)
            loss['one'].append(los)
    for element in data.get('stream'):
        with open('{}/history.pickle'.format(element.get('path')), 'rb') as fp:
            dt = pickle.load(fp)
            acc = []
            los = []
            for point in dt:
                inf = point.get('accuracy')
                acc.append(inf[1])
                los.append(inf[0])
            accuracy['stream'].append(acc)
            loss['stream'].append(los)

    for index, dt in enumerate(accuracy.get('one')):
        plt.plot(dt,  # linestyle=linestyles[index], color='royalblue',
                 label='Sp={} Cp={}'.format(
                     data.get('one')[index].get('server')/100, data.get('one')[index].get('client')/100))

    for index, dt in enumerate(accuracy.get('stream')):
        plt.plot(dt,  # linestyle=linestyles[index], color='orangered',
                 label='Sp={} Cp={}'.format(
                     data.get('stream')[index].get('server')/100, data.get('stream')[index].get('client')/100))
    plt.legend(loc='lower right', fontsize='small')
    plt.ylabel("Accuracy")
    plt.xlabel("Communication rounds")
    plt.title('{}'.format(data.get('name')))
    plt.show()

    for index, dt in enumerate(loss.get('one')):
        plt.plot(dt,  # linestyle=linestyles[index], color='royalblue',
                 label='Sp={} Cp={}'.format(
                     data.get('one')[index].get('server')/100, data.get('one')[index].get('client')/100))

    for index, dt in enumerate(loss.get('stream')):
        plt.plot(dt,  # linestyle=linestyles[index], color='orangered',
                 label='Sp={} Cp={}'.format(
                     data.get('stream')[index].get('server')/100, data.get('stream')[index].get('client')/100))
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel("Loss")
    plt.xlabel("Communication rounds")
    plt.title('{}'.format(data.get('name')))
    plt.show()


DATA_1 = {
    'name': 'CNN B=10 E=10',
    'one': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_100_50_one',
             'server': 50,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_100_30_one',
             'server': 30,
             'client': 100}
            ],
    'stream': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_CNN_100_50_stream',
                'server': 50,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_CNN_100_30_stream',
                'server': 30,
                'client': 100}
               ]}
DATA_2 = {
    'name': 'CNN B=40 E=10',
    'one': [{'path': 'model/quic/data/good/10_40_0.01_CNN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_40_0.01_CNN_100_50_one',
             'server': 100,
             'client': 50},
            {'path': 'model/quic/data/good/10_40_0.01_CNN_100_30_one',
             'server': 100,
             'client': 30}
            ],
    'stream': [{'path': 'model/quic/data/good/10_40_0.01_CNN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_40_0.01_CNN_100_50_stream',
                'server': 100,
                'client': 50},
               {'path': 'model/quic/data/good/10_40_0.01_CNN_100_30_stream',
                'server': 100,
                'client': 30}
               ]}

DATA_3 = {
    'name': 'CNN B=10 E=10',
    'one': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_90_90_one',
             'server': 90,
             'client': 90},
            {'path': 'model/quic/data/good/10_10_0.01_CNN_70_70_one',
             'server': 70,
             'client': 70}
            ],
    'stream': [{'path': 'model/quic/data/good/10_10_0.01_CNN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_CNN_90_90_stream',
                'server': 90,
                'client': 90},
               #    {'path': 'model/quic/data/good/10_10_0.01_CNN_70_70_stream',
               #     'server': 70,
               #     'client': 70}
               ]}
DATA_4 = {
    'name': '2NN B=10 E=10',
    'one': [{'path': 'model/quic/data/good/10_10_0.01_2NN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_2NN_100_50_one',
             'server': 100,
             'client': 50},
            {'path': 'model/quic/data/good/10_10_0.01_2NN_100_30_one',
             'server': 100,
             'client': 30}
            ],
    'stream': [{'path': 'model/quic/data/good/10_10_0.01_2NN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_2NN_100_50_stream',
                'server': 100,
                'client': 50},
               {'path': 'model/quic/data/good/10_10_0.01_2NN_100_30_stream',
                'server': 100,
                'client': 30}
               ]}
DATA_5 = {
    'name': '2NN B=40 E=10',
    'one': [{'path': 'model/quic/data/good/10_40_0.01_2NN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_40_0.01_2NN_100_50_one',
             'server': 100,
             'client': 50},
            {'path': 'model/quic/data/good/10_40_0.01_2NN_100_30_one',
             'server': 100,
             'client': 30}
            ],
    'stream': [{'path': 'model/quic/data/good/10_40_0.01_2NN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_40_0.01_2NN_100_50_stream',
                'server': 100,
                'client': 50},
               {'path': 'model/quic/data/good/10_40_0.01_2NN_100_30_stream',
                'server': 100,
                'client': 30}
               ]}

DATA_6 = {
    'name': '2NN B=10 E=10',
    'one': [{'path': 'model/quic/data/good/10_10_0.01_2NN_100_100_one',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/good/10_10_0.01_2NN_90_90_one',
             'server': 90,
             'client': 90},
            {'path': 'model/quic/data/good/10_10_0.01_2NN_70_70_one',
             'server': 70,
             'client': 70}
            ],
    'stream': [{'path': 'model/quic/data/good/10_10_0.01_2NN_100_100_stream',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/good/10_10_0.01_2NN_90_90_stream',
                'server': 90,
                'client': 90},
               {'path': 'model/quic/data/good/10_10_0.01_2NN_70_70_stream',
                'server': 70,
                'client': 70}
               ]}

DATA_7 = {
    'name': 'CNN B=10 E=15',
    'one': [{'path': 'model/quic/data/history_CNN_200_100_100_one.pickle',
             'server': 100,
             'client': 100},
            {'path': 'model/quic/data/history_CNN_200_90_90_one.pickle',
             'server': 90,
             'client': 90},
            {'path': 'model/quic/data/history_CNN_200_80_80_one.pickle',
             'server': 80,
             'client': 80}
            ],
    'stream': [{'path': 'model/quic/data/history_CNN_200_100_100_streams.pickle',
                'server': 100,
                'client': 100},
               {'path': 'model/quic/data/history_CNN_200_90_90_streams.pickle',
                'server': 90,
                'client': 90},
               {'path': 'model/quic/data/history_CNN_200_80_80_streams.pickle',
                'server': 80,
                'client': 80}
               ]}

DATA_8 = {
    'name': '2NN B=10 E=20',
    'one': [
        # {'path': 'model/quic/data/good/10_20_0.01_2NN_100_50_one',
        #      'server': 50,
        #      'client': 100}
    ],
    'stream': [{'path': 'model/quic/data/good/10_20_0.01_2NN_100_50_stream',
                'server': 50,
                'client': 100},
               {'path': 'model/quic/data/good/10_20_0.01_2NN_100_100_stream',
                'server': 100,
                'client': 100}
               ]}
plot(DATA_8)


def plot2(data):
    linestyles = ['-', '--', ':', '-.']
    accuracy = {'one': [], 'stream': []}
    loss = {'one': [], 'stream': []}
    for element in data.get('one'):
        with open(element.get('path'), 'rb') as fp:
            dt = pickle.load(fp)
            acc = []
            los = []
            for point in dt:
                inf = point.get('accuracy')
                acc.append(inf[1])
                los.append(inf[0])
            accuracy['one'].append(acc)
            loss['one'].append(los)
    for element in data.get('stream'):
        with open(element.get('path'), 'rb') as fp:
            dt = pickle.load(fp)
            acc = []
            los = []
            for point in dt:
                inf = point.get('accuracy')
                acc.append(inf[1])
                los.append(inf[0])
            accuracy['stream'].append(acc)
            loss['stream'].append(los)

    for index, dt in enumerate(accuracy.get('one')):
        plt.plot(dt, linestyle=linestyles[index], color='royalblue', label='Sp={} Cp={}'.format(
            data.get('one')[index].get('server')/100, data.get('one')[index].get('client')/100))

    for index, dt in enumerate(accuracy.get('stream')):
        plt.plot(dt, linestyle=linestyles[index], color='orangered', label='Sp={} Cp={}'.format(
            data.get('stream')[index].get('server')/100, data.get('stream')[index].get('client')/100))
    plt.legend(loc='lower right', fontsize='small')
    plt.ylabel("Accuracy")
    plt.xlabel("Communication rounds")
    plt.title('{}'.format(data.get('name')))
    plt.show()

    for index, dt in enumerate(loss.get('one')):
        plt.plot(dt, linestyle=linestyles[index], color='royalblue', label='Sp={} Cp={}'.format(
            data.get('one')[index].get('server')/100, data.get('one')[index].get('client')/100))

    for index, dt in enumerate(loss.get('stream')):
        plt.plot(dt, linestyle=linestyles[index], color='orangered', label='Sp={} Cp={}'.format(
            data.get('stream')[index].get('server')/100, data.get('stream')[index].get('client')/100))
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel("Loss")
    plt.xlabel("Communication rounds")
    plt.title('{}'.format(data.get('name')))
    plt.show()


plot2(DATA_7)
