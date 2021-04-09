import pickle
import matplotlib.pyplot as plt
import numpy as np


# def send_time():
#     with open('model/quic/history.pickle', 'rb') as fp:
#         a = pickle.load(fp)
#         x = []
#         for index, element in enumerate(a):
#             if index == 10:
#                 print(element['client_info'])
#                 break
#             times = element['client_info']
#             for time in times:
#                 if time:
#                     b = time['send_time']
#                     if b:
#                         # start = min([c[0] for c in b if c])
#                         # end = max([c[1] for c in b if c])
#                         start = b[0]
#                         end = b[1]
#                         x.append(end-start)
#                     else:
#                         print(b)
#                 else:
#                     print(time)
#     with open('model/quic/history.pickle', 'rb') as fp:
#         a = pickle.load(fp)
#         y = []
#         for index, element in enumerate(a):
#             if index == 10:
#                 print(element['client_info'])
#                 break
#             times = element['client_info']
#             for time in times:
#                 if time:
#                     b = time['send_time']
#                     if b:
#                         start = min([c[0] for c in b if c])
#                         end = max([c[1] for c in b if c])
#                         y.append(end-start)
#                     else:
#                         print(b)
#                 else:
#                     print(time)

#         plt.scatter(range(len(x)), x)
#         plt.scatter(range(len(y)), y)
#         plt.show()


def loss():
    with open('model/quic/history.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[0])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)
    with open('model/quic/data/good/10_20_0.01_2NN_100_100_stream/history.pickle', 'rb') as fp:
        # with open('model/quic/data/5ep/history_CNN_200_100_100_stream.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[0])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)
    with open('model/quic/data/good/10_20_0.01_2NN_100_80_stream/history.pickle', 'rb') as fp:
        # with open('model/quic/data/5ep/history_CNN_200_100_50_one.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[0])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)
    plt.show()


def accuracy():
    with open('model/quic/history.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[1])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)

    with open('model/quic/data/good/10_20_0.01_2NN_100_100_stream/history.pickle', 'rb') as fp:
        # with open('model/quic/data/5ep/history_CNN_200_100_100_stream.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[1])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)
    with open('model/quic/data/good/10_20_0.01_2NN_100_80_stream/history.pickle', 'rb') as fp:
        # with open('model/quic/data/5ep/history_CNN_200_100_50_one.pickle', 'rb') as fp:
        a = pickle.load(fp)
        x = []
        for index, element in enumerate(a):
            acc = element['accuracy']
            x.append(acc[1])
        plt.plot(x)
        # a = np.polyfit(range(len(x)), x, 20)
        # pl = np.poly1d(a)
        # v = pl(range(len(x)))
        # v[0] = x[0]
        # plt.plot(range(len(x)), v)
    plt.show()


# send_time()
loss()
accuracy()

# with open('model/quic/EMNINST_TEST.pickle', 'rb') as fp:
#     DATA = pickle.load(fp)
#     CLIENTS = list(DATA.keys())
#     print(len(CLIENTS))
